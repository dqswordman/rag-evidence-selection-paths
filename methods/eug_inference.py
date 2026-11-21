"""Frozen generator-loading and teacher-forced scoring primitives for EUG.

This module contains execution mechanics only.  It never chooses a scientific
threshold, candidate universe, query split, or outcome class.  All generator
snapshots are resolved by exact commit and loaded locally on CUDA without
quantization or CPU offload.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROMPT_VERSION = "EUG-TEACHER-FORCED-PROMPT-v1"
SYSTEM_PROMPT = (
    "You are a question-answering system. Use the supplied evidence when it is "
    "available. Return only the shortest correct answer span and do not explain."
)
NO_EVIDENCE_MARKER = "[NO EVIDENCE]"
TARGET_PREFIX = " "


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    revision: str
    architecture: str
    preferred_dtype: str = "bfloat16"


MODEL_SPECS: Mapping[str, ModelSpec] = {
    "qwen": ModelSpec(
        key="qwen",
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        architecture="causal_lm",
    ),
    "gemma": ModelSpec(
        key="gemma",
        repo_id="google/gemma-3-4b-it",
        revision="093f9f388b31de276ce2de164bdc2081324b9767",
        architecture="image_text_to_text",
    ),
    "mistral": ModelSpec(
        key="mistral",
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
        architecture="causal_lm",
    ),
}


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def huggingface_hub_cache() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def snapshot_path(spec: ModelSpec) -> Path:
    repository_directory = "models--" + spec.repo_id.replace("/", "--")
    path = huggingface_hub_cache() / repository_directory / "snapshots" / spec.revision
    if not path.is_dir():
        raise FileNotFoundError(
            f"pinned snapshot is absent for {spec.repo_id}@{spec.revision}: {path}"
        )
    return path


def _weight_files_from_index(path: Path) -> Tuple[str, ...]:
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid model index: {index_path}")
        return tuple(sorted({str(name) for name in weight_map.values()}))
    single = path / "model.safetensors"
    if single.is_file():
        return (single.name,)
    raise FileNotFoundError(f"no safetensors index or single weight file in {path}")


def audit_snapshot(spec: ModelSpec) -> Dict[str, Any]:
    """Validate the pinned local snapshot and return a compact content manifest."""
    path = snapshot_path(spec)
    required_small = ("config.json", "tokenizer_config.json")
    for name in required_small:
        if not (path / name).is_file():
            raise FileNotFoundError(f"missing required snapshot file: {path / name}")

    weight_records = []
    for name in _weight_files_from_index(path):
        linked = path / name
        if not linked.is_file():
            raise FileNotFoundError(f"missing indexed weight shard: {linked}")
        resolved = linked.resolve(strict=True)
        if resolved.name.endswith(".incomplete"):
            raise RuntimeError(f"weight shard remains incomplete: {resolved}")
        # Hugging Face's cache stores LFS objects by their SHA-256 object ID.
        # Re-read every shard here: a transport process interrupted during the
        # final move can otherwise leave a correctly named but corrupt blob.
        content_address = resolved.name
        if len(content_address) != 64 or any(
            character not in "0123456789abcdef" for character in content_address.lower()
        ):
            content_address = None
        verified_sha256 = sha256_file(resolved)
        if content_address is not None and verified_sha256 != content_address:
            raise RuntimeError(
                "weight content hash mismatch for "
                f"{spec.repo_id}/{name}: expected {content_address}, "
                f"observed {verified_sha256}"
            )
        weight_records.append(
            {
                "snapshot_name": name,
                "resolved_blob": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "hub_content_sha256": content_address,
                "verified_sha256": verified_sha256,
            }
        )

    small_records = []
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
    ):
        candidate = path / name
        if candidate.is_file():
            small_records.append(
                {
                    "name": name,
                    "size_bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )

    return {
        "model_key": spec.key,
        "repo_id": spec.repo_id,
        "model_revision": spec.revision,
        "tokenizer_revision": spec.revision,
        "architecture": spec.architecture,
        "snapshot_path": str(path),
        "weight_shards": weight_records,
        "weight_bytes_total": sum(item["size_bytes"] for item in weight_records),
        "small_files": small_records,
        "incomplete_file_count": len(list(path.parent.parent.rglob("*.incomplete"))),
    }


def render_user_prompt(question: str, evidence_context: str) -> str:
    question_clean = str(question).strip()
    if not question_clean:
        raise ValueError("question cannot be empty")
    evidence = str(evidence_context)
    if not evidence:
        evidence = NO_EVIDENCE_MARKER
    return (
        "Question:\n"
        + question_clean
        + "\n\nEvidence:\n"
        + evidence
        + "\n\nReturn only the answer."
    )


def prompt_identity(question: str) -> str:
    """Hash prompt semantics excluding subset evidence, which has its own hash."""
    return stable_json_hash(
        {
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "user_template": (
                "Question:\n{question}\n\nEvidence:\n{evidence_or_marker}"
                "\n\nReturn only the answer."
            ),
            "question": str(question).strip(),
            "no_evidence_marker": NO_EVIDENCE_MARKER,
            "target_prefix": TARGET_PREFIX,
        }
    )


def build_prefix_ids(tokenizer: Any, question: str, evidence_context: str) -> List[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(question, evidence_context)},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("unexpected batched chat-template output")
        ids = ids[0]
    result = [int(token) for token in ids]
    if not result:
        raise ValueError("chat template produced an empty prefix")
    return result


def build_target_ids(tokenizer: Any, canonical_answer: str) -> Tuple[str, List[int]]:
    answer = str(canonical_answer).strip()
    if not answer:
        raise ValueError("canonical answer cannot be empty")
    target_text = TARGET_PREFIX + answer
    ids = tokenizer.encode(target_text, add_special_tokens=False)
    result = [int(token) for token in ids]
    if not result:
        raise ValueError("canonical answer produced no target tokens")
    return target_text, result


@dataclass(frozen=True)
class TeacherForcedExample:
    prefix_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]
    target_text: str
    prompt_hash: str
    serialization_hash: str


def build_teacher_forced_example(
    tokenizer: Any,
    *,
    question: str,
    evidence_context: str,
    canonical_answer: str,
) -> TeacherForcedExample:
    prefix_ids = build_prefix_ids(tokenizer, question, evidence_context)
    target_text, target_ids = build_target_ids(tokenizer, canonical_answer)
    return TeacherForcedExample(
        prefix_ids=tuple(prefix_ids),
        target_ids=tuple(target_ids),
        target_text=target_text,
        prompt_hash=prompt_identity(question),
        serialization_hash=hashlib.sha256(
            str(evidence_context).encode("utf-8")
        ).hexdigest(),
    )


@dataclass
class LoadedGenerator:
    spec: ModelSpec
    model: Any
    tokenizer: Any
    device: Any
    floating_dtypes: Tuple[str, ...]
    parameter_devices: Tuple[str, ...]


def load_generator(spec: ModelSpec, *, device_name: str = "cuda:0") -> LoadedGenerator:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; silent CPU fallback is forbidden")
    requested = torch.device(device_name)
    if requested.type != "cuda":
        raise ValueError("EUG claim-bearing generators must be requested on CUDA")

    path = snapshot_path(spec)
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, revision=spec.revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"{spec.repo_id} has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs = {
        "local_files_only": True,
        "revision": spec.revision,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": {"": device_name},
    }
    if spec.architecture == "image_text_to_text":
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(str(path), **kwargs)
    elif spec.architecture == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    else:
        raise ValueError(f"unsupported architecture: {spec.architecture}")
    model.eval()

    parameter_devices = tuple(sorted({str(parameter.device) for parameter in model.parameters()}))
    if not parameter_devices or any(not item.startswith("cuda") for item in parameter_devices):
        raise RuntimeError(
            f"silent CPU/offload placement detected for {spec.repo_id}: {parameter_devices}"
        )
    floating_dtypes = tuple(
        sorted(
            {
                str(parameter.dtype).replace("torch.", "")
                for parameter in model.parameters()
                if parameter.is_floating_point()
            }
        )
    )
    if "bfloat16" not in floating_dtypes:
        raise RuntimeError(
            f"primary bfloat16 weights not observed for {spec.repo_id}: {floating_dtypes}"
        )
    return LoadedGenerator(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=requested,
        floating_dtypes=floating_dtypes,
        parameter_devices=parameter_devices,
    )


def score_teacher_forced_batch(
    loaded: LoadedGenerator,
    examples: Sequence[TeacherForcedExample],
) -> List[Dict[str, Any]]:
    """Score a deterministic right-padded batch of fixed targets."""
    import torch

    if not examples:
        raise ValueError("cannot score an empty batch")
    sequences = [list(item.prefix_ids) + list(item.target_ids) for item in examples]
    lengths = [len(item) for item in sequences]
    max_length = max(lengths)
    pad_id = int(loaded.tokenizer.pad_token_id)
    input_ids = torch.full(
        (len(sequences), max_length),
        pad_id,
        dtype=torch.long,
        device=loaded.device,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_length),
        dtype=torch.long,
        device=loaded.device,
    )
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(
            sequence, dtype=torch.long, device=loaded.device
        )
        attention_mask[row, : len(sequence)] = 1
    if str(input_ids.device) != str(loaded.device):
        raise RuntimeError("actual input device differs from requested CUDA device")

    with torch.inference_mode():
        output = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    logits = output.logits
    if logits.ndim != 3 or logits.shape[0] != len(sequences):
        raise ValueError("generator returned unexpected logits shape")

    results: List[Dict[str, Any]] = []
    for row, example in enumerate(examples):
        target_start = len(example.prefix_ids)
        sequence_length = lengths[row]
        prediction_logits = logits[row, target_start - 1 : sequence_length - 1, :]
        target_ids = input_ids[row, target_start:sequence_length]
        if prediction_logits.shape[0] != len(example.target_ids):
            raise AssertionError("target/logit alignment failure")
        token_logprobs = torch.log_softmax(prediction_logits.float(), dim=-1)
        gathered = token_logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        if not torch.isfinite(gathered).all():
            raise FloatingPointError("non-finite target token log probability")
        results.append(
            {
                "mean_logprob": float(gathered.mean().item()),
                "target_token_count": int(gathered.numel()),
                "prefix_token_count": int(target_start),
                "sequence_token_count": int(sequence_length),
                "token_logprobs": [float(value) for value in gathered.cpu().tolist()],
                "actual_input_device": str(input_ids.device),
            }
        )
    del logits, output, input_ids, attention_mask
    return results


def unload_generator(loaded: LoadedGenerator) -> None:
    import gc
    import torch

    del loaded.model
    del loaded.tokenizer
    gc.collect()
    torch.cuda.empty_cache()
