"""Frozen greedy answer-generation primitive for EUG Phase 8."""

from __future__ import annotations

from typing import Any, Dict

from .eug_inference import LoadedGenerator, build_prefix_ids


GENERATION_VERSION = "EUG-GREEDY-ANSWER-v1"


def generate_greedy_answer(
    loaded: LoadedGenerator,
    *,
    question: str,
    evidence_context: str,
    max_new_tokens: int = 64,
    _test_allow_cpu: bool = False,
) -> Dict[str, Any]:
    """Generate only newly decoded answer tokens with the frozen settings."""
    import torch

    if max_new_tokens != 64:
        raise ValueError("frozen generated-answer validation requires max_new_tokens=64")
    prefix_ids = build_prefix_ids(loaded.tokenizer, question, evidence_context)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=loaded.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=loaded.device)
    if str(input_ids.device) != "cuda:0" and not _test_allow_cpu:
        raise RuntimeError("generated-answer input is not on cuda:0")
    eos_id = loaded.tokenizer.eos_token_id
    pad_id = loaded.tokenizer.pad_token_id
    if eos_id is None or pad_id is None:
        raise ValueError("frozen loader must provide EOS and PAD token IDs")
    with torch.inference_mode():
        output_ids = loaded.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=64,
            use_cache=True,
            eos_token_id=int(eos_id),
            pad_token_id=int(pad_id),
        )
    if output_ids.ndim != 2 or output_ids.shape[0] != 1:
        raise ValueError("generator returned an unexpected output shape")
    new_ids = output_ids[0, len(prefix_ids) :]
    text = loaded.tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return {
        "prediction": text,
        "generated_token_count": int(new_ids.numel()),
        "prefix_token_count": len(prefix_ids),
        "sequence_token_count": int(output_ids.shape[1]),
        "actual_input_device": str(input_ids.device),
    }
