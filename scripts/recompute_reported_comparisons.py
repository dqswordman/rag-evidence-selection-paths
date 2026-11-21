"""Recompute paired answer contrasts from saved question-level observations.

No model execution is required. Reported F1/EM are on a 0--1 scale here;
the manuscript multiplies these values by 100.
"""
from pathlib import Path
from collections import defaultdict
import csv
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from r2_stats import QuestionBootstrap, summarize_tracks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "analysis"


def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    bootstrap = QuestionBootstrap(repetitions=20000, seed=20261107)
    all_results = []
    comparisons = []

    def compare(label, rows, source, keys):
        expected = {tuple(r[k] for k in keys): r for r in source}
        errors = []
        for row in rows:
            key = tuple(str(row[k]) for k in keys)
            reference = expected[key]
            error = max(abs(float(row[k]) - float(reference[k])) for k in ("mean", "ci_low", "ci_high"))
            errors.append(error)
            if error > 1e-10:
                raise AssertionError((label, key, error))
            all_results.append(dict(experiment=label, **row))
        comparisons.append(dict(experiment=label, checked_statistics=len(rows), maximum_absolute_error=max(errors)))

    a = read("A_per_question.csv")
    for r in a:
        for metric in ("f1", "em"):
            assert abs(float(r["C_" + metric]) + float(r["J_" + metric]) - float(r["total_" + metric])) < 1e-12
    a_all = a + [dict(r, cohort="parent539") for r in a]
    a_summary = summarize_tracks(a_all, ("cohort", "model"), ("C_f1", "J_f1", "total_f1"), bootstrap)
    compare("A", a_summary, read("A_bootstrap_summary.csv"), ("cohort", "model", "track", "metric"))

    b = read("B_fixed_mask_and_reselection.csv")
    for r in b:
        assert abs(float(r["fixed_f1"]) + float(r["reselected_f1"]) - float(r["total_f1"])) < 1e-12
    b_summary = summarize_tracks(b, ("model", "role"), ("fixed_f1", "reselected_f1", "total_f1"), bootstrap)
    compare("B", b_summary, [r for r in read("B_effects.csv") if r["role"]], ("model", "role", "track", "metric"))

    c = read("C_policy_per_question.csv")
    c_summary = summarize_tracks(c, ("model", "policy"), ("f1", "em"), bootstrap)
    compare("C_policy", c_summary, [r for r in read("C_policy_effects.csv") if r["policy"]],
            ("model", "policy", "track", "metric"))

    paired = read("C_paired_per_question.csv")
    counts = []
    for model in ("qwen", "gemma"):
        rows = [r for r in paired if r["model"] == model and r["track"] == "B" and r["contrast"] == "J_R"]
        assert len(rows) == 89
        counts.append(dict(model=model, questions=len(rows),
                           changed_sets=sum(r["mask_change"] == "True" for r in rows),
                           answer_changes=sum(r["answer_change"] == "True" for r in rows),
                           f1_increases=sum(float(r["F1_change"]) > 0 for r in rows),
                           f1_decreases=sum(float(r["F1_change"]) < 0 for r in rows)))
    result = dict(bootstrap_repetitions=20000, seed=20261107,
                  comparisons=comparisons, cross_encoder_search_counts=counts)
    output = Path(os.environ.get("REPLAY_OUT", ROOT / "replay"))
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "recomputed_comparisons.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "recomputed_statistics.json").write_text(
        json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
