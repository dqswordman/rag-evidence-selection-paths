# Reproducing the reported results

**From Evidence Scores to RAG Answers: Budget, Search, and Presentation** · Wenzhang Du · v1.0.1

To reproduce the budget and search comparison, begin with the 539 question-level pairs and their saved answers. To inspect the effect of order, use the separate 30-question panel. The commands below produce the summary tables on CPU.

```sh
python -m pip install -r requirements.txt
python -B verify_manifest.py
python -B reproduce.py --out replay
```

Run the commands at the root of the extracted archive with Python 3.12. CPU verification uses NumPy 2.5.3, the version used to check this release. The manifest verifies the publication files before replay. A successful run prints `PASS` and writes its derived tables or summary into the empty output directory.

| Study result | Files | Unit and operation |
| --- | --- | --- |
| Budget and search · Table 2 | [Input](../data/analysis/A_role_answers.csv) · [Reference](../reference/budget_search_539.csv) | 539 questions, two candidate pools per question, three generators. Average tracks inside each question before drawing the paired intervals. |
| Answer surface and coverage | [Input](../data/freeze/old_answers.jsonl.gz) · [Reference](../reference/coverage_per_pool.csv) | On 89 questions, both pools and both generators have all 64 canonical subset answers. Saved utility cubes support exhaustive selection. |
| Order and reselection · Figure 1 | [Input](../data/analysis/B_answers_42.jsonl.gz) · [Reference](../scripts/replay_figure1.py) | 30 matched questions with answers for 42 feasible subsets in both orders. The worked example reselects a mask and scores each saved answer. |

## Evidence boundaries

The public inputs preserve benchmark question IDs, passages’ titles, order, retrieval ranks and text digests. The original benchmark repositories supply passage bodies. Answer strings are recorded generations. New model inference needs the original sources and checkpoints. The public bootstrap uses 20,000 draws.

Selections use the reference answer’s likelihood. The order comparison tests one reversal per selected set on 30 matched short-answer questions. For a single worked case, run `python -B scripts/replay_figure1.py`. [Original manuscript](../paper/manuscript.pdf) · [English report](report.en.md) · [中文报告](report.zh.md) · [Third-party terms](../THIRD_PARTY_NOTICES.md)

## Method implementation

The original selection implementation in [methods/eug_core.py](../methods/eug_core.py) includes subset enumeration, centering, teacher-forced utility, exact and greedy selection, marginals and oracle regret. The [pinned model loader and batch scorer](../methods/eug_inference.py) and [generation routine](../methods/eug_generation.py) are available for use with separately obtained local model snapshots. The CPU check independently selects the canonical and reversed masks from all 64 saved utilities for the worked question.

Run `python -B check_methods.py`. [Method guide](methods.md)
