# Which change in the evidence changed the answer?

From Evidence Scores to RAG Answers: Budget, Search, and Presentation

Wenzhang Du · Open Research Reports · v1.0.2 · 2026-11-22

How much of a gain comes from finding a better trio of passages, and how much from letting the model use fewer? Across 539 multi-hop questions, the option to use fewer passages accounts for 4.79 of Mistral’s 5.91 token-F1 points of improvement. In a separate 30-question comparison, changing the order of the same passages can reverse the apparent gain from choosing a new set.

[Original PDF](../paper/manuscript.pdf) · [Online report](https://dqswordman.github.io/technical-reports/rag-evidence-selection-paths/)

## One question, the same two passages, a different band

Which band did the musician who would later sing for Devilment found? Qwen answers “Cradle of Filth” with the two selected passages in their original order. Swapping those passages changes the answer to “Falling in Reverse.” If the selector can choose passages again under the new order, Qwen returns to “Cradle of Filth.” A comparison of the first and last answers would record no change.

![Three saved answers to the Devilment question across changes in order and membership.](../figures/design.png)

Figure 1. Read the saved Qwen answers from left to right. The first change reverses the same two passages. The second allows the selector to replace a passage under the reversed order.

## Separate the search for three passages from the option to use fewer

Start with six candidate passages. A greedy selector picks three, one at a time. An exhaustive selector scores every three-passage combination. Comparing the two measures what better search buys at the same size. The next step allows the exhaustive selector to choose zero, one or two passages as well. All selections use the reference answer’s likelihood. Token F1 measures the answer the generator actually produced.

The 539-question panel spans HotpotQA, 2WikiMultiHopQA and MuSiQue, with two pool constructions for each question. The analysis averages the two tracks within question before estimating uncertainty. A nested panel of 89 questions has all 64 canonical subset answers for Qwen and Gemma. On 30 matched questions, each of the 42 sets of size at most three has an answer in both orders.

## Most of Mistral’s gain comes from the option to use fewer passages

Mistral gains 1.11 token-F1 points when exhaustive search replaces greedy selection at exactly three passages. Allowing shorter selections adds 4.79 points (paired 95% interval 3.12 to 6.53). The total gain is 5.91 points before the components are rounded. For Qwen and Gemma, the intervals for allowing fewer passages span both gains and losses.

Paired changes in token F1 (points) on 539 questions. “Search” changes how a trio is chosen and “Budget” allows shorter sets. Each question contributes the mean of its two pools. Manuscript Table 2.

| Generator | Budget | Search | Total |
| --- | --- | --- | --- |
| Qwen | −1.46 | −0.35 | −1.81 |
| Gemma | +0.95 | +0.65 | +1.60 |
| Mistral | +4.79 | +1.11 | +5.91 |

[Source table](https://github.com/dqswordman/rag-evidence-selection-paths/blob/v1.0.1/data/analysis/A_bootstrap_summary.csv)

## A new set of passages can help in one order and hurt in the other

On 30 matched questions with exactly three passages, reversing Gemma’s original selection costs 20.56 F1 points. With that reversed order fixed, choosing a new set recovers 16.54 points. Compare the very same old and new sets in the original order and the new set loses 23.35 points. The effect attributed to changing membership therefore depends on the order used for the comparison.

An answer-blind cross-encoder offers a check on a different selection objective. On the 89 retrieval-built pools, greedy and exact three-passage search choose identical sets in 75 cases. The observed answer changes still differ by generator. The size of the pool, the short-answer format and the single reversal bound the interpretation.

## Trace an answer back to its selected passages

The CPU replay checks 9,702 records that link a selected set to its generated answer, then recomputes 279 paired statistics. Passage IDs and text hashes identify the inputs in the source benchmarks. Use those benchmarks for the original passage text.

The original selection code independently finds masks 5 and 33 on the two saved utility surfaces for the Figure 1 question. The archived model loader, likelihood scorer and greedy generation routines are provided alongside it.

```sh
python -B verify_manifest.py
python -B reproduce.py --out replay
python -B check_methods.py
```

## Citation and materials

[Original manuscript · PDF](https://dqswordman.github.io/assets/reports/rag-evidence-selection-paths/manuscript.pdf) · [Frozen release and checksums](https://github.com/dqswordman/rag-evidence-selection-paths/releases/tag/v1.0.1) · [Result map and reproduction scope](https://github.com/dqswordman/rag-evidence-selection-paths/blob/v1.0.1/docs/reproduction.md) · [Interactive evidence explorer](https://huggingface.co/spaces/DQSWORDMAN/rag-evidence-selection-paths) · [Method implementation and worked checks](https://github.com/dqswordman/rag-evidence-selection-paths/blob/v1.0.1/docs/methods.md)
