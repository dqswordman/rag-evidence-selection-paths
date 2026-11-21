# Sources and attribution

HotpotQA by Yang et al. supplies part of the multi-hop question panel and declares CC BY-SA 4.0. MuSiQue by Trivedi et al. declares CC BY 4.0. 2WikiMultiHopQA by Ho et al. publishes its repository under Apache-2.0. Wikipedia and Wikidata material also retains its original terms. Only the question, reference answers and passage metadata needed to identify the frozen evidence are redistributed here. The benchmark identifiers in `data/freeze/questions.jsonl.gz` and the per-pool passage digests preserve source attribution. Token-F1 and exact match are recomputed from the archived generations.

[HotpotQA and its license](https://hotpotqa.github.io/) · [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) · [2Wiki repository license](licenses/2Wiki-Apache-2.0.txt) · [MuSiQue](https://github.com/stonybrooknlp/musique) · [Wikipedia reuse](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)

Recorded answers were produced by Qwen2.5-7B-Instruct, Gemma 3 4B IT and Mistral-7B-Instruct-v0.3. The pinned checkpoints and generation parameters are listed in [the design file](data/design.json). Upstream models, their generated material and dependencies keep their respective terms.
