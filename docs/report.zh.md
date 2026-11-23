# 是哪一步证据选择改变了回答？

From Evidence Scores to RAG Answers: Budget, Search, and Presentation

Wenzhang Du · 开放研究报告 · v1.0.2 · 2026-11-22

给模型更好的三段证据，和允许它少用几段，哪一步更有用？在 539 道多跳问题上，Mistral 的答案词元 F1 提高了 5.91 分，其中 4.79 分来自放宽段数限制。另取 30 道问题做交叉比较，同一组新证据换个排列顺序，得分变化的方向就可能翻转。

[原版 PDF](../paper/manuscript.pdf) · [在线报告](https://dqswordman.github.io/zh/technical-reports/rag-evidence-selection-paths/)

## 同一道题、同两段证据，模型却答出了另一支乐队

后来担任 Devilment 主唱的音乐人成立过哪支乐队？按原顺序读两段选中的证据，Qwen 答“Cradle of Filth”。把这两段对调，它改答“Falling in Reverse”。再允许选择器按新顺序重新选段，它又答回“Cradle of Filth”。只看首尾，会把这两次变化都漏掉。

![Devilment 问题在证据反序和重选前后的三次归档回答。](../figures/design.png)

图 1。从左到右是 Qwen 实际生成的三个答案。第一步只对调原来的两段，第二步才允许选择器在反序条件下更换段落。

## 先比较怎样选三段，再比较能否少选

每道题给定六段候选。贪心法逐段选满三段，穷举法给每一种三段组合打分。先比较这两种选法，就能估计段数相同时搜索有多少用。然后只放宽段数，允许穷举法也选择零段、一段或两段。选段依据是参考答案的似然，最终得分则用模型实际回答的词元 F1 计算。

539 道题来自三个多跳问答基准，每题有两种候选池构造。统计先在同一问题内平均两条轨道，再估计不确定性。嵌套的 89 道题保存了 Qwen 和 Gemma 对全部 64 个子集的常规顺序回答。另有 30 道配对题，最多三段的 42 种组合均有两种顺序的回答。

## Mistral 的提升主要来自可以少选几段

固定选三段时，用穷举替代贪心，Mistral 提高 1.11 个词元 F1 分。允许少选后再提高 4.79 分，配对 95% 区间为 3.12 至 6.53 分。总增益为 5.91 分，各项数值分别四舍五入到小数点后两位。对 Qwen 和 Gemma，放宽段数的区间同时覆盖上升和下降。

539 道题的配对词元 F1 差值，单位为分。搜索项改变三段组合的选法，预算项允许选择更短的组合。每道题先取两个候选池的均值。见原稿表 2。

| 生成器 | 预算 | 搜索 | 总变化 |
| --- | --- | --- | --- |
| Qwen | −1.46 | −0.35 | −1.81 |
| Gemma | +0.95 | +0.65 | +1.60 |
| Mistral | +4.79 | +1.11 | +5.91 |

[数据来源](https://github.com/dqswordman/rag-evidence-selection-paths/blob/v1.0.1/data/analysis/A_bootstrap_summary.csv)

## 同一组新段落带来的收益取决于排列顺序

在固定三段的 30 道配对题上，只把 Gemma 原选的段落反序，答案 F1 就下降 20.56 分。保持反序再重新选段，可挽回 16.54 分。但把同一对新旧组合都按原顺序呈现，新组合反而低 23.35 分。判断换段是否有益，必须同时说明两组段落是按什么顺序读的。

另用无需参考答案的交叉编码器检查不同选择目标。在 89 个检索构造的池中，贪心与精确搜索有 75 次选中相同三段，实际答案变化仍因生成器而异。候选池大小、短答案形式及单一反序限定了结论的范围。

## 从归档回答追查当时选中的段落

CPU 回放先核对 9,702 条选段与生成回答的对应记录，再复算 279 项配对统计。可凭段落编号和文本哈希回到原始基准查阅正文。

用原始选择代码在图 1 问题的两个效用曲面上独立搜索，会分别得到掩码 5 和 33。包内同时提供当时的模型加载、似然评分与贪心生成实现。

```sh
python -B verify_manifest.py
python -B reproduce.py --out replay
python -B check_methods.py
```

## 引用与材料

[原版论文 · PDF](https://dqswordman.github.io/assets/reports/rag-evidence-selection-paths/manuscript.pdf) · [固定版本与校验值](https://github.com/dqswordman/rag-evidence-selection-paths/releases/tag/v1.0.1) · [结果映射与复现范围](https://github.com/dqswordman/rag-evidence-selection-paths/blob/v1.0.1/docs/reproduction.md) · [交互式证据浏览](https://huggingface.co/spaces/DQSWORDMAN/rag-evidence-selection-paths) · [方法源码与核验说明](https://github.com/dqswordman/rag-evidence-selection-paths/blob/v1.0.1/docs/methods.md)
