# Method implementation

The original selection implementation in [methods/eug_core.py](../methods/eug_core.py) includes subset enumeration, centering, teacher-forced utility, exact and greedy selection, marginals and oracle regret. The [pinned model loader and batch scorer](../methods/eug_inference.py) and [generation routine](../methods/eug_generation.py) are available for use with separately obtained local model snapshots. The CPU check independently selects the canonical and reversed masks from all 64 saved utilities for the worked question.

From the release root, run `python -B check_methods.py`. The model loader requires previously obtained pinned snapshots and the original inference dependencies.
