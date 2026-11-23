# Method implementation

The [selection code](../methods/eug_core.py) enumerates the 64 subsets of six passages and scores a selection by its gap to the best feasible subset. The CPU check uses two saved score arrays to recover the selected passages before and after reversing their order. The [model loader and batch scorer](../methods/eug_inference.py) and [generation routine](../methods/eug_generation.py) support new inference with the separately obtained pinned checkpoints.

From the release root, run `python -B check_methods.py`. The model loader requires previously obtained pinned snapshots and the original inference dependencies.
