"""Replay the stored RAG decisions, answer scores, intervals and Figure 1."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    env = dict(os.environ, REPLAY_OUT=str(output), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for script in ["analyze_v2.py", "recompute_reported_comparisons.py", "replay_figure1.py"]:
        result = subprocess.run([sys.executable, "-B", str(ROOT/"scripts"/script)], env=env, check=True, capture_output=True)
        (output/(Path(script).stem+".txt")).write_bytes(result.stdout)
    raw = json.loads((output/"v2_raw_validation.json").read_text())
    archived = json.loads((ROOT/"reference/v2_raw_validation.json").read_text())
    for key, value in archived.items():
        if "error" in key:
            assert abs(raw[key]-value) < 1e-10, key
        else:
            assert raw[key] == value, key
    checks = json.loads((output/"recomputed_comparisons.json").read_text())
    assert sum(row["checked_statistics"] for row in checks["comparisons"]) == 279
    case = json.loads((output/"replay_figure1.txt").read_text())
    assert [row["answer"] for row in case["states"]] == ["Cradle of Filth","Falling in Reverse","Cradle of Filth"]
    print(json.dumps({"status":"PASS","paired_statistics":279,"bootstrap_draws":20000,"case":case["states"],"checks":raw},indent=2))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args().out)
