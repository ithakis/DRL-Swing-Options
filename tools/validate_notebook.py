"""Execute every code cell of the findings notebook in order and report errors."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Jupyter Notebooks" / "7: Phase 1 Findings - Semi-Analytical Kernel.ipynb"


def main() -> int:
    nb = json.loads(NB.read_text())
    glob = {"math": math, "np": np, "pd": pd, "sps": sps, "json": json, "__name__": "__main__"}
    # Headless matplotlib backend so plt.show() doesn't pop windows
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    glob["plt"] = plt
    errors = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            exec(src, glob)
            plt.close("all")
            print(f"Cell {i}: OK")
        except Exception as e:
            print(f"Cell {i}: ERROR {type(e).__name__}: {e}")
            print(src[:400])
            errors += 1
    print(f"=== Total errors: {errors} ===")
    return errors


if __name__ == "__main__":
    sys.exit(main())
