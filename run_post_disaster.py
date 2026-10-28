"""Run the original post-disaster cooperative hierarchical framework."""

import os
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    scenario = root / "PostD" / "ours"
    sys.path.insert(0, str(scenario))
    sys.path.insert(0, str(root))
    os.chdir(scenario)
    runpy.run_path(str(scenario / "our_proposed.py"), run_name="__main__")
