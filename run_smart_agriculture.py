"""Run the original smart-agriculture PG-GA script."""

import os
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    scenario = root / "SmartA" / "ours"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(scenario))
    os.chdir(scenario)
    runpy.run_path(str(scenario / "our_proposed.py"), run_name="__main__")
