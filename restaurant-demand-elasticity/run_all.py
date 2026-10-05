"""Run the full pipeline: download -> build dataset -> estimate -> tables/figures."""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

skip_download = "--no-download" in sys.argv
steps = ["01_download_data.py", "02_build_dataset.py", "03_estimate_models.py"]
for step in steps:
    if skip_download and step.startswith("01"):
        print("== skipping download, using files already in data/raw/")
        continue
    print(f"== {step}")
    runpy.run_path(str(SRC / step), run_name="__main__")
print("Done. Tables in output/tables/, figures in output/figures/.")
