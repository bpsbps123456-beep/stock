"""Regenerate every grid scenario and analysis log for the strategy lab.

    py run_all.py            # reuse cached prices/panels
    py run_all.py --refresh  # re-read snapshots and re-download prices first
"""

import subprocess
import sys
from pathlib import Path

import data as D

HERE = Path(__file__).resolve().parent
LOGS = D.CACHE / "logs"
GRIDS = [[], ["--cost", "0", "--tag", "_c0"], ["--cost", "0.0015", "--tag", "_c15"],
         ["--cost", "0.0025", "--tag", "_c25"], ["--exec", "open", "--tag", "_open"],
         ["--delay", "1", "--tag", "_d1"], ["--delay", "1", "--cost", "0.003", "--tag", "_d1c30"],
         ["--exclude", "SNDK", "--tag", "_xSNDK"],
         ["--exclude", "SNDK,WDC,STX,MU,SKHY", "--tag", "_xMEM"]]
STAGES = ["analyze1", "analyze2", "analyze3", "analyze4", "analyze5", "analyze6", "analyze7",
          "analyze8", "analyze9", "regime", "finalists"]


def run(args, log=None):
    out = subprocess.run([sys.executable, "-W", "ignore", *args], cwd=HERE, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise SystemExit(f"{args} failed:\n{out.stderr[-2000:]}")
    if log:
        (LOGS / log).write_text(out.stdout, encoding="utf-8")
    return out.stdout


if __name__ == "__main__":
    LOGS.mkdir(parents=True, exist_ok=True)
    if "--refresh" in sys.argv:
        print(run(["data.py", "--refresh"]).strip())
    for g in GRIDS:
        print(run(["grid.py", *g]).strip().splitlines()[-1])
    for stage in STAGES:
        run([f"{stage}.py"], log=f"{stage}.txt")
        print(f"{stage}: logs/{stage}.txt")
