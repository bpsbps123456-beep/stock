"""mtfaudit 5: 일봉 16:00 같은 시각 사용의 크기. 독립 재계산에서 일봉 반영을 16:00 → 16:05(다음 b봉부터)로 늦춰 zD 판정 칸을 다시 잰다.

16:00에 끝나는 b봉은 그날 정규장 종가(16:00 직후 확정)를 쓰고 다음 봉 시가(16:00)에 체결한다. 실제 판은 16:00 ET가 한국 05시라 체결이 19:00 ET로 미뤄지므로
영향이 0이어야 하고, 이상적 판(모든 봉 체결)만 바뀔 수 있다.

py code/mtfaudit_daily1600.py → results/mtfaudit_daily1600.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtfaudit_core as M  # noqa: E402

SYMS = ["MU", "BE", "NBIS"]
CELLS = [(20, "zD_ER", "A0"), (90, "zD_EXT", "A0"), (40, "zD_AGREE", "A0"), (240, "zD_ADX", "A2"), (10, "zD_AGREE", "A2")]
BOARDS = {"real": (0.001, False), "real0": (0.0, False), "ideal": (0.0, True)}
_daily = M.daily


def late_daily(raw):
    d = _daily(raw)
    d["end"] = d["end"] + 5 * M.MIN
    return d


def main():
    rows = []
    for sym in SYMS:
        raw = M.read_raw(sym)
        for b, clf, act in CELLS:
            res = {}
            for lab, fn in (("16:00", _daily), ("16:05", late_daily)):
                M.daily = fn
                P = M.build(raw, b)
                res[lab] = P
            M.daily = _daily
            ch = int((res["16:00"]["flags"][clf] != res["16:05"]["flags"][clf]).sum())
            for bd, (cost, ideal) in BOARDS.items():
                a, _ = M.metrics(res["16:00"], M.target(res["16:00"]["s"], res["16:00"]["flags"][clf], act), cost, ideal)
                c, _ = M.metrics(res["16:05"], M.target(res["16:05"]["s"], res["16:05"]["flags"][clf], act), cost, ideal)
                rows.append(dict(sym=sym, b=b, clf=clf, action=act, board=bd, flag_changed=ch,
                                 d_ret_sel=c["ret_sel"] - a["ret_sel"], d_ret_conf=c["ret_conf"] - a["ret_conf"]))
    r = pd.DataFrame(rows)
    r.to_csv(M.HERE / "results" / "mtfaudit_daily1600.csv", index=False)
    print(r.groupby("board")[["d_ret_sel", "d_ret_conf"]].agg(lambda x: np.abs(x).max()).to_string())
    print(r.flag_changed.describe())


if __name__ == "__main__":
    main()
