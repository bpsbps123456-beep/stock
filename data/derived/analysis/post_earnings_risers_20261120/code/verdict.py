"""사전 등록 M1~M6과 대조(EPS 서프라이즈)의 판정표 → results/prereg_verdict.csv"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
M = [("M1", "kr_buy_share_pre", -1), ("M2", "kr_ever_before", -1), ("M3", "occ_call_ab", -1),
     ("M4", "px_vsurge5", -1), ("M5", "px_pre20_x", -1), ("M6", "a_h01_wiki", -1), ("대조", "obv_eps_surprise", +1)]


def main() -> None:
    rows = []
    for tag in ("", "_broad"):
        ic = pd.read_csv(BASE / f"results/ic{tag}.csv")
        for code, feat, sign in M:
            for y in ("d20_x", "r0_x"):
                s = ic[(ic.feature == feat) & (ic.outcome == y)].set_index("half")
                h1, h2 = s.stat.get("H1"), s.stat.get("H2")
                kind = s.kind.iloc[0]
                thr = 8 if kind == "bool" else 0.10
                ok = h1 is not None and h2 is not None and sign * h1 > 0 and sign * h2 >= thr
                rows.append({"sample": "보유 당시" if tag == "" else "2026 보유 전체", "code": code, "feature": feat,
                             "outcome": y, "kind": kind, "H1": round(h1, 3), "H2": round(h2, 3),
                             "n_H1": int(s.n.get("H1", 0)), "n_H2": int(s.n.get("H2", 0)),
                             "pass": bool(ok) if code != "대조" else None})
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "results/prereg_verdict.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
