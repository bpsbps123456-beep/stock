"""◆ 바닥 공백 보완안 비교 (2026-10-05). validate.py와 같은 시점·진입·비교 기준을 쓴다.

지수가 바닥을 만드는 날에는 '저점 이후'가 모두 0이라 ◆가 빈다. 아래 안을 미리 정해 비교한다.
  지금 규칙        : 저점 이후로 조건①② 판단, 비면 현금
  보류안           : 지금 규칙과 같되, 비면 AI 전체(유니버스 균등)를 그대로 든다
  대체안(N일)      : 기준 저점이 최근 N거래일 안이면 조건①②를 '최근 21거래일 수익'으로 판단
  1개월안          : 항상 최근 21거래일 수익으로 판단
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate as V  # noqa: E402
import ai_ath_gap  # noqa: E402
from desk import giant  # noqa: E402

OUT = Path(__file__).resolve().parent


def feats_at(F, uni, t, anchor):
    out = {}
    for s in uni:
        if s not in F:
            continue
        df = F[s]
        i = df.index.searchsorted(pd.Timestamp(t), side="right") - 1
        if i < 0:
            continue
        now = df.iloc[i]
        if now["n"] < 60 or pd.isna(now["ud"]):
            continue
        low = V.asof(df, anchor["low_date"])
        out[s] = {"since": float(now["c"] / low["c"] - 1) if low is not None else None,
                  "r21": float(now["c"] / df["c"].iloc[i - 21] - 1) if i >= 21 else None,
                  "gap": float(now["gap"]), "ud": float(now["ud"])}
    return out


def diamonds(feats, sub_of, lens):
    vals = {s: f for s, f in feats.items() if f[lens] is not None}
    if not vals:
        return []
    uni_med = median(f[lens] for f in vals.values())
    by_sub = {}
    for s in vals:
        by_sub.setdefault(sub_of[s], []).append(s)
    picks = []
    for members in by_sub.values():
        med = median(vals[s][lens] for s in members)
        for rank, s in enumerate(sorted(members, key=lambda x: -vals[x][lens]), 1):
            f = vals[s]
            if med > uni_med and rank <= giant.LEADERS_PER_SUB and f["gap"] >= ai_ath_gap.NEAR_PCT / 100 and f["ud"] >= 1.0:
                picks.append(s)
    return picks


def main():
    rows, uni, _, bars = V.load()
    sub_of = {s: (g, sub) for g, sub, s in rows}
    F = V.features(bars)
    cal = list(F["SOXX"].index[F["SOXX"].index >= "2025-06-01"])
    soxx = F["SOXX"]["c"]
    breadth = pd.DataFrame(json.loads((OUT / "validation.json").read_text(encoding="utf-8"))["breadth"]).set_index("date")
    first = next(i for i, d in enumerate(cal) if d >= pd.Timestamp(V.EVAL_START))
    recs = []
    for ti in range(first, len(cal) - 1, V.STEP):
        t = cal[ti]
        anchor = giant.find_anchor(soxx[soxx.index <= t])
        age = int((soxx.index > pd.Timestamp(anchor["low_date"])).sum() - (soxx.index > t).sum())  # 저점 뒤 지난 거래일
        feats = feats_at(F, uni, t, anchor)
        variants = {"지금 규칙": diamonds(feats, sub_of, "since"), "1개월안": diamonds(feats, sub_of, "r21")}
        for n in (5, 10, 15):
            variants[f"대체안({n}일)"] = diamonds(feats, sub_of, "r21" if age <= n else "since")
        for h in (10, 20):
            if ti + 1 + h >= len(cal):
                continue
            entry, exit_ = cal[ti + 1], cal[ti + 1 + h]
            rets = {s: V.fwd(F[s], entry, exit_) for s in feats}
            uni_ret = float(np.mean([r for r in rets.values() if r is not None]))
            for name, picks in variants.items():
                vals = [rets[s] for s in picks if rets.get(s) is not None]
                ret_cash = float(np.mean(vals)) if vals else 0.0
                recs.append({"t": t.date().isoformat(), "h": h, "variant": name, "n": len(vals), "age": age,
                             "ret": ret_cash, "uni": uni_ret})
                if name == "지금 규칙":
                    recs.append({"t": t.date().isoformat(), "h": h, "variant": "보류안", "n": len(vals), "age": age,
                                 "ret": float(np.mean(vals)) if vals else uni_ret, "uni": uni_ret})
    df = pd.DataFrame(recs)
    df["excess"] = df.ret - df.uni
    df["wide"] = df.t.map(breadth.ai_near10) >= 0.19
    out = []
    for (name, h), g in df.groupby(["variant", "h"], sort=False):
        grid = sorted(g.t.unique())[:: h // V.STEP]
        gg = g.set_index("t").loc[grid]
        rec = {"variant": name, "h": h, "dates": len(g), "empty": int((g.n == 0).sum()),
               "excess": round(g.excess.mean() * 100, 2), "hit": round((g.excess > 0).mean() * 100),
               "wide_excess": round(g[g.wide].excess.mean() * 100, 2), "narrow_excess": round(g[~g.wide].excess.mean() * 100, 2),
               "rebound_excess": round(g[g.t >= "2026-07-30"].excess.mean() * 100, 2),
               "compound": round((np.prod(1 + gg.ret) - 1) * 100, 1), "uni_compound": round((np.prod(1 + gg.uni) - 1) * 100, 1)}
        out.append(rec)
    res = pd.DataFrame(out)
    (OUT / "variants.json").write_text(json.dumps({"summary": out, "fresh_dates": df[df.age <= 15][["t", "h", "variant", "n", "age", "ret", "uni"]].to_dict("records")},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    with pd.option_context("display.width", 220):
        print(res[res.h == 20].to_string(index=False))
        print(res[res.h == 10].to_string(index=False))
        fresh = df[(df.age <= 10) & (df.h == 20)][["t", "age", "variant", "n", "ret", "uni"]]
        print(fresh.to_string(index=False))


if __name__ == "__main__":
    main()
