"""추세 5단계 × SOXX 주간 경고(2026-10-10, zones.py 다음 단계, 연구용).

zones.py 결과: 1~5월로 고른 1위가 6~10월에 뒤집혔다(앞 S +4.6 → 뒤 -4.1). 자격 107개 전체에서도 1~5월은 '추세가 이어짐'(T1 양수 99%),
6~10월·2025는 '극단이 되돌아옴'(T2·T3 양수 78~80%)이었다. 질문: 이 갈림이 SOXX 주간 경고와 겹치나?
경고(사전 정의, gate_main.py와 같음): t+1일이 속한 ISO 주 직전 거래일 종가의 SOXX < SMA20(일봉). t+1 시가 전에 안다.
방법: 설정 하나를 고르지 않는다. 자격 107개 전체(+사전 등록 1위, 옛 칼만)에서 세 항을 경고 주 / 평상 주로 나눠 재고,
      앞·뒤·2025 각각 양수 비율과 중앙값을 낸다. 고른 것이 아니라 분포라서 선택편향이 작다. 다만 이 질문 자체는 결과를 보고 나왔다.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/zones_regime.py
"""
import json, os, sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402

SOXX_SRC = os.path.join(Zm.ROOT, "data/derived/analysis/stop_research/cache/ohlc.parquet")


def soxx_week_warning(dates):
    o = pd.read_parquet(SOXX_SRC, columns=["date", "sym", "Close"])
    s = o[o.sym == "SOXX"].copy()
    s["d"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
    s = s.set_index("d")["Close"].astype(float).sort_index()
    s = s[~s.index.duplicated()]
    below = (s < s.rolling(20, min_periods=20).mean())
    days = list(s.index)
    iso = {d: date.fromisoformat(d).isocalendar()[:2] for d in days}
    # 각 거래일의 '이번 주 경고' = 그 주 첫 거래일 직전 거래일 종가 상태
    week_state, cur_wk, state = {}, None, False
    for i, d in enumerate(days):
        if iso[d] != cur_wk:
            cur_wk = iso[d]
            state = bool(below.iloc[i - 1]) if i > 0 else False
        week_state[d] = state
    # 신호일 t → t+1일의 주 상태
    out = np.zeros(len(dates), dtype=bool)
    for i in range(len(dates) - 1):
        out[i] = week_state.get(dates[i + 1], False)
    return out


def main():
    I = Zm.Ind(*Zm.load())
    F = Zm.Fwd(I, Zm.H)
    warn = soxx_week_warning(I.dates)
    res = json.load(open(os.path.join(Zm.OUT, "zones.json"), encoding="utf-8"))
    cfgs = [g["cfg"] for g in res["grid"] if g["eligible"]]
    best = next(g["cfg"] for g in res["grid"] if g["name"] == res["best"])
    keys = ("T1 상방-하방", "T2 극하방-하방", "T3 상방-극상방")
    lines = [f"자격 {len(cfgs)}개, h={Zm.H}, 초과 수익. 경고 주 비율(신호일 기준):"]
    split = {}
    for p in ("앞", "뒤", "2025", "2024"):
        rows = Zm.period_rows(I, p, Zm.H)
        split[p] = {"경고": rows[warn[rows]], "평상": rows[~warn[rows]]}
        lines.append(f"  {p}: 경고 {warn[rows].mean() * 100:.0f}% ({warn[rows].sum()}일)")
    table = {}
    for c in cfgs + [best, Zm.BASELINE]:
        Z = Zm.zones(I, c)
        nm = Zm.cname(c)
        table[nm] = {p: {s: Zm.evaluate(Z, F, r)["terms"] for s, r in split[p].items()} for p in split}
    lines += ["", "== 자격 107개 분포: 항이 양수인 설정 비율 / 중앙값(%p)"]
    for p in ("앞", "뒤", "2025", "2024"):
        for s in ("평상", "경고"):
            parts = []
            for k in keys:
                v = np.array([table[Zm.cname(c)][p][s][k] for c in cfgs])
                v = v[np.isfinite(v)]
                parts.append(f"{k} {np.mean(v > 0) * 100:3.0f}% {np.median(v) * 100:+5.2f}")
            lines.append(f"[{p}·{s}] " + " | ".join(parts))
    for label, c in (("사전 등록 1위", best), ("옛 칼만", Zm.BASELINE)):
        lines += ["", f"== {label}: {Zm.cname(c)}"]
        for p in ("앞", "뒤", "2025", "2024"):
            for s in ("평상", "경고"):
                t = table[Zm.cname(c)][p][s]
                lines.append(f"[{p}·{s}] " + "  ".join(f"{k} {t[k] * 100:+5.2f}" for k in keys))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(Zm.OUT, "zones_regime.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(Zm.OUT, "zones_regime.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
