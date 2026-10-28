"""+7% 확인 시점(지금과 같은 자리)에서 종목 고르는 규칙 7개를 2026 네 국면에 비교한다 (2026-10-28).

진입: SOXX가 국면 저점 +7%를 처음 넘긴 날 종가. 보유: 21거래일(예측 기간 한 달과 같음).
규칙(모두 진입일에 아는 값, 상위 10 동일 비중):
  beta          고점 전 60거래일 SOXX 베타(높을수록)
  early         저점→진입일 수익(반등 초반 주도)
  deep          고점→저점 하락(많이 빠질수록)
  resilient     하락 중 베타 대비 덜 빠짐(하락의 베타 회귀 잔차, 클수록)
  mom120        고점까지 120거래일 수익(급락 전 강세)
  near_high     진입일 52주 고점 대비 거리(가까울수록)
  prior         직전 반등(앞 저점→이번 고점)
결과: ../results/entry_rules.json, 표준 출력 표.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze import OUT, beta, episodes, ret

HOLD = 21
TOP_N = 10
ENTRY_UP = 0.07


def factors(px: pd.DataFrame, ep: dict, entry) -> pd.DataFrame:
    names = [c for c in px.columns if c not in ("SOXX", "QQQ")]
    p = px[names]
    pos_peak = px.index.get_loc(ep["peak"])
    drop = (p.loc[ep["trough"]] / p.loc[ep["peak"]] - 1)
    b = beta(px, ep["peak"])[names]
    j = pd.concat([drop, b], axis=1, keys=["d", "b"]).dropna()
    fit = np.polyfit(j["b"], j["d"], 1)
    resid = j["d"] - np.polyval(fit, j["b"])
    hi = p.loc[:entry].iloc[-252:].max()
    f = pd.DataFrame({
        "beta": b,
        "early": p.loc[entry] / p.loc[ep["trough"]] - 1,
        "deep": -drop,
        "resilient": resid,
        "mom120": p.iloc[pos_peak] / p.iloc[max(0, pos_peak - 120)] - 1,
        "near_high": p.loc[entry] / hi - 1,
    })
    if ep["prev_trough"] is not None:
        f["prior"] = p.loc[ep["peak"]] / p.loc[ep["prev_trough"]] - 1
    return f


def main() -> None:
    px = pd.read_csv(OUT / "prices.csv", index_col=0, parse_dates=True)
    rows, detail = [], {}
    for ep in episodes(px):
        if ep["trough"] < pd.Timestamp("2026-01-01") or not ep["complete"]:
            continue
        s = px["SOXX"].loc[ep["trough"]:]
        entry = s[s >= s.iloc[0] * (1 + ENTRY_UP)].index[0]
        pos = px.index.get_loc(entry)
        end = px.index[min(pos + HOLD, len(px) - 1)]
        names = [c for c in px.columns if c not in ("SOXX", "QQQ")]
        fwd = px[names].loc[end] / px[names].loc[entry] - 1
        soxx = float(px["SOXX"].loc[end] / px["SOXX"].loc[entry] - 1)
        allm = float(fwd.mean())
        f = factors(px, ep, entry)
        key = ep["trough"].date().isoformat()
        detail[key] = {"entry": entry.date().isoformat(), "exit": end.date().isoformat(), "soxx": round(soxx, 4),
                       "all": round(allm, 4), "rules": {}}
        for rule in f.columns:
            x = f[rule].dropna()
            top = x.sort_values(ascending=False).index[:TOP_N]
            rho, p = stats.spearmanr(x, fwd[x.index], nan_policy="omit")
            r = {"top10": round(float(fwd[top].mean()), 4), "vs_soxx": round(float(fwd[top].mean()) - soxx, 4),
                 "vs_all": round(float(fwd[top].mean()) - allm, 4), "rho": round(float(rho), 3),
                 "p": round(float(p), 4), "top": list(top)}
            detail[key]["rules"][rule] = r
            rows.append({"trough": key, "rule": rule, **{k: r[k] for k in ("vs_soxx", "vs_all", "rho")}})
    (OUT / "entry_rules.json").write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
    t = pd.DataFrame(rows)
    for col in ("vs_all", "rho"):
        print(f"\n== {col} (상위 10 − 43종목 평균, 또는 순위 상관) ==")
        print(t.pivot(index="rule", columns="trough", values=col).round(3).to_string())
    wins = t.assign(win=t["vs_all"] > 0).groupby("rule")["win"].sum()
    print("\n43종목 평균을 이긴 국면 수:", wins.to_dict())
    for k, v in detail.items():
        print(k, v["entry"], "→", v["exit"], "SOXX", v["soxx"], "43종목", v["all"])


if __name__ == "__main__":
    main()
