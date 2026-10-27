"""사후 질문(2026-10-27): "최소가 8%일까? 7%는?" — SOXX 반등 문턱을 촘촘히 바꿔 잰다(결과를 본 뒤 탐색, 판정 없음).

문턱 k = 3·4·5·5.5·6·6.5·7·7.5·8·8.5·9·10·12%. 규칙 up ≥ k 를 등록 엔진(bc_lib, 수정 뒤)으로 그대로 평가한다(in_dd, 20일 쿨다운).
문턱마다
  - 첫 신호 hold20·hold40(2024~2026·2026·2025·2024), up 중앙
  - 실패한 신호(hold20 = 0)의 날짜·up
  - 국면 단위 최종 저점(전고점 회복 전까지 저점 안 깨짐) 비율: episodes.py 와 같은 정의
  - 날짜 단위(쿨다운 없음) in_dd 날 up 구간 [k, k+1%) 의 hold20
py code/threshold_sweep.py → results/sweep_SOXX_posthoc_{summary,signals,verdict}.csv · sweep_table.csv · sweep_tables.txt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bc_lib as L

KS = [0.03, 0.04, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.10, 0.12]


def final_low(C: pd.Series, fr, date) -> str:
    idx = C.index
    t = idx.get_loc(date)
    ldd = float(fr["L60"]["SOXX"].iloc[t])
    peak = float(C.iloc[t - 59:t + 1].max())
    fut = C.iloc[t + 1:]
    b = fut.index[fut < ldd - 1e-12]
    r = fut.index[fut >= peak - 1e-12]
    if len(b) and (not len(r) or b[0] < r[0]):
        return "no"
    if len(r):
        return "yes"
    return "open"


def main():
    pd.set_option("display.width", 250)
    d = L.load_soxx()
    fr = L.soxx_frames()
    base = L.matched_baseline(fr, "SOXX")
    rules = {f"up{k * 100:g}": fr["up"] >= k for k in KS}
    summ, ver = L.run_rules(rules, "SOXX", "sweep", fr=fr, base=base, tag="sweep_SOXX_posthoc")
    sig = pd.read_csv(L.RES / "sweep_SOXX_posthoc_signals.csv", encoding="utf-8-sig", parse_dates=["date"])
    up, st, h20 = fr["up"]["SOXX"], fr["in_dd"]["SOXX"], fr["hold20"]["SOXX"]
    m = st & (up.index >= L.EVAL_START["SOXX"])
    rows = []
    for k in KS:
        name = f"up{k * 100:g}"
        s = summ[summ.rule == name].set_index("period")
        g = sig[sig.rule == name].copy()
        g["final"] = [final_low(d.C, fr, x) for x in g.date]
        fails = g[g.hold20 == 0]
        decl = g.drop_duplicates(subset=["date"]).copy()
        decl["Ldd"] = [round(float(fr["L60"]["SOXX"].loc[x]), 2) for x in decl.date]
        ep = decl.groupby("Ldd").final.agg(lambda v: "no" if (v == "no").any() else ("yes" if (v == "yes").any() else "open"))
        band = m & (up >= k) & (up < k + 0.01)
        hv = h20[band].dropna()

        def cell(p, col):
            return s.loc[p, col] if (p in s.index and col in s.columns and pd.notna(s.loc[p, col])) else np.nan

        rows.append(dict(k=k, n=int(cell("2024~2026", "hold20_n") or 0), hold20=cell("2024~2026", "hold20"),
                         hold40=cell("2024~2026", "hold40"), n26=int(cell("2026", "hold20_n") or 0), hold20_26=cell("2026", "hold20"),
                         hold40_26=cell("2026", "hold40"), hold20_25=cell("2025", "hold20"), hold20_24=cell("2024", "hold20"),
                         near5=cell("2024~2026", "near5"), win20=cell("2024~2026", "win20"), r20=cell("2024~2026", "r20"),
                         up_median=cell("2024~2026", "up_median"),
                         final_yes=int((ep == "yes").sum()), final_no=int((ep == "no").sum()), final_open=int((ep == "open").sum()),
                         fails="; ".join(f"{x.date()} +{u:.1%}" for x, u in zip(fails.date, fails.up)),
                         band_days=len(hv), band_hold20=hv.mean() if len(hv) else np.nan,
                         band_episodes=int(fr["L60"]["SOXX"][hv.index].round(2).nunique()) if len(hv) else 0))
    t = pd.DataFrame(rows)
    t.to_csv(L.RES / "sweep_table.csv", index=False, encoding="utf-8-sig")
    txt = t.round(3).to_string(index=False)
    (L.RES / "sweep_tables.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
