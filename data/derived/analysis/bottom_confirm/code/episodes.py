"""사후 질문(2026-10-27): "반등이 8% 이상이면 그 하락에서 가장 낮은 저점이었다는 건가?"

hold20 은 '한 달 안에 그 저점을 다시 깨지 않음'이다. '그 하락의 최종 저점'은 다른 질문이라 국면 단위로 따로 잰다.
대상: 등록 규칙 F1 up5·up8·up10 의 SOXX 첫 신호(results/f1_SOXX_signals.csv, 엔진 수정 뒤 값).
신호마다
  - Ldd(그때까지의 하락 저점), 전고점 = 신호일의 60일 종가 최고
  - 저점이 처음 다시 깨진 날(종가 < Ldd)과 그때까지 거래일 수(2026-10-23까지 안 깨졌으면 '안 깨짐')
  - 최종 저점 여부 = 종가가 전고점을 되찾기 전에 Ldd 가 깨지지 않았음(자료 끝까지 전고점도 못 되찾고 안 깨졌으면 '아직 모름')
  - 깨졌다면 전고점 회복 전까지의 최저 종가와 Ldd·신호일 종가 대비 깊이
설명용이다(판정 없음).

py code/episodes.py → results/episodes_up.csv · episodes_tables.txt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bc_lib as L


def main():
    d = L.load_soxx()
    C = d.C
    idx = C.index
    sig = pd.read_csv(L.RES / "f1_SOXX_signals.csv", encoding="utf-8-sig", parse_dates=["date"])
    fr = L.soxx_frames()
    rows = []
    for rule in ("up5", "up8", "up10"):
        for _, s in sig[sig.rule == rule].iterrows():
            t = idx.get_loc(s.date)
            ldd = float(fr["L60"]["SOXX"].iloc[t])
            peak = float(C.iloc[t - 59:t + 1].max())
            fut = C.iloc[t + 1:]
            brk = fut[fut < ldd - 1e-12]
            rec = fut[fut >= peak - 1e-12]
            b_day = brk.index[0] if len(brk) else None
            r_day = rec.index[0] if len(rec) else None
            if b_day is not None and (r_day is None or b_day < r_day):
                final = "아님(전고점 회복 전에 깨짐)"
            elif r_day is not None:
                final = "맞음(전고점까지 안 깨짐)"
            else:
                final = "아직 모름(전고점 미회복·안 깨짐)"
            end = r_day if r_day is not None else idx[-1]
            seg = C.loc[s.date:end]
            low_after = float(seg.min())
            rows.append(dict(rule=rule, date=s.date.date().isoformat(), close=float(C.iloc[t]), Ldd=ldd, up=float(s.up),
                             peak=peak, break_date=None if b_day is None else b_day.date().isoformat(),
                             sessions_to_break=None if b_day is None else int(idx.get_loc(b_day) - t),
                             recover_date=None if r_day is None else r_day.date().isoformat(), final_low=final,
                             low_before_recover=low_after, low_vs_Ldd=low_after / ldd - 1, low_vs_signal=low_after / float(C.iloc[t]) - 1,
                             hold20=s.hold20, hold40=s.hold40))
    t = pd.DataFrame(rows)
    t.to_csv(L.RES / "episodes_up.csv", index=False, encoding="utf-8-sig")
    lines = []
    for rule, g in t.groupby("rule", sort=False):
        lines.append(f"== {rule} 첫 신호 {len(g)}개 ==")
        lines.append(g[["date", "close", "Ldd", "up", "peak", "break_date", "sessions_to_break", "recover_date", "final_low",
                        "low_vs_Ldd", "low_vs_signal", "hold20", "hold40"]].round(3).to_string(index=False))
        vc = g.final_low.value_counts().to_dict()
        lines.append(f"최종 저점: {vc}")
        lines.append("")
    txt = "\n".join(lines)
    (L.RES / "episodes_tables.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
