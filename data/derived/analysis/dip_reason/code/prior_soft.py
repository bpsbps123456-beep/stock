"""전일 급등 규칙을 '무조건 안 삼' 대신 '더 깊게 삼'으로 부드럽게 바꾸면 어떤가(2026-10-07 사용자: 무조건 조심은 과하다).
전일 +5·7·10% 이상 다음 날 정규장 매수를 안 삼 / 매수선 -3·-4·-5%로. 사용자 78종목, 앞·뒤 기간, 24시간·정규장만.
사용: py prior_soft.py <raw 폴더>"""
import sys
import numpy as np, pandas as pd
from htf_study import PERIODS, base_arrays, run_cfg, metrics, NOBUY
from fetch import user_universe
from rules import hot_days

if __name__ == "__main__":
    raw = sys.argv[1]
    us = user_universe()["합집합"]
    out = []
    for pname, (mins, el, rl, lo, hi) in PERIODS.items():
        S = {}
        for s in us:
            try:
                S[s] = base_arrays(raw, s, mins, el, rl)
            except FileNotFoundError:
                pass
        days = sorted({d for A in S.values() for d in set(A["day"]) if lo <= d <= hi})
        H = {th: {s: hot_days(raw, s, A["d"], th) for s, A in S.items()} for th in (0.05, 0.07, 0.10)}
        for scope in ("24시간", "정규장만"):
            reg_only = scope == "정규장만"
            bg = lambda A, s: np.where(A["reg"] | (not reg_only), 0.02, NOBUY)
            base = run_cfg(S, bg, lo, hi)
            out.append(dict(기간=pname, 범위=scope, 규칙="지금 지표", **metrics(base, base, us, days)))
            for th, HH in H.items():
                for aname, av in (("안 삼", NOBUY), ("-3%", 0.03), ("-4%", 0.04), ("-5%", 0.05)):
                    fn = lambda A, s: np.where(bg(A, s) >= 1, NOBUY, np.where(HH[s], av, 0.02))
                    t = run_cfg(S, fn, lo, hi)
                    out.append(dict(기간=pname, 범위=scope, 규칙=f"전일 +{th:.0%} → {aname}", **metrics(t, base, us, days)))
    r = pd.DataFrame(out)
    piv = r.pivot_table(index="규칙", columns=["기간", "범위"], values="차이", sort=False).round(2)
    print("지금 지표 대비 합 차이 (사용자 78종목)\n" + piv.to_string())
    tem = r[r["기간"].str.startswith("뒤")].pivot_table(index="규칙", columns="범위", values="TEM", sort=False)
    print("\nTEM 10/6 결과(빈칸 = 그날 정규장 매수 없음)\n" + tem.round(3).to_string())
    worst = r.pivot_table(index="규칙", columns=["기간", "범위"], values="최악", sort=False).round(3)
    print("\n최악 거래\n" + worst.to_string())
