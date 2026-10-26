"""전일 +5% 다음 날 정규장 금지 규칙을 사용자 종목(트레이딩뷰 AI·TIME·KoAct)에서 5분봉 7/14~9/8(겹치지 않는 앞 기간)로 확인한다.
사용: py user_rule_5m.py <raw 폴더>"""
import sys
from datetime import date
from events import bars, prep
from fetch import user_universe
from rules import run, hot_days, line

if __name__ == "__main__":
    raw = sys.argv[1]
    U = user_universe()
    D = {}
    for s in sorted(set().union(*U.values())):
        try:
            D[s] = prep(bars(raw, s, 5), 36, 9)
        except FileNotFoundError:
            pass
    H = {s: hot_days(raw, s, d) for s, d in D.items()}
    lo, hi = date(2026, 7, 14), date(2026, 9, 8)
    for reg in (False, True):
        b = run(D, None, lo, hi, H, reg_only=reg)
        r = run(D, None, lo, hi, H, reg_only=reg, hot=True)
        print(f"5분봉 7/14~9/8 {'정규장만' if reg else '24시간'}")
        for k, syms in U.items():
            print(f"  {k:8s} 지금 {line(b[b.sym.isin(syms)])}\n  {'':8s} 규칙 {line(r[r.sym.isin(syms)])}")
