"""'혼자 빠짐' 단서 확인(2026-10-07): 자기 i-1 종가 <= 매수가 x (1-Y) 이고 같은 그룹 중앙값의 매수 뒤 수익 > -X 이면 판다.
이웃 매개변수(Y 4·5·6%, X 0.5·1·2%)와 하루 쏠림(가장 이득 본 날 빼기)을 본다. 정의·그룹 계산은 stops/comove.py의 make('idio')를 쓴다."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
from stop_lab import compare, run, load, base_trades, USER78
from comove import make

rows = []
for Y in (0.04, 0.05, 0.06):
    for X in (0.005, 0.01, 0.02):
        df = compare(make("idio", X, Y), f"Y{Y:.0%} X{X:.1%}", show=False)
        u = df[df["종목"] == "사용자78"].set_index(["기간", "범위"])
        a = df[df["종목"] == "전체137"].set_index(["기간", "범위"])
        rows.append(dict(규칙=f"자기 -{Y:.0%} & 그룹 > -{X:.1%}",
                         e24=u.loc[("early", "24시간"), "차이"], e정=u.loc[("early", "정규장만"), "차이"],
                         l24=u.loc[("late", "24시간"), "차이"], l정=u.loc[("late", "정규장만"), "차이"],
                         e24_137=a.loc[("early", "24시간"), "차이"], l24_137=a.loc[("late", "24시간"), "차이"]))
r = pd.DataFrame(rows)
print(r.round(2).to_string(index=False))
print("네 칸(사용자78) 모두 플러스:", int((r[["e24", "e정", "l24", "l정"]] > 0).all(axis=1).sum()), "/", len(r))
# 하루 쏠림: 기본 정의(Y5%, X1%)
mk = make("idio", 0.01, 0.05)
for p in ("early", "late"):
    t = run(load(p), mk, p, False); b = base_trades(p, False)
    t, b = t[t.sym.isin(USER78)], b[b.sym.isin(USER78)]
    d = t.groupby("day").ret.sum().sub(b.groupby("day").ret.sum(), fill_value=0)
    best = d.idxmax()
    print(f"{p} 24시간 사용자78: 차이 {d.sum():+.2f}, 가장 이득 본 날 {best} {d.max():+.2f} → 빼면 {d.drop(best).sum():+.2f} (바뀐 날 {(d.abs() > 1e-12).sum()}일)")
