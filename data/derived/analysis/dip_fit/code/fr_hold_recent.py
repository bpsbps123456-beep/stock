"""빠른 반등 때 '+2% 위쪽 지정가'(fr_hold의 a2)가 최근으로 올수록 더 잘 맞는가(2026-10-10 사용자 질문).
fr_hold.py의 사건 계산을 그대로 쓰고, 정규장 매수 중 빠른 반등(RSI 매도까지 3분봉 환산 20봉 이하, 7~9월 5분봉은 12봉 이하)만 구간별로 나눈다.
구간: 7/14~7/31, 8월, 9/1~9/8(5분봉 근사), 9/10~9/18, 9/21~9/30, 10/1~10/9(3분봉).
차이 = (a2 매도가 - RSI 매도가)/매수가, 지금 규칙 대비 거래당 이득. 같은 날 QQQ 정규장 수익(시가→종가)도 붙여 시장 반등일 쏠림을 본다.
같은 구간의 K=30(90분) 결과도 함께 낸다(표본을 늘린 확인용, 고르기에 쓰지 않음).
사용: py fr_hold_recent.py"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fr_hold

BUCKETS = [("7/14~7/31", date(2026, 7, 14), date(2026, 7, 31)), ("8월", date(2026, 8, 1), date(2026, 8, 31)),
           ("9/1~9/8", date(2026, 9, 1), date(2026, 9, 8)), ("9/10~9/18", date(2026, 9, 10), date(2026, 9, 18)),
           ("9/21~9/30", date(2026, 9, 21), date(2026, 9, 30)), ("10/1~10/9", date(2026, 10, 1), date(2026, 10, 9))]


def qqq_day():
    out = {}
    for raw in (fr_hold.RAW, fr_hold.stop_lab.RAW):
        d = pd.read_parquet(os.path.join(raw, "1d_QQQ.parquet"))
        d.index = pd.to_datetime(d.index).date
        for k, r in d.iterrows():
            out.setdefault(k, r.Close / r.Open - 1)
    return out


def table(E, K, q):
    F = E[(E.ses == "정규") & (E.bars <= K)].copy()
    F["d"] = (F.x_a2 - F.px) / F.ent
    F["fill"] = F.x_a2 >= F.px * 1.02 - 1e-9
    F["q"] = F.day.map(q)
    rows = []
    for name, a, b in BUCKETS:
        g = F[(F.day >= a) & (F.day <= b)]
        if not len(g):
            rows.append(dict(구간=name, 건수=0))
            continue
        dd = g.groupby("day").d.sum()
        worst_day = dd.idxmin()
        rows.append(dict(구간=name, 봉=("5분" if a < date(2026, 9, 10) else "3분"), 건수=len(g), 날수=g.day.nunique(),
                         지정가체결=f"{int(g.fill.sum())}/{len(g)}", 차이합=g.d.sum(), 거래당=g.d.mean(),
                         나아짐=int((g.d > 1e-9).sum()), 나빠짐=int((g.d < -1e-9).sum()), 최악=g.d.min(),
                         최악일=f"{worst_day:%m/%d} {dd.min():+.1%}", 최악일빼면=g[g.day != worst_day].d.sum(),
                         QQQ그날평균=g.q.mean(), t=fr_hold.crt(g.d, g.day)))
    return pd.DataFrame(rows), F


if __name__ == "__main__":
    q = qqq_day()
    El, _, _, _, _ = fr_hold.late_events()
    Ee, _ = fr_hold.early_events()
    E = pd.concat([Ee, El], ignore_index=True)
    E["day"] = pd.to_datetime(E.day).dt.date
    pd.set_option("display.width", 250)
    out = []
    for K in (20, 30):
        T, F = table(E, K, q)
        for c in ("차이합", "거래당", "최악", "최악일빼면", "QQQ그날평균"):
            T[c] = T[c].map(lambda v: "" if pd.isna(v) else f"{v:+.2%}")
        T["t"] = T["t"].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
        out.append(f"\n== 빠른 반등 K={K}(3분봉 환산, {K * 3}분) 정규장 매수, '+2% 위쪽 지정가' 대 지금 규칙 ==\n" + T.to_string(index=False))
        if K == 20:
            lr = F[F.day >= date(2026, 9, 10)].sort_values("day")
            out.append("\n9/10 이후 거래별(K=20): " + ", ".join(f"{r.day:%m/%d} {r.sym} {r.d:+.1%}" for r in lr.itertuples()))
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(HERE, "../results/fr_hold_recent.txt"), "w", encoding="utf-8").write(__doc__ + "\n" + txt + "\n")
