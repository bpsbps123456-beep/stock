"""지금 지표 매수의 매수 뒤 경로를 세 갈래로 나눠 본다(칼손절 연구 출발점).
  반등: 수익으로 끝남 / 폭포: -5% 넘게 빠졌고 그 바닥의 절반 이상을 매수 뒤 60분 안에 찍음 / 흘러내림: -3% 넘게 잃고 끝났으나 폭포가 아님
매수 뒤 k봉(분) 시점의 값(그때 이미 아는 값)이 갈래별로 다른지 본다. 사용: py stop_paths.py"""
import numpy as np, pandas as pd
from stop_lab import load, base_trades, PNAME, USER78

def paths(period):
    S = load(period); b = base_trades(period).copy()
    mins = S[next(iter(S))]["mins"]
    rows = []
    for x in b.itertuples():
        A = S[x.sym]; ei, xi, ent = x.ei, x.xi, x.ent
        l, c = A["l"][ei:xi + 1], A["c"][ei:xi + 1]
        dd = l / ent - 1
        k60 = max(1, 60 // mins)
        mae = dd.min()
        fast = dd[:k60 + 1].min() <= mae * 0.5 if mae < 0 else False
        cls = "반등" if x.ret > 0 else ("폭포" if mae <= -0.05 and fast else ("흘러내림" if x.ret <= -0.03 else "작은 손실"))
        f = {"cls": cls, "ret": x.ret, "mae": mae, "ses": x.ses, "sym": x.sym, "hold_min": x.bars * mins}
        for m in (15, 30, 60, 120):
            k = max(1, m // mins)
            if ei + k < len(A["c"]) and ei + k <= xi:
                f[f"r{m}"] = A["c"][ei + k] / ent - 1                  # 매수 뒤 m분 종가
                f[f"low{m}"] = A["l"][ei:ei + k + 1].min() / ent - 1   # 매수 뒤 m분 최저
                f[f"rsi{m}"] = A["r"][ei + k]
            else:
                f[f"r{m}"] = f[f"low{m}"] = f[f"rsi{m}"] = np.nan    # 이미 팔림
        rows.append(f)
    return pd.DataFrame(rows)

for p in ("early", "late"):
    d = paths(p)
    print(f"\n===== {PNAME[p]} 지금 지표 매수 {len(d)}건 (137종목)")
    g = d.groupby("cls").agg(건수=("ret", "size"), 평균=("ret", "mean"), 평균MAE=("mae", "mean"), 보유중앙=("hold_min", "median"))
    g["비중"] = g.건수 / len(d)
    print(g.assign(평균=g.평균.map("{:+.2%}".format), 평균MAE=g.평균MAE.map("{:+.1%}".format), 비중=g.비중.map("{:.0%}".format)).to_string())
    print(" 갈래별 합(수익 합): " + ", ".join(f"{k} {v:+.2f}" for k, v in d.groupby("cls").ret.sum().items()))
    print(" 세션 비중(폭포/흘러내림):", d[d.cls.isin(["폭포", "흘러내림"])].groupby(["cls", "ses"]).size().to_dict())
    for m in (15, 30, 60):
        q = d.groupby("cls")[[f"r{m}", f"low{m}", f"rsi{m}"]].median()
        print(f" 매수 뒤 {m}분 중앙값(아직 들고 있는 거래):\n" + q.to_string(float_format=lambda v: f"{v:+.3f}"))
