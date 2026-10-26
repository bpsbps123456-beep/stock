"""이격매수 혼합: 본장 신호만 아래 지지 구역 하단에서 사기(2026-10-10 사용자 후속 질문, 키 defer_zone_hybrid).

질문(사용자): 구역 매수를 본장에서만 하면 어떤가. 큰 하락은 주로 본장에서 나온다.
규칙(defer_zone.machine의 hyb 옵션)
  프리·애프터 신호: 지금 지표 그대로 신호 가격 P에 산다.
  정규장 신호: P에 사지 않고 Zk 하단(또는 대조군 P x (1-d))에 지정가. 확장 '프리·본장'은 프리장 신호도 같게 한다. 수명 Lr = 정규장이 끝날 때까지(본장에서만),
    L1 = 지금 지표가 팔았을 시각까지(체결되면 지금 지표와 같은 봉·가격에 팜), L2 = 신호 뒤 첫 한국 아침 8시까지.
  매도·구역·기간·종목 묶음은 defer_zone.py와 같다. 비교 기준은 24시간 지금 지표다.
표
  0) 전제 확인: 지금 지표 거래의 큰 손실·매수 뒤 추가 하락이 어느 세션 신호에서 나오나
  1) 혼합 규칙 vs 지금 지표(24시간). 신호 세션별 합도 나눠 본다
  2) 수명 L1 짝 비교(정규장 신호): 큰 손실 거래에서 아낀 몫 vs 구역까지 안 내려와 놓친 승자 몫
  3) TEM 10/6 사례
사용: py defer_zone_hybrid.py → results/support/defer_zone_hybrid.txt, 신호표 scratchpad defer_zone_hybrid_trades.parquet
"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import defer_zone as D

OUT = os.path.normpath(os.path.join(D.CODE, "../results/support/defer_zone_hybrid.txt"))
BASE = "지금 지표(신호 때 삼)"
HYB = {
    "혼합 Z1 하단, Lr(본장 마감까지)": dict(mode=1, k=1, life=4, hyb=True),
    "혼합 Z2 하단, Lr(본장 마감까지)": dict(mode=1, k=2, life=4, hyb=True),
    "혼합 Z1 하단, L1(원래 매도까지)": dict(mode=1, k=1, life=1, hyb=True),
    "혼합 Z2 하단, L1(원래 매도까지)": dict(mode=1, k=2, life=1, hyb=True),
    "혼합 Z1 하단, L2(아침 8시까지)": dict(mode=1, k=1, life=2, hyb=True),
    "혼합 Z2 하단, L2(아침 8시까지)": dict(mode=1, k=2, life=2, hyb=True),
    # 확장: 프리장 신호도 구역까지 기다림(프리장에 샀다가 본장에서 빠지는 손실을 막으려는 뜻). 애프터 신호만 지금처럼
    "프리·본장 Z1 하단, L1(원래 매도까지)": dict(mode=1, k=1, life=1, hyb=True, dfr_ses=("프리", "정규")),
    "프리·본장 Z2 하단, L1(원래 매도까지)": dict(mode=1, k=2, life=1, hyb=True, dfr_ses=("프리", "정규")),
    "프리·본장 Z1 하단, L2(아침 8시까지)": dict(mode=1, k=1, life=2, hyb=True, dfr_ses=("프리", "정규")),
    "프리·본장 Z2 하단, L2(아침 8시까지)": dict(mode=1, k=2, life=2, hyb=True, dfr_ses=("프리", "정규")),
    "프리·본장 고정 -3%, L2": dict(mode=2, fixd=0.03, life=2, hyb=True, dfr_ses=("프리", "정규")),
}
FIX = {f"혼합 고정 -{d}%, {nm}": dict(mode=2, fixd=d / 100, life=lf, hyb=True)
       for lf, nm in ((4, "Lr"), (1, "L1")) for d in (1, 2, 3, 4, 5)}
PERIODS = ("early", "late", "oos")


def mae_of(p, t):
    """체결 거래마다 (체결 봉~매도 봉 저가 최소 / 매수가 - 1, 그 저가가 나온 봉의 세션)."""
    C = D.ctx(p)
    out, ses = np.full(len(t), np.nan), np.full(len(t), "", dtype=object)
    for j, (s, f, x, e, st) in enumerate(zip(t.sym, t.f, t.x, t.ent, t.stat)):
        if st == 0:
            A = C[s][0]
            k = f + int(np.argmin(A["l"][f:x + 1]))
            out[j], ses[j] = A["l"][k] / e - 1, A["ses"][k]
    return out, ses


def main():
    pd.set_option("display.width", 250)
    L = []
    w = L.append
    w("이격매수 혼합: 본장 신호만 아래 지지 구역 하단에서 사기 (2026-10-10, defer_zone_hybrid.py)")
    w("기준 = 24시간 지금 지표. 칸 뜻은 defer_zone.txt와 같다. 정규장만 매매(프리·애프터 매수 없음)의 구역 결과는 defer_zone.txt '정규장만' 표.")
    T = {}
    for p in PERIODS:
        T[(p, BASE)] = D.run(p, D.BASE, False)
        for nm, sp in {**HYB, **FIX}.items():
            T[(p, nm)] = D.run(p, sp, False)

    # 0) 전제 확인
    w("")
    w("=" * 20 + " 0) 전제 확인: 지금 지표(24시간) 거래를 신호 세션별로. 큰 손실 = 수익 -5% 이하, 추가 하락 = 매수 뒤 매도까지 저가 최소")
    rows = []
    for p in PERIODS:
        b = T[(p, BASE)]
        b = b[b.stat == 0].copy()
        b["mae"], b["lses"] = mae_of(p, b)
        for un, us in D.UNIV.items():
            x = b[b.sym.isin(us)]
            loss = x[x.ret < 0].ret.sum()
            for ses in ("프리", "정규", "애프터"):
                y = x[x.ses == ses]
                big = y[y.ret <= -0.05]
                rows.append(dict(기간=p, 종목=un, 세션=ses, 건수=len(y), 평균=y.ret.mean(), 합=y.ret.sum(),
                                 큰손실=len(big), 큰손실합=big.ret.sum(), 손실몫=y[y.ret < 0].ret.sum() / loss if loss else np.nan,
                                 큰손실비율=len(big) / len(y) if len(y) else np.nan,
                                 추가하락중앙=y.mae.median(), 추가하락3pct=(y.mae <= -0.03).mean(), 추가하락5pct=(y.mae <= -0.05).mean()))
    Z = pd.DataFrame(rows)
    f = Z.copy()
    for k, fm in (("평균", "{:+.2%}"), ("합", "{:+.2f}"), ("큰손실합", "{:+.2f}"), ("손실몫", "{:.0%}"), ("큰손실비율", "{:.1%}"),
                  ("추가하락중앙", "{:+.1%}"), ("추가하락3pct", "{:.0%}"), ("추가하락5pct", "{:.0%}")):
        f[k] = f[k].map(lambda v: "" if pd.isna(v) else fm.format(v))
    w("칸: 손실몫 = 그 세션 손실 합 / 전체 손실 합, 추가하락3pct·5pct = 매수 뒤 -3%·-5% 넘게 더 빠진 거래 비율")
    w(f.to_string(index=False))
    w("")
    w("0b) 큰 손실(-5% 이하)·중간 손실(-5~-2%) 거래의 저점이 실제로 찍힌 세션(신호 세션 → 저점 세션, 건수)")
    rows = []
    for p in PERIODS:
        b = T[(p, BASE)]
        b = b[b.stat == 0].copy()
        b["mae"], b["lses"] = mae_of(p, b)
        for un, us in D.UNIV.items():
            x = b[b.sym.isin(us)]
            for nm, y in (("≤-5%", x[x.ret <= -0.05]), ("-5~-2%", x[(x.ret > -0.05) & (x.ret <= -0.02)])):
                d = dict(기간=p, 종목=un, 손실=nm, 건수=len(y))
                for a in ("프리", "정규", "애프터"):
                    for z in ("프리", "정규", "애프터"):
                        d[f"{a}→{z}"] = int(((y.ses == a) & (y.lses == z)).sum())
                d["저점 정규 비율"] = (y.lses == "정규").mean() if len(y) else np.nan
                rows.append(d)
    f = pd.DataFrame(rows)
    f["저점 정규 비율"] = f["저점 정규 비율"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    w(f.to_string(index=False))

    # 1) 혼합 규칙 vs 지금 지표
    w("")
    w("=" * 20 + " 1) 혼합 규칙 vs 24시간 지금 지표. 정규합·밖합 = 정규장 신호 / 프리·애프터 신호 거래의 수익 합")
    rows = []
    for p in PERIODS:
        b = T[(p, BASE)]
        for nm in [BASE] + list(HYB) + list(FIX):
            t = T[(p, nm)]
            for un, us in D.UNIV.items():
                m = D.metrics(t, b, us)
                tr = t[(t.stat == 0) & t.sym.isin(us)]
                rows.append(dict(기간=p, 종목=un, 규칙=nm, **m, 정규합=tr[tr.ses == "정규"].ret.sum(), 밖합=tr[tr.ses != "정규"].ret.sum()))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(D.TMP, "defer_zone_hybrid_metrics.csv"), index=False, encoding="utf-8-sig")
    cols = ["규칙", "신호", "체결", "체결률", "평균", "승률", "최악", "하위5", "합", "정규합", "밖합", "차이", "t", "나은날", "나쁜날",
            "계좌복리", "최악낙폭"]
    for un in D.UNIV:
        for p in PERIODS:
            x = R[(R.기간 == p) & (R.종목 == un)][cols].copy()
            for k in ("정규합", "밖합"):
                x[k] = x[k].map("{:+.2f}".format)
            w(f"\n[{un} · {D.PN[p]} · 24시간]")
            w(D.fmt_table(x))

    # 2) L1 짝 비교: 정규장 신호에서 아낀 몫과 놓친 몫
    w("")
    w("=" * 20 + " 2) 수명 L1 짝 비교(정규장 신호만): 지금 지표 거래를 수익 구간별로 나눠 혼합이 바꾼 몫")
    w("   체결 = 구역(깊이)까지 내려와 싸게 산 거래(매도는 같은 봉·가격), 놓침 = 안 내려와 안 산 거래(지금 지표 수익을 잃음)")
    rows = []
    for p in PERIODS:
        b = T[(p, BASE)]
        b = b[(b.stat == 0) & (b.ses == "정규")].set_index(["sym", "s"])
        for nm in ("혼합 Z1 하단, L1(원래 매도까지)", "혼합 Z2 하단, L1(원래 매도까지)", "혼합 고정 -1%, L1", "혼합 고정 -3%, L1"):
            t = T[(p, nm)]
            t = t[(t.stat != 3) & (t.ses == "정규")].set_index(["sym", "s"])
            j = b[["ret"]].join(t[["stat", "ret"]].rename(columns={"ret": "hret"}), how="inner")
            j["hret"] = j.hret.fillna(0.0)                        # 안 산 거래는 0
            j["chg"] = j.hret - j.ret
            j["구간"] = pd.cut(j.ret, [-1, -0.05, -0.02, 0, 0.02, 10], labels=["≤-5%", "-5~-2%", "-2~0%", "0~+2%", ">+2%"])
            for un, us in D.UNIV.items():
                q = j[j.index.get_level_values(0).isin(us)]
                d = dict(기간=p, 종목=un, 규칙=nm.replace("혼합 ", "").replace(", L1(원래 매도까지)", "").replace(", L1", ""),
                         짝=len(q), 체결=int((q.stat == 0).sum()), 합변화=q.chg.sum())
                for g, gg in q.groupby("구간", observed=False):
                    d[f"{g} 건"] = len(gg)
                    d[f"{g} 변화"] = gg.chg.sum()
                rows.append(d)
    P2 = pd.DataFrame(rows)
    f = P2.copy()
    for c in f.columns:
        if c.endswith("변화"):
            f[c] = f[c].map("{:+.2f}".format)
    w(f.to_string(index=False))

    # 3) TEM 10/6
    w("")
    w("=" * 20 + " 3) TEM 10/6: 전날 +9%라 지금 지표는 정규장 매수를 쉰다(10/7 채택한 전날 +7% 규칙). 그 규칙 전에는 09:57 ET 83.30 매수 → -13.1%")
    for nm in [BASE] + list(HYB):
        t = T[("late", nm)]
        x = t[(t.sym == "TEM") & (t.day == date(2026, 10, 6))]
        for _, r in x.iterrows():
            st = {0: "체결·매도", 1: "미체결 취소", 2: "구역 없음", 3: "자료 끝"}[r.stat]
            w(f"  {nm}: 신호 {r.ses} P {r.P:.2f} 주문가 {r.L:.2f} → {st}" + (f" 매수 {r.ent:.2f} 매도 {r.px:.2f} 수익 {r.ret:+.1%}" if r.stat == 0 else ""))

    keep = []
    for (p, nm), t in T.items():
        x = t.copy()
        x["기간"], x["범위"], x["규칙"] = p, "24시간", nm
        keep.append(x)
    pd.concat(keep).to_parquet(os.path.join(D.TMP, "defer_zone_hybrid_trades.parquet"))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
