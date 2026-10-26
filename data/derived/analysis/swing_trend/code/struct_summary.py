"""struct 갈래 요약: results/struct_*.csv → results/struct_summary.json (머리 숫자·검정 개수)."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from swing import HERE, PERIODS, load  # noqa: E402
from struct_run import ETF, WARM, eligible  # noqa: E402

R = HERE / "results"
MAIN = ("P1", "P2", "Y25")
ZZ_MAIN = ("P15", "A3")


def f(x, d=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def main():
    st = pd.read_csv(R / "struct_zz_stats.csv")
    vr = pd.read_csv(R / "struct_vr_acf.csv")
    hi = pd.read_csv(R / "struct_len_hist.csv")
    cl = pd.read_csv(R / "struct_cluster.csv")
    wk = pd.read_csv(R / "struct_cluster_weeks.csv")
    out = {"시험범위": {}, "구조": {}, "분산비": {}, "갑작반전": {}, "시장공통": {}, "몰린주": {}, "변동성": {}}
    # 시험 범위와 우연 기대치(합산 지표, 전체 묶음)
    core = ["구간수", "평균길이", "평균크기", "몫_1_4일", "몫_5_22일", "몫_23일이상", "갑작반전비율", "첫3일몫", "확정뒤10일", "확정뒤20일"]
    x = st[(st.group == "전체") & st.metric.isin(core)]
    n = len(x)
    out["시험범위"] = dict(지그재그=6, 구간=5, 귀무=2, 귀무표본=200, 핵심지표=len(core), 검정수=n,
                       단측5퍼_위=int((x["합산_p_up"] <= 0.05).sum()), 단측5퍼_아래=int((x["합산_p_dn"] <= 0.05).sum()),
                       우연기대_각방향=round(n * 0.05, 1))
    by = x.assign(up=x["합산_p_up"] <= 0.05, dn=x["합산_p_dn"] <= 0.05).groupby("period")[["up", "dn"]].sum()
    out["시험범위"]["구간별_유의"] = {p: dict(위=int(by.loc[p, "up"]), 아래=int(by.loc[p, "dn"]), 검정=int((x.period == p).sum())) for p in by.index}
    # 구조 머리 숫자(전체 묶음, 섞기 귀무)
    for per in PERIODS:
        for zz in ("P10", "P15", "P20", "A3", "A4", "A5"):
            h = hi[(hi.period == per) & (hi.zz == zz)].set_index("data")
            row = {}
            for lab in ("실제", "섞기", "5일묶음"):
                r = h.loc[lab]
                row[lab] = dict(중앙길이=f(r["중앙길이"], 1), 구간수=f(r["구간수"], 0),
                                몫_1_4일=f(r["움직임_1_4"]), 몫_5_22일=f(r["움직임_5_10"] + r["움직임_11_22"]),
                                몫_23일이상=f(r["움직임_23_44"] + r["움직임_45_150"]))
            y = st[(st.group == "전체") & (st.period == per) & (st.zz == zz) & (st.null == "섞기")].set_index("metric")
            for m in ("평균크기", "평균길이", "확정뒤10일", "첫3일몫", "갑작반전비율"):
                row.setdefault("실제", {})[m] = f(y.loc[m, "합산_real"])
                row.setdefault("섞기", {})[m] = f(y.loc[m, "합산_null_med"])
                row.setdefault("p_위", {})[m] = f(y.loc[m, "합산_p_up"])
                row.setdefault("p_아래", {})[m] = f(y.loc[m, "합산_p_dn"])
            out["구조"][f"{per}_{zz}"] = row
    # 분산비: 주식 횡단 중앙과 지수
    for per in PERIODS:
        d = {}
        for grp in ("주식",):
            for q in ("VR5", "VR10", "VR20"):
                for null in ("섞기", "5일묶음"):
                    r = vr[(vr.period == per) & (vr.group == grp) & (vr.metric == q) & (vr.null == null)].iloc[0]
                    d[f"{grp}_{q}_{null}"] = dict(실제중앙=f(r["중앙_real"]), 귀무중앙=f(r["중앙_null_med"]),
                                                 귀무5=f(r["중앙_null_p05"]), 귀무95=f(r["중앙_null_p95"]),
                                                 p_위=f(r["중앙_p_up"]), p_아래=f(r["중앙_p_dn"]),
                                                 종목_귀무중앙초과=f(r["종목_귀무중앙초과"]))
        r = vr[(vr.period == per) & (vr.group == "전체") & (vr.metric == "VR20") & (vr.null == "섞기")].iloc[0]
        for e in ("SPY", "QQQ", "SOXX", "SMH", "IGV"):
            d[f"{e}_VR20"] = dict(실제=f(r[f"{e}_실제"]), 귀무5=f(r[f"{e}_귀무p05"]), 귀무95=f(r[f"{e}_귀무p95"]))
        out["분산비"][per] = d
    # 시장 공통성
    for _, r in cl.iterrows():
        out["시장공통"][f"{r.period}_{r.zz}"] = dict(
            동시반전_실제=f(r.ci_real), 동시반전_무작위=f(r.ci_null), 배=f(r.ci_real / r.ci_null, 2),
            갑작반전_동시=f(r.ci_sudden), 보통반전_동시=f(r.ci_normal), 갑작반전수=int(r.n_sudden), 반전수=int(r.n_piv),
            SOXX2ATR겹침_실제=f(r.SOXX_A2_ov_real), SOXX2ATR겹침_무작위=f(r.SOXX_A2_ov_null),
            SOXX2ATR겹침_갑작=f(r.SOXX_A2_ov_sudden), SOXX2ATR겹침_보통=f(r.SOXX_A2_ov_normal),
            QQQ2ATR겹침_실제=f(r.QQQ_A2_ov_real), QQQ2ATR겹침_무작위=f(r.QQQ_A2_ov_null),
            최대주저점_실제=f(r["주최대_저점_real"]), 최대주저점_무작위95=f(r["주최대_저점_null_p95"]),
            최대주고점_실제=f(r["주최대_고점_real"]), 최대주고점_무작위95=f(r["주최대_고점_null_p95"]))
    for per in MAIN:
        for zz in ZZ_MAIN:
            w = wk[(wk.period == per) & (wk.zz == zz)].copy()
            w["m"] = w[["저점비율", "고점비율"]].max(axis=1)
            w = w.sort_values("m", ascending=False).head(5)
            out["몰린주"][f"{per}_{zz}"] = [dict(주=f"{a}~{b}", 저점=f(lo), 고점=f(hh), SOXX주수익=f(sx))
                                         for a, b, lo, hh, sx in w[["start", "end", "저점비율", "고점비율", "SOXX주수익"]].values]
    # 일 변동성(주식 중앙): 무작위 걸음에서 θ 구간 길이의 눈금
    G = load()
    for per in PERIODS:
        dP, ok = eligible(G, per)
        sd = []
        for s, (j0, j1) in ok.items():
            if s in ETF:
                continue
            c = G[s].C.values[j0 - 1:j1 + 1]
            sd.append(np.std(np.diff(np.log(c)), ddof=1))
        out["변동성"][per] = dict(주식_일변동성_중앙=f(np.median(sd), 4), 종목수=len(sd),
                                 θ15_무작위걸음_눈금일=f((0.15 / np.median(sd)) ** 2, 1))
    (R / "struct_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["시험범위"], ensure_ascii=False))
    print(json.dumps(out["변동성"], ensure_ascii=False))
    for per in PERIODS:
        print(per, {k: v for k, v in out["분산비"][per].items() if "VR20" in k or "VR10" in k})


if __name__ == "__main__":
    main()
