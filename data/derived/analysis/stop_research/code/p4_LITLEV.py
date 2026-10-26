"""4단계 LITLEV 보조 계산(2026-10-11): 문헌의 변동성 손실 공식을 우리 자료로 확인한다. 전략 시험이 아니다(고르기 없음).

(A) 레짐 분해: 기초자산이 t-1 종가 기준 자기 200일선 위/아래일 때 t일 수익의 연 평균·연 변동성·1일 자기상관,
    그리고 Avellaneda-Zhang 근사 변동성 손실 (L^2-L)/2·σ^2 (연)과 합성 L배의 그 레짐 연 수익.
    대상 기초 5개(QQQ·SOXX·SPY·NVDA·TSLA) × L {2, 3} × 레짐 2 = 20칸 + 해별 표(2018·2020·2022·2025·2026앞·2026뒤).
(B) 실제 단일 종목 2배의 추적차 진단: 실제-합성 일수익 차가 |차| > 5%인 날(배당·분배·자료 오류 의심)을 따로 떼어
    '그 며칠을 빼면 연 추적차가 얼마인가'를 본다. 대상 = lev_lab.products()에서 L=2이고 기초가 단일 종목인 전부.
(C) lev_lab.validate()의 '연추적차'는 CAGR 차(%p)라서 CAGR이 클수록 부풀려진다. 합성에 보수처럼 더 빼야 하는 값은
    비율 기준 (실제/합성)^(1/년) - 1이다. 8개 상품에서 둘을 나란히 보인다.
시점: 레짐은 t-1 종가와 t-1까지의 200일 평균으로 정한다(미래 정보 없음). 공유 엔진은 읽기만 한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import lev_lab as LL

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"

UNDERS = ["QQQ", "SOXX", "SPY", "NVDA", "TSLA"]   # 미리 고정
LS = [2, 3]
SMA_N = 200
START, END = "2015-01-02", "2026-10-09"
YEARS = {"2018": ("2018-01-02", "2018-12-31"), "2020": ("2020-01-02", "2020-12-31"), "2022": ("2022-01-03", "2022-12-30"),
         "2025": ("2025-01-02", "2025-12-31"), "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09")}
OUTLIER = 0.05                                     # 실제-합성 일수익 차 문턱(미리 고정)
INDEX_UNDERS = {"QQQ", "SOXX", "SPY", "XLK", "IWM", "XBI"}


def regime_stats(under: str, lo: str, hi: str) -> list[dict]:
    u = LL.load()[under]
    c = u.c
    r = np.full(len(c), np.nan); r[1:] = c[1:] / c[:-1] - 1
    sma = np.full(len(c), np.nan)
    cs = np.cumsum(np.insert(c, 0, 0.0))
    sma[SMA_N - 1:] = (cs[SMA_N:] - cs[:-SMA_N]) / SMA_N
    above = np.full(len(c), np.nan)
    above[1:] = np.where(np.isnan(sma[:-1]), np.nan, (c[:-1] > sma[:-1]).astype(float))   # t-1 정보
    m = (u.days >= lo) & (u.days <= hi) & ~np.isnan(above) & ~np.isnan(r)
    rows = []
    synths = {L: LL.synth(under, L) for L in LS}
    for name, flag in (("200일선 위", 1.0), ("200일선 아래", 0.0)):
        mm = m & (above == flag)
        x = r[mm]
        if len(x) < 20:
            continue
        idx = np.where(mm)[0]
        prev_ok = np.isin(idx - 1, idx)
        ac = float(np.corrcoef(r[idx[prev_ok] - 1], r[idx[prev_ok]])[0, 1]) if prev_ok.sum() > 20 else float("nan")
        row = {"기초": under, "구간": f"{lo}~{hi}", "레짐": name, "일수": int(len(x)), "비중": float(mm.sum() / m.sum()),
               "연평균_산술": float(x.mean() * 252), "연변동성": float(x.std(ddof=1) * np.sqrt(252)),
               "연_로그": float(np.log1p(x).mean() * 252), "1일자기상관": ac}
        rf = LL.rf_series().reindex(u.days).ffill().fillna(0.0).to_numpy()
        row["최적배수_(μ-r)/σ²"] = float((row["연평균_산술"] - rf[mm].mean()) / row["연변동성"] ** 2)
        for L in LS:
            sc = synths[L].c
            rl = np.full(len(sc), np.nan); rl[1:] = sc[1:] / sc[:-1] - 1
            row[f"손실근사_{L}배"] = float((L * L - L) / 2 * row["연변동성"] ** 2)
            row[f"합성{L}배_연_로그"] = float(np.log1p(rl[mm]).mean() * 252)
            row[f"단순{L}배_연_로그"] = float(L * row["연_로그"])
        rows.append(row)
    return rows


def tracking_diag(product: str) -> dict | None:
    info = LL.products()[product]
    a = LL.load().get(product)
    if a is None:
        return None
    s = LL.synth(info["under"], info["L"])
    common = np.intersect1d(a.days, s.days)
    if len(common) < 60:
        return None
    ia, is_ = np.searchsorted(a.days, common), np.searchsorted(s.days, common)
    ra = a.c[ia][1:] / a.c[ia][:-1] - 1
    rs = s.c[is_][1:] / s.c[is_][:-1] - 1
    d = np.log1p(ra) - np.log1p(rs)
    big = np.abs(ra - rs) > OUTLIER
    yrs = len(d) / 252
    worst = np.argsort(d)[:3]
    return {"상품": product, "기초": info["under"], "L": info["L"], "시작": str(common[0]), "일수": int(len(d)),
            "연추적차_전체": float(np.expm1(d.sum() / yrs)),
            "이상일수": int(big.sum()), "이상일_로그합": float(d[big].sum()),
            "연추적차_이상일제외": float(np.expm1(d[~big].sum() / yrs)),
            "최악3일": [{"날짜": str(common[1:][k]), "실제": float(ra[k]), "합성": float(rs[k])} for k in worst]}


def main():
    out = {"branch": "LITLEV", "date": "2026-10-11", "grid": {"기초": UNDERS, "L": LS, "SMA": SMA_N, "기간": [START, END],
           "해별": list(YEARS), "이상일문턱": OUTLIER}, "regime_full": [], "regime_years": [], "tracking": []}
    for u in UNDERS:
        out["regime_full"] += regime_stats(u, START, END)
        for y, (lo, hi) in YEARS.items():
            for row in regime_stats(u, lo, hi):
                row["해"] = y
                out["regime_years"].append(row)
    singles = [p for p, i in LL.products().items() if i["L"] == 2 and i["under"] not in INDEX_UNDERS]
    for p in sorted(singles):
        t = tracking_diag(p)
        if t:
            out["tracking"].append(t)
    # validate()의 '연추적차'(CAGR 차, %p)와 비율 기준 연 추적차(보수처럼 빼야 하는 값)의 차이
    for p in ["NVDL", "MUU", "AMDL", "TSLT", "MSTX", "TQQQ", "SOXL", "UPRO"]:
        v = LL.validate(p)
        yrs = v["일수"] / 252
        ca, cs = 1 + v["실제_누적"], 1 + v["합성_누적"]
        out.setdefault("cagr_vs_ratio", []).append({"상품": p, "CAGR차": v["연추적차"], "실제CAGR": ca ** (1 / yrs) - 1,
                                                    "합성CAGR": cs ** (1 / yrs) - 1, "비율기준_연": (ca / cs) ** (1 / yrs) - 1})
    # 실제 상품의 구간 수익(기초 1배와 나란히): 2026 앞·뒤, 2025, 폭락 구간
    wins = {"2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"), "2025": ("2025-01-02", "2025-12-31"),
            "2022": ("2022-01-03", "2022-12-30"), "2020-02-19~03-23": ("2020-02-19", "2020-03-23"),
            "2018-10-01~12-24": ("2018-10-01", "2018-12-24")}
    d = LL.load()
    out["windows_actual"] = []
    for w, (lo, hi) in wins.items():
        row = {"구간": w}
        for sym in ["QQQ", "TQQQ", "SOXX", "SOXL", "SPY", "UPRO"]:
            b = d[sym]; i0, i1 = b.idx(lo), b.idx(hi, "right") - 1
            row[sym] = float(b.c[i1] / b.c[i0 - 1] - 1)   # 직전 거래일 종가 → 구간 마지막 종가
        out["windows_actual"].append(row)
    clean = [t["연추적차_이상일제외"] for t in out["tracking"] if t["이상일수"] <= 5]
    out["tracking_summary"] = {"단일2배_상품수": len(out["tracking"]), "이상일5일이하_상품수": len(clean),
                               "비율기준_중앙값": float(np.median(clean)), "하위25%": float(np.percentile(clean, 25)),
                               "상위25%": float(np.percentile(clean, 75))}
    (OUT / "p4_LITLEV.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("== 전체 2015~2026-10-09 ==")
    for r in out["regime_full"]:
        print(f"{r['기초']:5s} {r['레짐']:7s} 비중 {r['비중']:.0%} 연로그 {r['연_로그']:+.1%} 변동성 {r['연변동성']:.1%} "
              f"자기상관 {r['1일자기상관']:+.3f} 최적배수 {r['최적배수_(μ-r)/σ²']:.2f} | 2배 손실근사 {r['손실근사_2배']:.1%} 합성 {r['합성2배_연_로그']:+.1%} (단순 {r['단순2배_연_로그']:+.1%})"
              f" | 3배 손실근사 {r['손실근사_3배']:.1%} 합성 {r['합성3배_연_로그']:+.1%} (단순 {r['단순3배_연_로그']:+.1%})")
    print("== 해별(QQQ·SOXX) ==")
    for r in out["regime_years"]:
        if r["기초"] in ("QQQ", "SOXX"):
            print(f"{r['해']:6s} {r['기초']:5s} {r['레짐']:7s} 일수 {r['일수']:3d} 연로그 {r['연_로그']:+.1%} 변동성 {r['연변동성']:.1%} "
                  f"최적배수 {r['최적배수_(μ-r)/σ²']:.2f} 3배 손실근사 {r['손실근사_3배']:.1%} 합성3배 {r['합성3배_연_로그']:+.1%}")
    print("== 단일 종목 2배 추적차 ==")
    for t in out["tracking"]:
        print(f"{t['상품']:5s}({t['기초']}) {t['시작']} {t['일수']}일 연추적차 {t['연추적차_전체']:+.1%} | 이상일 {t['이상일수']}일 "
              f"로그합 {t['이상일_로그합']:+.2f} → 제외 시 {t['연추적차_이상일제외']:+.1%} | 최악 {t['최악3일'][0]}")
    print("== CAGR 차 vs 비율 기준 ==")
    for c in out["cagr_vs_ratio"]:
        print(f"{c['상품']:5s} CAGR차 {c['CAGR차']:+.1%} (실제 {c['실제CAGR']:+.1%} 합성 {c['합성CAGR']:+.1%}) 비율기준 {c['비율기준_연']:+.1%}")
    print(out["tracking_summary"])


if __name__ == "__main__":
    main()
