"""htf_ema_study.py 결과 요약: 1) 사건 분석(160분 구조·RSI별 이격 매수 성적) 2) 규칙 비교(사용자 78종목) 3) 두 기간 일관성.
사용: py htf_ema_report.py <출력 폴더>"""
import sys
import numpy as np, pandas as pd
from analyze import cl_t

R = sys.argv[1]
ev = pd.read_parquet(f"{R}/htf_ema_events.parquet")
res = pd.read_csv(f"{R}/htf_ema_rules.csv", encoding="utf-8-sig")
P = list(dict.fromkeys(ev["기간"]))
pd.set_option("display.width", 250)


def state(e, kind, x=0.03):
    sp = e[f"spread_{kind}"]
    return np.select([~e[f"valid_{kind}"], e[f"up_{kind}"] & (sp >= x), e[f"up_{kind}"], e[f"mix_{kind}"], e[f"down_{kind}"]],
                     ["(EMA 없음)", "정배열·넓음(≥3%)", "정배열·좁음(<3%)", "엉킴", "역배열"], "?")


def grp(v, key):
    g = v.groupby(key).ret.agg(["count", "mean", lambda r: (r > 0).mean(), "min"])
    g.columns = ["건수", "평균", "승률", "최악"]
    return g.assign(평균=g.평균.map("{:+.2%}".format), 승률=g.승률.map("{:.0%}".format), 최악=g.최악.map("{:+.1%}".format))


print("=" * 20, "1) 사건 분석: 지금 지표 매수를 160분 구조(진행 중 값)·RSI로 나눔 (사용자+127종목)")
for p in P:
    e = ev[ev["기간"] == p].copy()
    e["구조"] = state(e, "dev")
    e["RSI≥70"] = np.where(e.rsi_dev >= 70, "RSI≥70", "RSI<70")
    for ses, v in (("정규장 매수", e[e.reg]), ("전체 매수", e)):
        print(f"\n[{p} · {ses} {len(v)}건] 구조별")
        print(grp(v, "구조").to_string())
        print(f"  RSI≥70(진행 중)만: {int((v.rsi_dev >= 70).sum())}건")
        if (v.rsi_dev >= 70).sum():
            print(grp(v[v.rsi_dev >= 70], "구조").to_string())
        vc = v.assign(구조=state(v, "conf"))
        print(f"  RSI≥70(완성 봉)만: {int((vc.rsi_conf >= 70).sum())}건")
        if (vc.rsi_conf >= 70).sum():
            print(grp(vc[vc.rsi_conf >= 70], "구조").to_string())
        for kind in ("dev", "conf"):
            for x in (0.01, 0.02, 0.03, 0.05):
                m = (v[f"rsi_{kind}"] >= 70) & v[f"valid_{kind}"] & (~v[f"up_{kind}"] | (v[f"spread_{kind}"] < x))
                if m.sum() >= 5:
                    _, t, _ = cl_t(m.astype(float).values, v.ret.values, v.day.astype(str).values)
                    print(f"  사용자 조건[{kind}] RSI≥70 & (정배열 아님 또는 간격<{x:.0%}): {m.sum()}건 평균 {v[m].ret.mean():+.2%} "
                          f"승률 {(v[m].ret > 0).mean():.0%} | 나머지 {v[~m].ret.mean():+.2%} | t {t:+.2f}")
                else:
                    print(f"  사용자 조건[{kind}] RSI≥70 & (정배열 아님 또는 간격<{x:.0%}): {m.sum()}건 (표본 부족)")

print("\n" + "=" * 20, "2) 규칙 비교 (사용자 78종목, 차이 = 지금 지표 대비 합 차이)")
u = res[res["종목"] == "사용자 78"].copy()
u["기간·범위"] = u["기간"].str[:1] + "·" + u["범위"]
piv = u.pivot_table(index="규칙", columns="기간·범위", values="차이", sort=False)
piv = piv[[c for c in ("앞·24시간", "앞·정규장만", "뒤·24시간", "뒤·정규장만") if c in piv]]
tem = u[u["기간"].str.startswith("뒤")].pivot_table(index="규칙", columns="범위", values="TEM", sort=False)
piv["TEM(뒤·24시간)"] = tem["24시간"]
key = [r for r in piv.index if ("RSI≥70 & " in r) or r.startswith("[dev] 정배열") or r.startswith("[dev] 역배열") or r.startswith("[dev] 엉킴")
       or r.startswith("[dev] 간격") or r.startswith("[dev] (대조)")]
print(piv.loc[[r for r in key if "-4%" not in r]].round(2).to_string())
print("\n-- 기준 60·65·75 (안 삼)")
print(piv.loc[[r for r in piv.index if any(f"RSI≥{t} " in r for t in (60, 65, 75)) and r.endswith("안 삼")]].round(2).to_string())
print("\n-- '-4% 깊게 사기' 요약: 사용자 조건(RSI≥70 & 정배열 아님 또는 간격<X)")
print(piv.loc[[r for r in piv.index if "RSI≥70 & 정배열 아님" in r and r.endswith("-4%")]].round(2).to_string())

print("\n" + "=" * 20, "3) 일관성: 두 기간·두 범위 모두 플러스인 규칙")
cols = ["앞·24시간", "앞·정규장만", "뒤·24시간", "뒤·정규장만"]
allpos = piv[(piv[cols] > 0).all(axis=1)]
print(f"규칙 {len(piv) - 1}개 중 네 칸 모두 플러스 {len(allpos)}개")
print(allpos.round(2).sort_values("뒤·24시간", ascending=False).to_string())
