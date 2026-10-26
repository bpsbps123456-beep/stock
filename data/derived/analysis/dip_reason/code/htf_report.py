"""htf_study.py 결과 요약.
1) 사건 분석: 지금 지표의 이격 매수 중 큰 봉 RSI가 기준 이상이었던 매수와 나머지 비교(정규장·전체, 날짜 군집 t)
2) 규칙 비교: 사용자 78종목, 기간·범위별로 지금 지표, 전일 +5%, 사용자 가설(160분·일봉 70), 상위 조합
3) 전방 확인: 앞 기간에서 고른 상위 조합이 뒤 기간에서 어땠나
사용: py htf_report.py <출력 폴더(htf_rules.csv, htf_events.parquet)>"""
import sys
import numpy as np, pandas as pd
from analyze import cl_t

R = sys.argv[1]
ev = pd.read_parquet(f"{R}/htf_events.parquet")
res = pd.read_csv(f"{R}/htf_rules.csv", encoding="utf-8-sig")
P = list(dict.fromkeys(ev["기간"]))
pd.set_option("display.width", 250)


def ev_line(v, col, th):
    x = v[col].astype(float)
    m = x >= th
    ok = x.notna()
    v, m = v[ok], m[ok]
    if m.sum() < 5:
        return f"{m.sum():4d}건 (표본 부족)"
    _, t, _ = cl_t(m.astype(float).values, v.ret.values, v.day.astype(str).values)
    return (f"{m.sum():4d}건({m.mean():.0%}) 평균 {v[m].ret.mean():+.2%} 승률 {(v[m].ret > 0).mean():.0%} "
            f"| 나머지 {v[~m].ret.mean():+.2%} {(v[~m].ret > 0).mean():.0%} | t {t:+.2f}")


print("=" * 20, "1) 사건 분석: 지금 지표 매수 중 큰 봉 RSI14 ≥ 70인 매수 vs 나머지 (127+사용자 종목 전체)")
for p in P:
    e = ev[ev["기간"] == p]
    for ses, v in (("정규장 매수", e[e.reg]), ("전체 매수", e)):
        print(f"\n[{p} · {ses} {len(v)}건]")
        for tf in ("60", "120", "160", "240", "D"):
            for kind in ("dev", "conf"):
                print(f"  {tf:>3}{'분' if tf != 'D' else '봉'} {kind:4s}: {ev_line(v, f'{tf}|14|{kind}', 70)}")
print("\n기준 민감도(정규장 매수, 160분·일봉, dev/conf):")
for p in P:
    e = ev[(ev["기간"] == p) & ev.reg]
    for tf in ("160", "D"):
        for kind in ("dev", "conf"):
            print(f"  [{p}] {tf} {kind}: " + " · ".join(
                f"≥{th}: {e[e[f'{tf}|14|{kind}'] >= th].ret.mean():+.2%}({(e[f'{tf}|14|{kind}'] >= th).sum()}건)" for th in (60, 65, 70, 75, 80))
                  + f" | 나머지(<60) {e[e[f'{tf}|14|{kind}'] < 60].ret.mean():+.2%}")
    print(f"  [{p}] 길이 7·21(≥70, 정규장): " + " · ".join(
        f"{tf}|{n}|{k} {ev_line(e, f'{tf}|{n}|{k}', 70)}" for tf in ("160", "D") for n in (7, 21) for k in ("dev",)))

print("\n" + "=" * 20, "2) 규칙 비교 (사용자 78종목) — 합 = 거래 수익률 합, 차이 = 지금 지표 대비, t = 날짜별 짝 비교")
cols = ["규칙", "건수", "평균", "승률", "최악", "합", "차이", "t", "나은날", "나쁜날", "유전체5", "TEM"]
fmt = {"평균": "{:+.2%}", "승률": "{:.0%}", "최악": "{:+.1%}", "합": "{:+.1f}", "차이": "{:+.2f}", "t": "{:+.2f}",
       "유전체5": "{:+.1%}", "TEM": "{:+.1%}"}


def show(df):
    d = df[cols].copy()
    for k, f in fmt.items():
        d[k] = d[k].map(lambda v: "" if pd.isna(v) else f.format(v))
    return d.to_string(index=False)


u = res[res["종목"] == "사용자 78"]
for p in P:
    for scope in ("24시간", "정규장만"):
        v = u[(u["기간"] == p) & (u["범위"] == scope)]
        fixed = v[v["규칙"].isin(["지금 지표", "전일 +5% 금지"])]
        hyp = v[(v["큰봉"].isin(["160", "D"])) & (v["길이"] == 14) & (v["기준"] == 70) & (v["조심"].isin(["안 삼", "-4%"]))]
        top = v[v["큰봉"].notna()].sort_values("차이", ascending=False).head(8)
        print(f"\n[{p} · {scope}]\n-- 기준과 사용자 가설(160분·일봉, RSI14 ≥70)\n{show(pd.concat([fixed, hyp]))}\n-- 차이 상위 8개\n{show(top)}")

print("\n" + "=" * 20, "3) 전방 확인: 앞 기간에서 고른 조합 → 뒤 기간 (사용자 78종목)")
for scope in ("24시간", "정규장만"):
    a = u[(u["기간"] == P[0]) & (u["범위"] == scope) & u["큰봉"].notna()].set_index("규칙")
    b = u[(u["기간"] == P[1]) & (u["범위"] == scope) & u["큰봉"].notna()].set_index("규칙")
    both = a[["차이", "t"]].join(b[["차이", "t", "TEM", "유전체5"]], lsuffix="_앞", rsuffix="_뒤").dropna(subset=["차이_앞", "차이_뒤"])
    n = len(both)
    up2 = ((both["차이_앞"] > 0) & (both["차이_뒤"] > 0)).sum()
    print(f"\n[{scope}] 조합 {n}개 중 두 기간 모두 나아진 것 {up2}개, 앞 기간 상위 10개의 뒤 기간 차이 평균 {both.sort_values('차이_앞', ascending=False).head(10)['차이_뒤'].mean():+.2f}"
          f" (전체 평균 {both['차이_뒤'].mean():+.2f}), 앞·뒤 차이 순위상관 {both['차이_앞'].corr(both['차이_뒤'], method='spearman'):+.2f}")
    print(both.sort_values("차이_앞", ascending=False).head(10).round(2).to_string())
