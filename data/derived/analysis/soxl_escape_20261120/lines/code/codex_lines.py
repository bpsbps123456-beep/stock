"""SOXL E–H research. Choose one winner per family on SEL, then evaluate three."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

import engine as E

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TF = (30, 60, "D")
SESS = ("reg", "ext")
WINDOWS = ("SEL", "CONF", "Y2025", "B1", "B2")


def settings():
    for sess in SESS:
        minutes = 390 if sess == "reg" else 960
        for tf in TF:
            basic = dict(tf=tf, sess=sess)
            def length(days):
                return days if tf == "D" else int(np.ceil(minutes / tf)) * days
            if tf != "D":
                for n in (1, 5):
                    for m in (1, 3):
                        for touch in (False, True):
                            yield dict(family="E", kind="channel", n=length(n), m=length(m), touch=touch, **basic)
            for days in (2, 5):
                for k in (1.5, 3.0):
                    for kind in ("chandelier", "supertrend", "keltner", "bollinger"):
                        yield dict(family="G", kind=kind, n=length(days), k=k, **basic)
            for anchor in ("week", "month"):
                yield dict(family="H", kind="vwap", anchor=anchor, **basic)
    for n in (2, 5, 10):
        for m in (1, 3, 5):
            yield dict(family="F", kind="daily_low", n=n, m=m, touch=True, tf="D", sess="reg")


def supertrend(b, atr, k):
    mid = (b.High + b.Low) / 2
    up = (mid + k * atr).to_numpy(copy=True)
    down = (mid - k * atr).to_numpy(copy=True)
    close = b.Close.to_numpy()
    bull = np.zeros(len(b), dtype=bool)
    for i in range(1, len(b)):
        if close[i-1] <= up[i-1]:
            up[i] = min(up[i], up[i-1])
        if close[i-1] >= down[i-1]:
            down[i] = max(down[i], down[i-1])
        bull[i] = close[i] >= down[i] if bull[i-1] else close[i] > up[i]
    return bull


def anchored(b, anchor, sess):
    raw = E.load()
    if sess == "reg":
        raw = raw.loc[raw.reg]
    idx = raw.index.tz_localize(None)
    key = idx.to_period("W-SUN" if anchor == "week" else "M")
    typical = (raw.High + raw.Low + raw.Close) / 3
    volume = raw.Volume.groupby(key).cumsum()
    line = (typical * raw.Volume).groupby(key).cumsum() / volume.replace(0, np.nan)
    last = raw.index.searchsorted(b.end, side="left") - 1
    return pd.Series(line.to_numpy()[last], index=b.index)


def simulate(params, awake=False):
    b = E.bars("SOXL", params["tf"], params["sess"])
    kind = params["kind"]
    n = params.get("n")
    k = params.get("k")
    close = b.Close
    signals = {}
    if kind in ("channel", "daily_low"):
        lower = b.Low.rolling(n).min().shift(1)
        upper = b.High.rolling(params["m"]).max().shift(1)
        if params["touch"]:
            signals["exit_touch"] = lower
        else:
            signals["exit_close"] = close < lower
        entry = close > upper
    elif kind == "supertrend":
        bull = supertrend(b, E.atr(b, n), k)
        signals["exit_close"] = ~bull
        entry = bull
    else:
        if kind == "chandelier":
            atr = E.atr(b, n)
            lower = b.High.rolling(n).max() - k * atr
            upper = b.Low.rolling(n).min() + k * atr
        elif kind == "keltner":
            upper = E.ema(close, n)
            lower = upper - k * E.atr(b, n)
        elif kind == "bollinger":
            upper = close.rolling(n).mean()
            lower = upper - k * close.rolling(n).std(ddof=0)
        else:
            upper = lower = anchored(b, params["anchor"], params["sess"])
        signals["exit_close"] = close < lower
        entry = close > upper
    if awake:
        raw = E.load()
        ticks = raw.index[raw.reg] if params["sess"] == "reg" else raw.index
        fills = ticks.searchsorted(b.end)
        korea = ticks[np.minimum(fills, len(ticks)-1)].tz_convert("Asia/Seoul")
        entry = entry & (fills < len(ticks)) & (korea.hour >= 8)
    return E.simulate(b, **signals, entry_close=entry, exec_sess=params["sess"])


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(x) for x in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def describe(p):
    tf = "일" if p["tf"] == "D" else str(p["tf"]) + "분"
    sess = "정규장" if p["sess"] == "reg" else "연장 포함"
    kind = p["kind"]
    n, k = p.get("n"), p.get("k")
    if kind in ("channel", "daily_low"):
        exit_rule = "첫 1분 저가 접촉" if p["touch"] else "완성봉 종가 엄격 하회"
        rule = f"직전 {n}봉 최저가에 {exit_rule} 시 대피, 직전 {p['m']}봉 최고가를 종가로 엄격 상회하면 복귀."
    elif kind == "vwap":
        anchor = "월요일 시작 주간" if p["anchor"] == "week" else "달력 월간"
        rule = f"{anchor} 첫 세션에서 누적을 초기화한 1분 HLC3×거래량 합/거래량 합을 봉 끝에서 읽고, 종가 하회 시 대피·상회 시 복귀."
    elif kind == "supertrend":
        rule = (f"중간값 HL2 ± {k}×ATR({n})을 기본 상·하단으로 삼는다. 직전 종가가 직전 상단 이하이면 상단은 기본값과 직전 상단 중 작은 값, "
                "직전 종가가 직전 하단 이상이면 하단은 기본값과 직전 하단 중 큰 값이다. 초기 하락, 하락 중 종가가 상단 초과면 상승, 상승 중 종가가 하단 미만이면 하락. 하락이면 대피·상승이면 복귀.")
    elif kind == "chandelier":
        rule = f"현재 봉 포함 {n}봉 최고가−{k}×ATR({n}) 종가 하회 시 대피, {n}봉 최저가+{k}×ATR({n}) 종가 상회 시 복귀."
    elif kind == "bollinger":
        rule = f"현재 봉 포함 종가 SMA({n})−{k}×모표준편차({n}) 종가 하회 시 대피, 같은 SMA 상회 시 복귀."
    else:
        rule = f"현재 봉 포함 종가 EMA({n})−{k}×ATR({n}) 종가 하회 시 대피, 같은 EMA 상회 시 복귀."
    return f"{p['family']} {kind} · SOXL {sess} {tf}봉. {rule}"


def main():
    grid = list(settings())
    rows = [dict(params=p, SEL=E.evaluate(*simulate(p), "SEL")) for p in grid]
    qualified = [r for r in rows if r["SEL"]["switches"] >= 10 and 20 <= r["SEL"]["expo%"] <= 90]
    qualified.sort(key=lambda r: -r["SEL"]["tv%p"])
    winners = {}
    for row in qualified:
        winners.setdefault(row["params"]["family"], row)
    finalists = []
    for row in list(winners.values())[:3]:
        params = row["params"]
        path = simulate(params)
        finalists.append(dict(params=params, definition=describe(params), scores=E.score_all(*path),
                              placebo={w: E.placebo(*path[:2], w) for w in ("SEL", "CONF")},
                              cost_001=E.score_all(*path, cost=0.001),
                              awake_entries=E.score_all(*simulate(params, awake=True))))
    counts = dict(Counter(p["family"] for p in grid))
    result = dict(count=counts, total=len(rows), qualified=len(qualified), ranking="SEL tv descending, one per E/F/G/H; ties in grid order",
                  grid=rows, finalists=finalists)
    (RESULTS / "codex_lines.json").write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summary(finalists, counts, len(qualified))


def summary(top, counts, qualified):
    npos = sum(x["scores"]["CONF"]["tv%p"] > 0 for x in top)
    resilient = [x for x in top if all(x["cost_001"][w]["tv%p"] > 0 for w in ("SEL", "CONF"))]
    details = []
    for x in resilient:
        details.append(f"{x['params']['kind']}의 확인 구간 비용 후 시점 값은 {x['cost_001']['CONF']['tv%p']:+.2f}로그%p였고, 비용 전 기준으로 가장 좋았던 3일을 빼면 {x['scores']['CONF']['tv_ex_top3%p']:+.2f}로그%p였다.")
    lines = [f"선택 구간에서 고른 세 계열 중 확인 구간에서 시점 값이 양수인 것은 {npos}개였다. 편도 0.1% 비용을 적용해 두 구간 모두 양수인 후보는 {len(resilient)}개였다.", *details,
             "확인 구간은 2026-10-15~11-19의 단일 급락 구간이며 세 후보 모두 2025년 보조 구간의 시점 값은 음수였다.",
             "", "후보 | 구간 | 수익% | 보유% | 고정% | TV(로그%p) | 일별 t | 상위3일 제외TV | MDD% | 고정MDD% | 보유MDD% | 노출% | 전환 | 위약 수익/낙폭 백분위 | 비용후 수익%/TV | 깨어 있을 때 복귀 TV"]
    for x in top:
        for w in WINDOWS:
            s = x["scores"][w]
            pl = x["placebo"].get(w)
            pc = f"{pl['ret_pct']}/{pl['mdd_pct']}" if pl else "—"
            cost = x["cost_001"][w]
            vals = [s[key] for key in ("ret%", "bh%", "fixed%", "tv%p", "tv_t", "tv_ex_top3%p", "mdd%", "fixed_mdd%", "bh_mdd%", "expo%", "switches")]
            lines.append(" | ".join(map(str, [x["params"]["kind"], w, *vals, pc, f"{cost['ret%']}/{cost['tv%p']}", x["awake_entries"][w]["tv%p"]])))
    lines.extend(["", f"SEL(1/2~10/14) {sum(counts.values())}개(E {counts['E']}, F {counts['F']}, G {counts['G']}, H {counts['H']}) 중 자격 충족 {qualified}개. 계열별 SEL TV 1위를 추려 상위 세 계열을 선택했다. 다중 탐색의 선택편향이 있다.",
                  "E는 30·60분, 2세션, 저가창 1·5일, 고가창 1·3일, 접촉·종가 2방식. F는 정규 일봉 저가 2·5·10일 접촉, 종가가 직전 1·3·5일 고가 상회 시 복귀.",
                  "G는 30·60분·일봉, 2세션, 4지표, 2·5일, 배수 1.5·3. H는 같은 봉·세션의 주·월 앵커. 일수는 정상 세션당 봉수(올림)로 환산했다.",
                  "위약은 경로 회전 200회. 비용은 전환당 로그수익 0.001 차감. 시간 민감도는 한국 00~08시 복귀 신호를 건너뛰며 대피 체결은 원래대로 둔다. 수동 종가 대피는 수면 중 제한될 수 있다.",
                  "B1·B2는 확인 구간의 앞·뒤 절반(11/04·11/05 경계), 2025년은 장세 전환 위험 점검용이다.", "", "Pine 재현 정의", *[x["definition"] for x in top],
                  "뉴욕시간 정규 09:30 또는 연장 04:00에 봉 정렬. 2025-01-02 보유 시작, 한 봉 한 전환, 보유 중 대피만·현금 중 복귀만 평가. 종가 조건은 같은 세션의 다음 1분 시가, 스탑은 min(첫 접촉 1분 시가, 선). ATR은 첫 TR부터 alpha=1/N 재귀, EMA는 첫 종가부터 2/(N+1) 재귀."])
    (RESULTS / "codex_lines_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
