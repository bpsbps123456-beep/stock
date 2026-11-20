"""Independent SOXL replay, then adversarial checks on the six locked specifications."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PERIODS = {"SEL": ("2026-01-02", "2026-10-14"), "CONF": ("2026-10-15", "2026-11-19"),
           "Y2025": ("2025-03-03", "2025-12-31")}
FIELDS = ("ret%", "tv%p", "mdd%", "expo%", "switches")


@dataclass(frozen=True)
class Rule:
    tf: object
    sess: str
    fast: int
    slow: int
    stop: bool
    exit_band: float = 0
    entry_band: float = 0
    channel: bool = False


RULES = {"C1": Rule(240, "ext", 10, 50, True), "C2": Rule(5, "ext", 200, 200, False, .02, .02),
         "C3": Rule("D", "reg", 10, 20, True), "C4": Rule("D", "reg", 10, 20, False),
         "C5": Rule("D", "reg", 10, 10, True, .01), "channel": Rule(30, "ext", 32, 32, True, channel=True)}
REFERENCE = {
    "C1": [[381.78,110.84,-41.05,54.4,50],[-12.42,-12.02,-13.54,1.4,4],[81.65,24.12,-25.72,57.7,62]],
    "C2": [[440.27,119.57,-43.15,57.6,168],[-11.73,23.13,-26.57,40.6,34],[64.34,13.43,-38.74,58.8,168]],
    "C3": [[283.93,93.78,-35.34,47.8,26],[0,0,0,0,0],[67.44,18.77,-32.58,53.2,32]],
    "C4": [[237.65,72.67,-42.77,57.4,21],[0,0,0,0,0],[151.24,52.58,-33.93,64.1,19]],
    "C5": [[253.24,87.56,-44.61,45.3,44],[-2.6,.74,-10.23,3.8,2],[52.7,12.56,-23.76,48.3,50]]}


def replay(raw, rule):
    minutes = np.asarray(raw.index.hour * 60 + raw.index.minute)
    start, end = (570, 960) if rule.sess == "reg" else (240, 1200)
    eligible = np.flatnonzero((minutes >= start) & (minutes < end))
    day = np.asarray(raw.index.dayofyear + 366 * raw.index.year)
    slot = (minutes - start) // (1440 if rule.tf == "D" else rule.tf)
    keys = (day * 1440 + slot)[eligible]
    limits = np.r_[0, 1 + np.flatnonzero(np.diff(keys)), len(keys)]
    opening, low, close = (raw[k].to_numpy(float) for k in ("Open", "Low", "Close"))
    last_fill = -1
    held = True
    switch = np.zeros(len(raw), dtype=int)
    adj = np.zeros(len(raw))
    fast = slow = close[eligible[limits[1]-1]]
    for i, (left, right) in enumerate(zip(limits[:-1], limits[1:])):
        level = min(fast, slow) * (1 - rule.exit_band)
        c = close[eligible[right-1]]
        fast += (c-fast) * 2 / (rule.fast+1)
        slow += (c-slow) * 2 / (rule.slow+1)
        if held and rule.stop:
            if i == 0:
                continue
            bars = eligible[left:right]
            hits = bars[(bars > last_fill) & (low[bars] <= level)]
            if not len(hits):
                continue
            fill = hits[0]
            adj[fill] = min(np.log(level / opening[fill]), 0)
        else:
            trigger = c < min(fast, slow)*(1-rule.exit_band) if held else c > max(fast, slow)*(1+rule.entry_band)
            if not trigger or right == len(eligible):
                continue
            fill = eligible[right]
        held = not held
        last_fill = fill
        switch[fill] = 1 if held else -1
    return 1+switch.cumsum(), adj, switch


def independent():
    raw = pd.read_parquet(ROOT / "cache/SOXL_1m.parquet")
    raw = raw.loc[~raw.index.duplicated()].sort_index()
    logs = np.log(raw.Open.to_numpy(float))
    checks = {}
    for name, expected in REFERENCE.items():
        pos, adj, switch = replay(raw, RULES[name])
        checks[name] = {}
        for (win, dates), reference in zip(PERIODS.items(), expected):
            left, right = raw.index.slice_locs(*dates)
            sl = slice(left, right-1)
            change = np.diff(logs[left:right])
            gain = change * pos[sl] + adj[sl]
            eq = np.exp(gain.cumsum())
            f = pos[sl].mean()
            values = [round(100*np.expm1(gain.sum()),2), round(100*(gain.sum()-f*change.sum()),2),
                      round(100*(eq/np.maximum.accumulate(eq)-1).min(),2),round(100*f,1),int(abs(switch[sl]).sum())]
            delta = np.round(np.array(values)-reference, 6)
            checks[name][win] = {"actual":dict(zip(FIELDS,values)), "expected":dict(zip(FIELDS,reference)),
                                 "delta": dict(zip(FIELDS,delta)), "match":bool((abs(delta)<=.05).all())}
    return checks


def engine_path(E, rule, awake=False):
    b = E.bars("SOXL", rule.tf, rule.sess)
    if rule.channel:
        lower = b.Low.rolling(rule.fast).min().shift()
        upper = b.High.rolling(rule.slow).max().shift()
    else:
        averages = pd.concat([E.ema(b.Close, n) for n in (rule.fast, rule.slow)], axis=1)
        lower = averages.min(axis=1)
        upper = averages.max(axis=1)
        if rule.stop:
            lower = lower.shift()
    lower = lower * (1-rule.exit_band)
    entry = b.Close > upper*(1+rule.entry_band)
    if awake:
        d = E.load()
        ticks = d.index[d.reg] if rule.sess == "reg" else d.index
        next_tick = ticks.searchsorted(b.end)
        hours = ticks[np.minimum(next_tick,len(ticks)-1)].tz_convert("Asia/Seoul").hour
        entry &= (next_tick < len(ticks)) & (hours >= 8)
    return E.simulate(b, exit_touch=lower if rule.stop else None,
                      exit_close=None if rule.stop else b.Close < lower,
                      entry_close=entry, exec_sess=rule.sess)


def neighbors(rule):
    yield "exit_mode", replace(rule, stop=not rule.stop)
    yield "session", replace(rule, sess="ext" if rule.sess=="reg" else "reg")
    scale = [3, 5, 15, 30, 60, 120, 240, "D"]
    i = scale.index(rule.tf)
    for tf in scale[max(0,i-1):i] + scale[i+1:i+2]:
        yield f"tf_{tf}", replace(rule,tf=tf)
    steps = {10:2,20:5,32:8,50:10,200:50}
    single = rule.fast==rule.slow and not rule.channel
    for side in (["single"] if single else ["fast","slow"]):
        n = rule.fast if side=="single" else getattr(rule,side)
        for length in (n-steps[n],n+steps[n]):
            updates = {"fast":length,"slow":length} if single else {side:length}
            yield f"{side}_{length}", replace(rule, **updates)
    for side in ("exit_band", "entry_band"):
        base = getattr(rule,side)
        for value in sorted({max(0,base-.005),base+.005}-{base}):
            yield f"{side}_{value}", replace(rule, **{side:value})
    if rule.exit_band == rule.entry_band and rule.exit_band > 0:
        for width in (rule.exit_band-.005, rule.exit_band+.005):
            yield f"symmetric_band_{width}", replace(rule,exit_band=width,entry_band=width)


def event_audit(E, path):
    pos, corr, switches = path
    d = E.load()
    opens = d.Open.to_numpy()
    last = len(d)-1
    fills = []
    episodes = []
    for i in range(0,len(switches),2):
        start = int(switches[i])
        closed = i+1 < len(switches)
        end = int(switches[i+1]) if closed else last
        sale = float(opens[start]*np.exp(corr[start]))
        buy = float(opens[end])
        record = dict(exit_et=d.index[start].isoformat(),exit_price=sale,
                      entry_et=d.index[end].isoformat() if closed else None,entry_price=buy if closed else None,
                      mark_et=d.index[end].isoformat(),mark_price=buy,
                      hold_return_pct=100*(buy/sale-1), avoided_log_pp=100*np.log(sale/buy))
        fills.append(record)
        episodes.append((start,end,sale,buy))
    crashes = {}
    for label,dates in (("fall_2026",("2026-10-01","2026-11-19")),("spring_2025",("2025-02-18","2025-04-30"))):
        left,right = E._win(dates)
        rows = []
        for (a,b,sale,buy),detail in zip(episodes,fills):
            if a > right or b < left:
                continue
            lo,hi = max(a,left),min(b,right)
            x = sale if a==lo else opens[lo]
            y = opens[hi]
            rows.append(dict(exit_et=detail["exit_et"],exit_price=sale,
                             entry_et=detail["entry_et"] if b<=right else None,
                             entry_price=detail["entry_price"] if b<=right else None,
                             eventual_entry_et=detail["entry_et"],eventual_entry_price=detail["entry_price"],
                             full_hold_return_pct=detail["hold_return_pct"] if detail["entry_et"] else None,
                             from_et=d.index[lo].isoformat(),to_et=d.index[hi].isoformat(),
                             from_price=x,to_price=y,hold_return_pct=100*(y/x-1)))
        crashes[label] = rows
    a,z = E._win("SEL")
    spans = []
    for left,right,sale,buy in episodes:
        lo,hi = max(a,left),min(z,right)
        if lo>=hi:
            continue
        x = sale if lo==left else opens[lo]
        val = np.log(x/opens[hi])
        spans.append(dict(start=d.index[lo].isoformat(), end=d.index[hi].isoformat(),minutes_observed=hi-lo, avoided_log_pp=100*val))
    spans.sort(key=lambda r:-r["avoided_log_pp"])
    bh = np.log(opens[z]/opens[a])
    p = pos[a:z]
    tv = 100*((p*E._base()["r"][a:z]+corr[a:z]).sum()-p.mean()*bh)
    removed = {}
    for n in (1,3):
        top = spans[:n]
        fixed = tv-sum(v["avoided_log_pp"] for v in top)
        corrected = fixed-100*sum(v["minutes_observed"] for v in top)/(z-a)*bh
        removed[str(n)] = dict(fixed_SEL_exposure_tv=fixed,rehold_recomputed_tv=corrected, removed=top)
    return dict(all_cash_intervals=fills,crash_tables=crashes,SEL_intervals_ranked=spans,SEL_remove_best=removed)


def audit(checks):
    import engine as E
    result = {"independent":checks, "candidates":{}}
    for name,rule in RULES.items():
        path = engine_path(E,rule)
        score = E.score_all(*path)
        if name in checks:
            for w in PERIODS:
                assert all(abs(score[w][k]-checks[name][w]["actual"][k]) <= .05 for k in FIELDS)
        else:
            assert [score[w]["tv%p"] for w in PERIODS] == [55.79,29.32,-25.85]
        adj = {}
        for label,neighbor in neighbors(rule):
            alternate = engine_path(E,neighbor)
            values = {w:E.evaluate(*alternate,w)["tv%p"] for w in PERIODS}
            adj[label] = dict(rule=asdict(neighbor), tv=values,
                              sign={w:int(np.sign(v)) for w,v in values.items()})
        a,z = E._win("SEL")
        usual = path[0][a:z].mean()
        a,z = E._win("CONF")
        r = E._base()["r"][a:z]
        tvl = 100*((path[0][a:z]*r+path[1][a:z]).sum()-usual*r.sum())
        result["candidates"][name] = dict(rule=asdict(rule),scores=score,CONF_tvL=tvl,
            placebo={label:E.placebo(*path[:2],window=w) for label,w in
                     (("SEL","SEL"),("full",("2025-03-03","2026-11-19")) )},
            neighbors=adj, costs={str(c):E.score_all(*path,cost=c) for c in (.001,.002)},
            awake_entries=E.score_all(*engine_path(E,rule,True)),events=event_audit(E,path),
            future_audit={"exit": "prior completed bar, first eligible minute touch" if rule.stop else "completed close; following eligible open",
            "entry":"previous completed 32-bar high; completed close, following open" if rule.channel else "EMA through completed close; following eligible open",
            "no_future_inputs":True})
    return result


def sanitize(x):
    if isinstance(x,np.generic):
        return sanitize(x.item())
    if isinstance(x,dict):
        return {k:sanitize(v) for k,v in x.items()}
    if isinstance(x,list):
        return [sanitize(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):
        return None
    return x


def write_json(result):
    (OUT/"codex_verify.json").write_text(json.dumps(sanitize(result),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def summarize(result, judgments, best, rationale):
    rows = result["candidates"]
    definitions = {
        "C1":"연장 240분 EMA10·50 스탑", "C2":"연장 5분 EMA200 ±2% 종가",
        "C3":"정규 일봉 EMA10·20 스탑", "C4":"정규 일봉 EMA10·20 종가",
        "C5":"정규 일봉 EMA10 −1% 스탑", "channel":"연장 30분 32봉 채널 스탑"}
    lines = [f"반박 검증에서 가장 잘 버틴 후보는 {best}({definitions[best]})다. {rationale}",
             "원자료 독립 검산 75개(5후보×3구간×5지표)는 엔진의 공개 수치와 표시 정밀도까지 일치했다.",
             "", "후보·판정 | 규칙 | TV 선택/확인/2025 | 확인 TVL | 위약 수익 백분위 선택/전체 | 이웃 TV 양수 선택/확인/2025 (모수) | 비용 0.2% TV 선택/확인/2025 | 한국시간 복귀 TV 선택/확인 | SEL 상위 1/3회 재보유 TV | 판정 근거"]
    for name,x in rows.items():
        tag,reason = judgments[name]
        x["verdict"] = {"label":tag,"reason":reason}
        def tv(obj,periods=PERIODS):
            return "/".join(f"{obj[w]['tv%p']:+.2f}" for w in periods)
        pl = x["placebo"]
        nb = x["neighbors"]
        count = "/".join(str(sum(v["tv"][w]>0 for v in nb.values())) for w in PERIODS)
        trim = x["events"]["SEL_remove_best"]
        fields = [name+" "+tag, definitions[name], tv(x["scores"]),f"{x['CONF_tvL']:+.2f}",
                  f"{pl['SEL']['ret_pct']}/{pl['full']['ret_pct']}",f"{count} ({len(nb)})",
                  tv(x["costs"]["0.002"]),tv(x["awake_entries"],["SEL","CONF"]),
                  "/".join(f"{v['rehold_recomputed_tv']:+.2f}" for v in trim.values()),reason]
        lines.append(" | ".join(fields))
    n = sum(len(x["neighbors"]) for x in rows.values())
    distinct = {tuple(asdict(rule).values()) for rule in RULES.values()}
    for rule in RULES.values():
        distinct.update(tuple(asdict(p).values()) for _,p in neighbors(rule))
    lines.extend(["",
        f"• 본체 6개와 한 변수씩 바꾼 이웃 {n}건(중복 제거 총 {len(distinct)}설정)을 시험했다. EMA 길이에 따라 ±2·5·10·50봉, 채널 ±8봉, 띠 ±0.5%p(하한 0), 대피 방식·세션 교체, 시간봉 인접 단계 변경. 탐색 뒤 추린 후보의 위약 백분위(200회)는 선택편향을 보정하지 않는다.",
        "• TV는 구간 평균 노출로 설명되는 부분을 뺀 로그수익(%p), TVL은 SEL 평균 노출을 CONF에 고정한 값이다. 확인 내내 현금인 C3·C4의 TV=0은 사전 대피의 효과와 양립하며, 그 기간의 복귀 능력은 검증되지 않았다.",
        "• 사건 제외값은 SEL과 겹치는 대피 구간에서 회피한 로그손실이 가장 큰 1·3건을 다시 보유했다고 두고 노출까지 재계산했다. JSON에는 기존 평균 노출을 고정한 값, 전체 대피 구간별 손익, 두 급락 구간의 체결·평가 시각과 가격표도 수록했다. 미복귀 구간은 마지막 시가로 평가했다.",
        "• 08~24시(한국) 밖의 복귀 신호는 건너뛰고 새 신호를 기다렸다. 대피는 원래 체결 조건이다. 스탑은 직전 완성봉 값과 접촉 1분 시가 중 낮은 가격, 종가 신호는 다음 허용 시가로 체결해 선 계산에 미래값을 쓰지 않았다.",
        "• keep은 추가 추적 후보, fragile은 비용·이웃·사건 또는 실제 복귀시간에 민감한 후보, reject는 확인 목적의 대피선으로 지지하기 어려운 후보이다. 2025는 보조 근거이며 확인 급락 사례는 한 번이다."])
    result["best"] = {"candidate":best,"reason":rationale}
    result["scope"] = {"originals":len(rows),"neighbor_runs":n,"distinct_including_originals":len(distinct),
                       "placebo_shifts":200,"event_table_timezone":"America/New_York",
                       "awake_window":"08:00 <= next eligible open KST < 24:00; skip disallowed entries",
                       "event_gain":"negative underlying fill-to-fill log return; excluded spells reheld over SEL intersection",
                       "crash_hold_return_pct":"ordinary underlying return between clipped from_price and to_price",
                       "cost_model":"one-way 0.001 or 0.002 subtracted from log gain per transition"}
    (OUT/"codex_verify_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__ == "__main__":
    checks = independent()
    assert all(rec["match"] for windows in checks.values() for rec in windows.values())
    result = audit(checks)
    judgments = {
        "C1":("fragile","확인 TV −12.02, 이웃도 확인 양수 0개. 사전 대피 효과는 있으나 상위 3회 재보유 시 SEL 부호 반전."),
        "C2":("keep","선택·확인에서 비용과 한국시간 복귀 제한을 각각 견뎠고 상위 3회 재보유 뒤에도 SEL TV가 양수. 2025년에는 두 제약 각각에서 음수."),
        "C3":("fragile","10/08 대피 이후 확인 내내 현금. 저비용·시간 제약에는 강하나 상위 3회 재보유 시 SEL 부호 반전."),
        "C4":("fragile","10/09 대피 이후 확인 내내 현금. 2025년 비용 후에도 강했으나 상위 3회 재보유 시 SEL 부호 반전."),
        "C5":("fragile","확인 전환은 2회, 비용 0.2% 후 확인 TV +0.34. 상위 3회 재보유 시 SEL 부호 반전."),
        "channel":("fragile","확인 비용 후 TV는 양수. 전체 위약 79.5백분위, 상위 3회 재보유 시 SEL 부호 반전, 한국시간 복귀 시 SEL TV +6.53.")}
    residual = result["candidates"]["C2"]["events"]["SEL_remove_best"]["3"]["rehold_recomputed_tv"]
    summarize(result,judgments,"C2",f"2026년 선택 구간에서 가장 유리했던 대피 3회를 되돌려도 TV {residual:+.2f}로그%p를 유지했고, 확인 구간에서 편도 비용 0.2% 후 TV가 +16.33로그%p였다. 2025년 비용 민감성은 장세 전환 위험이다.")
    write_json(result)
