"""(2026-11-22 독립 검토자 작성, 저장소로 옮김) soxl_lines.pine 의 차트 쪽 논리를 연장 40분봉(04:00 기준)에서 흉내 내 연구 정의(grid40 느린 선, warn_check 경고)와 대조한다.
봉 대응 가정(H1): lookahead_on 은 '시작 시각 <= 차트 봉 시작'인 마지막 요청 봉에 붙는다. 일봉 시작 = 09:30.
판본: cur = 지금 코드, fix = 완성 판정(차트 봉 시작 >= 요청 봉 끝이면 그 봉 값) 수정, fixreg = fix + 09:20 봉 저가를 정규장 1분봉만으로.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../soxl_escape_20261120/lines/code"))
import engine as E  # noqa: E402

Y0, Y1 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-11-20")


def daily_states(sym, touch):
    bd = E.bars(sym, "D", "reg")
    C, Lo = bd["Close"].to_numpy(float), bd["Low"].to_numpy(float)
    W = np.minimum(E.ema(bd["Close"], 10), E.ema(bd["Close"], 20)).to_numpy()
    R = np.maximum(E.ema(bd["Close"], 10), E.ema(bd["Close"], 20)).to_numpy()
    st, out = True, []
    for j in range(len(bd)):
        if st:
            if (touch and j > 0 and Lo[j] <= W[j - 1]) or ((not touch) and C[j] < W[j]):
                st = False
        elif C[j] > R[j]:
            st = True
        out.append(st)
    return bd, np.array(out), W, R


def run(sym, touch=False, ver="cur"):
    bd, hold_after, W, R = daily_states(sym, touch)
    d_open_t = (bd.index + pd.Timedelta(minutes=570)).tz_convert("UTC").tz_localize(None).to_numpy()   # 일봉 time = 09:30
    d_end_t = pd.DatetimeIndex(bd["end"]).tz_convert("UTC").tz_localize(None).to_numpy()
    d_open_px = bd["Open"].to_numpy(float)
    r40 = E.bars(sym, 40, "reg")
    e100 = E.ema(r40["Close"], 100).to_numpy()
    r_start = pd.DatetimeIndex(r40.index).tz_convert("UTC").tz_localize(None).to_numpy()
    r_end = pd.DatetimeIndex(r40["end"]).tz_convert("UTC").tz_localize(None).to_numpy()
    r_date = r40["date"].to_numpy()
    first = np.r_[True, r_date[1:] != r_date[:-1]]
    today = np.full(len(r40), np.nan)            # 그날 첫 봉에서 e[1](전날 마지막 봉 EMA)로 고정
    cur = np.nan
    for k in range(len(r40)):
        if first[k]:
            cur = e100[k - 1] if k > 0 else np.nan
        today[k] = cur
    x = E.bars(sym, 40, "ext")
    xs = pd.DatetimeIndex(x.index).tz_convert("UTC").tz_localize(None).to_numpy()
    xe = pd.DatetimeIndex(x["end"]).tz_convert("UTC").tz_localize(None).to_numpy()
    O, L = x["Open"].to_numpy(float), x["Low"].to_numpy(float)
    hm = x.index.hour * 60 + x.index.minute
    ehm = pd.DatetimeIndex(x["end"]).hour * 60 + pd.DatetimeIndex(x["end"]).minute
    xd = x["date"].to_numpy()
    # 정규장 1분봉만의 봉 저가(09:20 봉 프리장 오염 측정용)
    m1 = E.load(sym)
    reg1 = m1[m1["reg"]]
    key = reg1.index.normalize() + pd.to_timedelta(240 + ((reg1["mod"].to_numpy() - 240) // 40) * 40, unit="m")
    regLow = reg1.groupby(key)["Low"].min()
    regLow.index = pd.DatetimeIndex(regLow.index).tz_convert("UTC").tz_localize(None)
    regLow = regLow.reindex(xs).to_numpy()
    regOpen = reg1.groupby(key)["Open"].first()
    regOpen.index = pd.DatetimeIndex(regOpen.index).tz_convert("UTC").tz_localize(None)
    regOpen = regOpen.reindex(xs).to_numpy()

    di = np.searchsorted(d_open_t, xs, side="right") - 1      # 대응 일봉
    ri = np.searchsorted(r_start, xs, side="right") - 1        # 대응 정규 40분봉
    ev, warns = [], []
    stopped = pendIn = pendOut = False
    entry = np.nan
    prev_hold, prev_baseT, prev_reg, warned = None, None, False, False
    for i in range(len(x)):
        D = di[i]
        if D < 1:
            prev_hold = None
            continue
        if ver == "cur":
            j = D - 1
        else:
            j = D if xs[i] >= d_end_t[D] else D - 1
        stDone, wLine, baseT = bool(hold_after[j]), W[j], j
        k = ri[i]
        if k < 0:
            wl = np.nan
        elif ver == "cur":
            wl = today[k]
        else:
            wl = e100[k] if xs[i] >= r_end[k] else today[k]
        dayOpen = d_open_px[D]
        bs, be = hm[i], ehm[i]
        regBar = 570 <= bs < 960
        regTouch = regBar or (bs < 570 < be)
        newDay = i == 0 or xd[i] != xd[i - 1]
        execBar = regBar and (newDay or not prev_reg)
        low = L[i]
        if ver == "fixreg" and regTouch and not regBar:
            low = regLow[i]
        if prev_baseT is not None and baseT != prev_baseT:
            stopped = False
        touchNow = touch and stDone and (not stopped) and regTouch and not np.isnan(wLine) and low <= wLine
        if touchNow:
            stopped = True
        hold = stDone and not stopped
        if prev_hold is not None:
            sigOut = (not hold) and prev_hold
            sigIn = hold and (not prev_hold)
            if sigOut:
                pendIn = False
                if touchNow:
                    op = O[i] if not (ver == "fixreg" and not regBar) else regOpen[i]
                    px = min(op, wLine)
                    ev.append((xd[i], "out_touch", str(x.index[i])[11:16], px / entry - 1 if entry == entry else np.nan))
                    entry = np.nan
                else:
                    pendOut = True
                    ev.append((xd[i], "sigOut", str(x.index[i])[11:16], np.nan))
            if sigIn:
                pendOut = False
                pendIn = True
                ev.append((xd[i], "sigIn", str(x.index[i])[11:16], np.nan))
        if execBar:
            if pendOut:
                ev.append((xd[i], "out_open", str(x.index[i])[11:16], dayOpen / entry - 1 if entry == entry else np.nan))
                entry = np.nan
                pendOut = False
            if pendIn:
                entry = dayOpen
                ev.append((xd[i], "in_open", str(x.index[i])[11:16], np.nan))
                pendIn = False
        if newDay:
            warned = False
        if hold and not pendOut and regTouch and not np.isnan(wl) and low <= wl and not warned:
            warned = True
            warns.append((xd[i], str(x.index[i])[11:16], wl))
        prev_hold, prev_baseT, prev_reg = hold, baseT, regBar
    return ev, warns


def research_warn(sym):
    """warn_check.py 와 같은 정의: 느린 선(종가) 보유일에 정규장 1분봉 저가가 전날 마지막 정규 40분봉 EMA100 에 처음 닿은 날."""
    bd, hold_after, W, R = daily_states(sym, False)
    hold_today = pd.Series(np.r_[True, hold_after[:-1]], index=pd.to_datetime(bd["date"].values))
    b40 = E.bars(sym, 40, "reg")
    lvl = E.ema(b40["Close"], 100).groupby(b40["date"].values).last().shift(1)
    lvl.index = pd.to_datetime(lvl.index)
    d1 = E.load(sym)
    reg = d1[d1["reg"]]
    out = {}
    for d, g in reg.groupby("date"):
        if not (Y0 <= d <= Y1) or not bool(hold_today.get(d, False)) or np.isnan(lvl.get(d, np.nan)):
            continue
        h = g[g["Low"] <= lvl[d]]
        if len(h):
            out[d] = (str(h.index[0])[11:16], float(lvl[d]))
    return out


def research_touch(sym):
    bd, _, W, R = daily_states(sym, True)
    pos, corr, sw = E.simulate(bd, exit_touch=pd.Series(W, index=bd.index).shift(1),
                               entry_close=bd["Close"] > pd.Series(R, index=bd.index), exec_sess="reg")
    m1 = E.load(sym)
    out = []
    for p in sw:
        t = m1.index[p]
        out.append((pd.Timestamp(t.date()), "in" if pos[p] else "out", str(t)[11:16]))
    return out


if __name__ == "__main__":
    syms = sys.argv[1:] or ["MU", "SOXL"]
    for sym in syms:
        E.use(sym)
        rw = research_warn(sym)
        print(f"===== {sym}: 연구 경고일 {len(rw)} (2026-01-02~11-20)")
        for ver in ("cur", "fix", "fixreg"):
            ev, warns = run(sym, touch=False, ver=ver)
            pw = {d: (t, l) for d, t, l in warns if Y0 <= pd.Timestamp(d) <= Y1}
            both = sorted(set(pw) & set(rw))
            only_p = sorted(set(pw) - set(rw))
            only_r = sorted(set(rw) - set(pw))
            lvl_diff = [d for d in both if abs(pw[d][1] - rw[d][1]) > 1e-6]
            print(f"[종가 확인 · {ver}] 지표 경고일 {len(pw)}, 같은 날 {len(both)}, 지표만 {len(only_p)}, 연구만 {len(only_r)}, 같은 날 중 선 값이 다른 날 {len(lvl_diff)}")
            for d in only_p[:8]:
                print("   지표만", str(d)[:10], pw[d][0], round(pw[d][1], 2))
            for d in only_r[:8]:
                print("   연구만", str(d)[:10], rw[d][0], round(rw[d][1], 2))
            sig = [e for e in ev if Y0 <= pd.Timestamp(e[0]) <= Y1]
            so = [e for e in sig if e[1] in ("sigOut", "sigIn")]
            print("   신호가 뜬 봉 시각 분포:", pd.Series([e[2] for e in so]).value_counts().to_dict())
        # 닿으면 방식
        rt = [e for e in research_touch(sym) if Y0 <= e[0] <= Y1]
        print(f"[닿으면 · 연구] 전환 {len(rt)}건: 대피 {sum(1 for e in rt if e[1]=='out')}, 재매수 {sum(1 for e in rt if e[1]=='in')}")
        rset = {(str(e[0])[:10], e[1]) for e in rt}
        for ver in ("cur", "fix", "fixreg"):
            ev, _ = run(sym, touch=True, ver=ver)
            ex = [(str(pd.Timestamp(e[0]))[:10], "out" if e[1].startswith("out") else "in", e[2]) for e in ev
                  if e[1] in ("out_touch", "out_open", "in_open") and Y0 <= pd.Timestamp(e[0]) <= Y1]
            pset = {(a, b) for a, b, _ in ex}
            print(f"[닿으면 · {ver}] 체결 {len(ex)}건, 연구와 같은 (날짜,방향) {len(pset & rset)}, 지표만 {len(pset - rset)}, 연구만 {len(rset - pset)}")
            for a in sorted(pset - rset)[:6]:
                print("   지표만", a, [e[2] for e in ex if (e[0], e[1]) == a])
            for a in sorted(rset - pset)[:6]:
                print("   연구만", a)
