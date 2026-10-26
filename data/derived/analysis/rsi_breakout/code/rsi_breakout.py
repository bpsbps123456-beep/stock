"""사용자 아이디어(2026-10-10): TIME·KoAct 미국 보유 종목에서 최근 흐름이 좋은 종목을 매일 고른다.

사용자 규칙
  매수      160분봉 RSI(14)가 70을 아래에서 위로 넘은 봉이 나온 날의 다음 미국 거래일 시가.
  매도      일봉 Stoch RSI K(RSI 14·스토캐스틱 14·K 평활 3, 사용자 Pine과 같음)가 95를 넘은 날의 다음 시가.
            사는 날부터 종가마다 본다. 이미 95 위에서 시작하면 첫 종가에 판다(변형: 95를 아래에서 위로 넘을 때만).
  빈자리    7자리. 모자라면 5일 수익률이 높은 순(사용자 '5일 가격 흐름').
비교      5일 수익 상위 7(매일), 5일 안정(5일 수익 ÷ 5일 변동) 상위 7(매일), 데스크 7종목(#774, desk.strategies),
          QQQ, 그날 대상 전체 같은 비중(매일).
시점      대상은 그 미국 거래일보다 앞선 날짜의 최신 PDF(두 펀드)에 있는 미국 상장 보유다(데스크 §3과 같음).
          신호는 봉이 끝난 뒤에만 쓰고, 체결은 모두 다음 시가(서머타임 한국 22:30)다. 수수료 0(§1), 호가 차이는 편도 0.1%로 따로 본다.
160분봉   1시간봉으로 하루 3봉(12:30·14:30·16:00 ET 종가)을 만들어 근사한다. 최근 60일은 5분봉으로 만든
          진짜 160분봉(12:10·14:50·16:00, 트레이딩뷰 정규장 기준)과 신호를 대조한다.
기간      2026-01-02~05-29(앞) / 06-01~10-09(뒤). 사용자 규칙은 고른 값이 없어 두 구간 모두 처음 보는 자료다.
사용      py rsi_breakout.py fetch → run → results/rsi_breakout.txt"""
import io, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, ROOT)
CACHE = os.path.join(HERE, "../cache")
RES = os.path.join(HERE, "../results")
FUNDS = {"TIME": "456600", "KoAct": "2ETFQ1"}
START, END = "2026-01-02", "2026-10-09"
HALVES = (("앞 1/2~5/29", "2026-01-02", "2026-05-29"), ("뒤 6/1~10/9", "2026-06-01", "2026-10-09"), ("전체", START, END))
SLOTS, COST = 7, 0.001
RSI_LEN, STOCH_LEN, SMOOTH = 14, 14, 3


# ---------------------------------------------------------------- 자료
def load_universe():
    from gbrc.etf.trades import load_fund_snapshots
    from gbrc.etf.market import yahoo_ticker
    from desk.strategies import us_weights
    held = {}
    for name, code in FUNDS.items():
        snaps = load_fund_snapshots(code, "9999-12-31")
        held[name] = {d: us_weights(s) for d, s in sorted(snaps.items())}
    keys = sorted({k for f in held.values() for w in f.values() for k in w})
    return held, {k: yahoo_ticker(k) for k in keys}


def fetch():
    import yfinance as yf
    os.makedirs(CACHE, exist_ok=True)
    held, tick = load_universe()
    syms = sorted(set(tick.values()) | {"QQQ"})
    json.dump({"tickers": tick}, open(os.path.join(CACHE, "universe.json"), "w", encoding="utf-8"), ensure_ascii=False)
    jobs = {"1d": dict(start="2025-06-01", end="2026-10-10", interval="1d"),
            "1h": dict(start="2025-09-02", end="2026-10-10", interval="1h"),
            "5m": dict(period="60d", interval="5m")}
    for name, kw in jobs.items():
        rows = []
        for i in range(0, len(syms), 40):
            d = yf.download(syms[i:i + 40], prepost=False, auto_adjust=False, progress=False, group_by="ticker",
                            threads=True, **kw)
            for s in syms[i:i + 40]:
                try:
                    x = d[s] if isinstance(d.columns, pd.MultiIndex) else d
                    x = x[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])
                except KeyError:
                    continue
                if len(x):
                    x = x.copy()
                    if name != "1d":
                        x.index = x.index.tz_convert("America/New_York")
                    x["sym"] = s
                    rows.append(x)
        df = pd.concat(rows)
        df.index.name = "ts"
        df.reset_index().to_parquet(os.path.join(CACHE, f"{name}.parquet"))
        print(name, len(df), "행", df.sym.nunique(), "종목")


def frames():
    out = {}
    for name in ("1d", "1h", "5m"):
        df = pd.read_parquet(os.path.join(CACHE, f"{name}.parquet"))
        out[name] = {s: g.set_index("ts").drop(columns="sym") for s, g in df.groupby("sym")}
    return out


# ---------------------------------------------------------------- 지표(트레이딩뷰 ta.rsi·ta.stoch와 같은 계산)
def rma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def rsi(close, n=RSI_LEN):
    close = np.asarray(close, float)
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    d = np.diff(close)
    ag, al = rma(np.clip(d, 0, None), n), rma(np.clip(-d, 0, None), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(al == 0, 100.0, np.where(ag == 0, 0.0, 100 - 100 / (1 + ag / al)))
    out[1:] = np.where(np.isnan(ag), np.nan, r)
    return out


def stoch_rsi_k(close):
    close = np.asarray(close, float)
    out = np.full(len(close), np.nan)
    ok = np.flatnonzero(np.isfinite(close))
    if not len(ok):
        return out
    seg = pd.Series(close[ok[0]:]).ffill().to_numpy()             # 상장 전 빈칸은 빼고, 중간 빈칸은 앞 값으로
    s = pd.Series(rsi(seg))
    lo, hi = s.rolling(STOCH_LEN).min(), s.rolling(STOCH_LEN).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        st = 100 * (s - lo) / (hi - lo)
    out[ok[0]:] = st.rolling(SMOOTH).mean().to_numpy()
    return out


def bars160(x, exact):
    """하루 3봉. exact=True면 5분봉에서 09:30부터 160분씩(12:10·14:50·16:00), 아니면 1시간봉으로 12:30·14:30·16:00."""
    m = x.index.hour * 60 + x.index.minute
    x = x[(m >= 570) & (m < 960)]
    m = x.index.hour * 60 + x.index.minute
    g = (m - 570) // 160 if exact else np.where(m < 750, 0, np.where(m < 870, 1, 2))
    ends = (730, 890, 960) if exact else (750, 870, 960)
    day = x.index.normalize().tz_localize(None)
    b = x.groupby([day, g]).agg(c=("Close", "last"))
    b["end"] = [ends[k] for k in b.index.get_level_values(1)]
    b.index = b.index.get_level_values(0)
    return b


def cross_days(b, th):
    """RSI가 th를 아래에서 위로 넘은 봉이 있는 날 → {날짜: 그 봉이 끝난 시각(분)}."""
    r = rsi(b.c.to_numpy())
    hit = (np.roll(r, 1) <= th) & (r > th)
    hit[0] = False
    out = {}
    for d, e in zip(b.index[hit], b.end.to_numpy()[hit]):
        out.setdefault(d.strftime("%Y-%m-%d"), int(e))
    return out


# ---------------------------------------------------------------- 표 만들기
class Panel:
    def __init__(self, th=70, exit_level=95, exit_cross=False, exact=False):
        self.held, self.tick = load_universe()
        f = frames()
        q = f["1d"]["QQQ"]
        self.days = [d.strftime("%Y-%m-%d") for d in q.index]
        self.syms = sorted(set(self.tick.values()) & set(f["1d"]))
        idx = pd.Index(self.days)

        def on_days(field):
            return pd.DataFrame({s: f["1d"][s][field].set_axis(f["1d"][s].index.strftime("%Y-%m-%d")).reindex(idx)
                                 for s in self.syms + ["QQQ"]})
        self.opn, self.cls = on_days("Open"), on_days("Close")
        self.k = pd.DataFrame({s: stoch_rsi_k(self.cls[s].to_numpy()) for s in self.syms}, index=idx)
        self.k[self.cls[self.syms].isna()] = np.nan
        self.f = f
        src = f["5m"] if exact else f["1h"]
        self.sig = {s: cross_days(bars160(src[s], exact), th) for s in self.syms if s in src}
        self.exit_level, self.exit_cross = exit_level, exit_cross
        self.pdfs = {n: sorted(h) for n, h in self.held.items()}
        self._elig = {}

    def eligible(self, day):
        if day not in self._elig:
            from desk.strategies import latest_before
            out = set()
            for n, h in self.held.items():
                p = latest_before(self.pdfs[n], day)
                if p:
                    out |= {self.tick[k] for k in h[p] if self.tick.get(k) in self.opn.columns}
            self._elig[day] = out
        return self._elig[day]

    def exit_hit(self, s, j):
        k = self.k[s].to_numpy()
        if not np.isfinite(k[j]):
            return False
        if self.exit_cross:
            return j > 0 and np.isfinite(k[j - 1]) and k[j - 1] <= self.exit_level < k[j]
        return k[j] > self.exit_level


# ---------------------------------------------------------------- 신호별 거래
def trades(P):
    rows = []
    pos = {d: i for i, d in enumerate(P.days)}
    for s in P.syms:
        sig = sorted(P.sig.get(s, {}))
        o, c = P.opn[s].to_numpy(), P.cls[s].to_numpy()
        busy_until = -1
        for d in sig:
            i = pos.get(d)
            if i is None or i + 1 >= len(P.days) or i < busy_until:
                continue
            e = i + 1
            if P.days[e] < START or s not in P.eligible(P.days[e]) or not np.isfinite(o[e]):
                continue
            j = e
            while j < len(P.days) and not P.exit_hit(s, j):
                j += 1
            done = j + 1 < len(P.days) and np.isfinite(o[j + 1])
            px_out = o[j + 1] if done else c[np.flatnonzero(np.isfinite(c))[-1]]
            out_i = j + 1 if done else len(P.days) - 1
            q = P.opn["QQQ"].to_numpy()
            rows.append(dict(sym=s, signal=d, entry=P.days[e], exit=P.days[out_i] if done else None,
                             days=out_i - e, ret=px_out / o[e] - 1,
                             qqq=(q[out_i] if done else P.cls["QQQ"].to_numpy()[out_i]) / q[e] - 1,
                             k_in=P.k[s].iloc[e - 1], ret5=c[i] / c[i - 5] - 1 if i >= 5 else np.nan,
                             mae=np.nanmin(P.cls[s].to_numpy()[e:out_i + 1]) / o[e] - 1, done=done))
            busy_until = j
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 7자리 계좌
def simulate(P, pick):
    """pick(k, held) → 오늘 시가에 들고 갈 종목 목록(최대 7). 각 1/7, 남는 자리는 현금. 시가→다음 시가 수익."""
    k0 = P.days.index(next(d for d in P.days if d >= START))
    held, val, rows = [], 1.0, []
    w_prev = {}
    o = P.opn
    for k in range(k0, len(P.days) - 1):
        held = pick(k, held)
        w = {s: 1 / max(SLOTS, len(held)) for s in held}               # 7자리면 각 1/7, 전체 같은 비중이면 1/종목 수
        turn = sum(abs(w.get(s, 0) - w_prev.get(s, 0)) for s in set(w) | set(w_prev))
        r = sum(wt * (o[s].iloc[k + 1] / o[s].iloc[k] - 1) for s, wt in w.items()
                if np.isfinite(o[s].iloc[k + 1]) and np.isfinite(o[s].iloc[k]))
        val *= 1 + r
        rows.append(dict(day=P.days[k], ret=r, turn=turn, n=len(held), val=val))
        # 다음 날 시작 비중: 수익만큼 흘러간 비중
        tot = 1 + r
        w_prev = {s: wt * (o[s].iloc[k + 1] / o[s].iloc[k]) / tot for s, wt in w.items()
                  if np.isfinite(o[s].iloc[k + 1]) and np.isfinite(o[s].iloc[k])}
    return pd.DataFrame(rows)


def ret5(P, k):
    c = P.cls
    return (c.iloc[k - 1] / c.iloc[k - 6] - 1) if k >= 6 else pd.Series(dtype=float)


def rule_picker(P):
    def pick(k, held):
        keep = [s for s in held if not P.exit_hit(s, k - 1)]           # 어제 종가에 95를 넘었으면 오늘 시가에 판다
        elig = P.eligible(P.days[k])
        new = [s for s in P.syms if s not in keep and s in elig and P.days[k - 1] in P.sig.get(s, {})
               and np.isfinite(P.opn[s].iloc[k])]
        r5 = ret5(P, k)
        new.sort(key=lambda s: -(r5.get(s, -9) if np.isfinite(r5.get(s, np.nan)) else -9))
        return keep + new[:max(0, SLOTS - len(keep))]
    return pick


def top5_picker(P, stable=False):
    def pick(k, held):
        elig = [s for s in P.eligible(P.days[k]) if np.isfinite(P.opn[s].iloc[k])]
        c = P.cls.iloc[k - 6:k]
        r5 = c.iloc[-1] / c.iloc[0] - 1
        score = r5 / c.pct_change().iloc[1:].std() if stable else r5
        if stable:
            score = score.where(r5 > 0)
        sc = score.reindex(elig).dropna().sort_values(ascending=False)
        return list(sc.index[:SLOTS])
    return pick


def ew_picker(P):
    def pick(k, held):
        return [s for s in P.eligible(P.days[k]) if np.isfinite(P.opn[s].iloc[k])]
    return pick


def curve_stats(df, a, b, cost=0.0):
    x = df[(df.day >= a) & (df.day <= b)]
    if x.empty:
        return None
    r = x.ret - cost * x.turn
    v = (1 + r).cumprod()
    dd = (v / np.maximum.accumulate(v) - 1).min()
    return dict(total=v.iloc[-1] - 1, mdd=dd, turn=x.turn.mean(), n=x.n.mean())


def desk_curve(P):
    from desk.strategies import load_book, run_model
    from desk.validate import codex_caps      # 기록한 비교는 #774(반도체 60%) 기준
    with codex_caps():
        path = run_model(load_book(), start=START)
    v = pd.Series(path.value, index=path.days)
    r = (v.shift(-1) / v - 1).dropna()
    return pd.DataFrame(dict(day=r.index, ret=r.to_numpy(), turn=0.0, n=7))


def run():
    os.makedirs(RES, exist_ok=True)
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    P = Panel()
    tr = trades(P)
    tr.to_csv(os.path.join(RES, "trades.csv"), index=False)
    p("[신호별 거래 — 160분봉 RSI 70 돌파 다음 시가 매수, 일봉 Stoch RSI K 95 넘은 다음 시가 매도]")
    for name, a, b in HALVES:
        x = tr[(tr.entry >= a) & (tr.entry <= b)]
        if x.empty:
            continue
        p(f"  {name}: {len(x)}건 · 평균 {x.ret.mean():+.2%} · 중앙 {x.ret.median():+.2%} · 승률 {(x.ret > 0).mean():.0%} · "
          f"보유 중앙 {x.days.median():.0f}일 · 같은 때 QQQ {x.qqq.mean():+.2%} · 차이 {(x.ret - x.qqq).mean():+.2%} · "
          f"하위10% {x.ret.quantile(.1):+.1%} · 최악 {x.ret.min():+.1%} · 산 날 전날 K 중앙 {x.k_in.median():.0f} · "
          f"그날 판 비율(보유 1일) {(x.days <= 1).mean():.0%}")
    curves = {"사용자 규칙(RSI70→K95, 7자리)": simulate(P, rule_picker(P)),
              "5일 수익 상위 7(매일)": simulate(P, top5_picker(P)),
              "5일 안정 상위 7(매일)": simulate(P, top5_picker(P, stable=True)),
              "대상 전체 같은 비중(매일)": simulate(P, ew_picker(P))}
    q = P.opn["QQQ"]
    qq = pd.DataFrame(dict(day=P.days[:-1], ret=(q.shift(-1) / q - 1).to_numpy()[:-1], turn=0.0, n=1))
    curves["QQQ"] = qq[qq.day >= START]
    try:
        curves["데스크 7종목(#774)"] = desk_curve(P)
    except Exception as exc:                       # 데스크 가격 캐시가 없으면 비교만 뺀다
        p(f"  (데스크 7종목 비교 실패: {type(exc).__name__}: {exc})")
    p("\n[7자리 계좌 — 시가 체결, 수수료 0 / 괄호는 호가 차이 편도 0.1%]")
    for name, df in curves.items():
        df.to_csv(os.path.join(RES, f"curve_{name.split('(')[0].strip().replace(' ', '_')}.csv"), index=False)
        cells = []
        for hn, a, b in HALVES:
            s0, s1 = curve_stats(df, a, b), curve_stats(df, a, b, COST)
            cells.append("—" if s0 is None else f"{hn} {s0['total']:+.1%}({s1['total']:+.1%}) 낙폭 {s0['mdd']:.1%}")
        st = curve_stats(df, START, END)
        p(f"  {name:<22} " + " | ".join(cells) + f" · 하루 회전 {st['turn']:.0%} · 평균 보유 {st['n']:.1f}")
    check(P, tr, p)
    sensitivity(p)
    txt = o.getvalue()
    open(os.path.join(RES, "rsi_breakout.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


def check(P, tr, p):
    """1시간봉 근사와 진짜 160분봉(5분봉)의 신호 날짜 대조, 그리고 PLTR."""
    E = Panel(exact=True)
    first = min(g.index.min() for g in P.f["5m"].values()).strftime("%Y-%m-%d")
    win = [d for d in P.days if d >= first][10:]                     # RSI가 자리 잡는 2주는 뺀다
    a = {(s, d) for s in P.syms for d in P.sig.get(s, {}) if d in win}
    x = {(s, d) for s in E.syms for d in E.sig.get(s, {}) if d in win}
    pos = {d: i for i, d in enumerate(P.days)}
    near = sum(1 for s, d in x if any((s, P.days[pos[d] + j]) in a for j in (-1, 0, 1) if 0 <= pos[d] + j < len(P.days)))
    p(f"\n[160분봉 근사 대조 {win[0]}~{win[-1]}] 진짜 160분봉 신호 {len(x)}건 중 같은 날 근사 신호 {len(a & x) / max(len(x), 1):.0%}, "
      f"±1일 {near / max(len(x), 1):.0%} · 근사 신호 {len(a)}건 중 같은 날 진짜 신호 {len(a & x) / max(len(a), 1):.0%}")
    s = "PLTR"
    if s in P.syms:
        c, k = P.cls[s], P.k[s]
        hi = P.f["1d"][s].High
        p(f"[PLTR] 근사 신호일 9/1 뒤: {', '.join(d for d in sorted(P.sig.get(s, {})) if d >= '2026-09-01') or '없음'} · "
          f"진짜 160분봉 신호일: {', '.join(d for d in sorted(E.sig.get(s, {})) if d >= '2026-09-01') or '없음'}")
        for d in P.days[-8:]:
            prior = hi[hi.index.strftime('%Y-%m-%d') < d].max()
            today = hi[hi.index.strftime('%Y-%m-%d') == d]
            p(f"  {d} 종가 {c[d]:.2f} · 일봉 K {k[d]:.0f} · 그날 고가 {float(today.iloc[0]) if len(today) else float('nan'):.2f}"
              f"(2025-06 뒤 이전 최고 {prior:.2f})")
        for _, r in tr[tr.sym == s].iterrows():
            p(f"  거래: 신호 {r.signal} → 매수 {r.entry} → 매도 {r.exit or '보유 중'} · {r.ret:+.1%} · {r.days}일")


def sensitivity(p):
    p("\n[값을 바꿔 보면 — 신호별 거래 평균(건수)과 7자리 계좌, 앞 / 뒤]")
    for th in (65, 70, 75):
        for lvl, cross in ((95, False), (95, True), (80, False)):
            P = Panel(th, lvl, cross)
            tr = trades(P)
            cur = simulate(P, rule_picker(P))
            cells = []
            for hn, a, b in HALVES[:2]:
                x = tr[(tr.entry >= a) & (tr.entry <= b)]
                st = curve_stats(cur, a, b)
                cells.append(f"{hn[:1]} 거래 {x.ret.mean():+.2%}({len(x)}) 계좌 {st['total']:+.1%}")
            mark = "★" if (th, lvl, cross) == (70, 95, False) else " "
            p(f"  {mark} RSI {th} / K {lvl}{'(위로 넘을 때)' if cross else ''}: " + " · ".join(cells))
    p("  ★ = 사용자 규칙. 시험 범위: RSI 문턱 3 × 매도 3 = 9개.")


if __name__ == "__main__":
    {"fetch": fetch, "run": run}[sys.argv[1]]()
