"""갈래 H: 이벤트 달력·펀드매니저 신호·7월 쇼크 사건 연구(2026-10-10).

1부 미리 아는 사건 앞에서 줄이기(실적 8종목·FOMC). 일정은 몇 주 전 공지 → 그날 시가에 바로 쓴다(사전 정보).
2부 펀드매니저 신호(TIME·KoAct PDF 현금·반도체 비중). PDF 날짜 D = 그날 이전 바스켓 → at_open(index=PDF 날짜).
3부 사건 연구: 대표 시장 신호가 하락 고점·저점 대비 언제 켜졌는지, 평온기 헛경보 수.

격자(사전 고정, 2026 뒤 구간·보유 기간을 보고 바꾸지 않는다)
  1부 사건 묶음 6 × 노출 2 × 창 2 = 24
  2부 신호 8 × 펀드 3 × 노출 2 × 최소 유지 2 = 96
  합계 120
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import harness as H

ROOT = H.HERE.parents[4]
sys.path.insert(0, str(ROOT))

# ----------------------------------------------------------------------------- 격자(사전 선언)
EARN_SYMS = ["NVDA", "MSFT", "META", "GOOGL", "AMZN", "AVGO", "MU", "TSM"]
EVENT_GROUPS = {
    "NVDA": ["NVDA"],
    "하이퍼4": ["MSFT", "META", "GOOGL", "AMZN"],
    "반도체4": ["NVDA", "AVGO", "MU", "TSM"],
    "실적8": EARN_SYMS,
    "FOMC": ["FOMC"],
    "실적8+FOMC": EARN_SYMS + ["FOMC"],
}
EVENT_EXPO = (0.5, 0.7)
EVENT_WINDOW = ("gap", "gap+day")       # gap: 반응 첫 시가 직전 거래일 하루 / gap+day: 반응일 하루 더

# 연준 공표 일정(성명 발표일, 14:00 ET). 2024-11-07은 선거 주라 목요일.
FOMC = ["2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]

# 2부: (신호종류, 문턱). cashL = 현금 비중 > x%, cashD = 1주(5세션, 주간 PDF는 1스냅샷) 새 현금 +Δ%p 이상,
# semiD = 1주 새 반도체 합계 비중 −Δ%p 이하
FUND_SIGNALS = [("cashL", 2.0), ("cashL", 3.0), ("cashL", 5.0), ("cashD", 1.5), ("cashD", 3.0),
                ("semiD", 2.0), ("semiD", 4.0), ("semiD", 6.0)]
FUNDS = ("TIME", "KoAct", "둘다")
FUND_EXPO = (0.5, 0.0)
FUND_HOLD = (0, 10)                      # 신호가 꺼져도 마지막 켜진 날부터 N거래일 더 줄인 채 유지

GRID_DESC = (f"1부 사건 묶음 {len(EVENT_GROUPS)}(NVDA·하이퍼4·반도체4·실적8·FOMC·실적8+FOMC) × 노출 {EVENT_EXPO} × "
             f"창 {EVENT_WINDOW} = {len(EVENT_GROUPS) * len(EVENT_EXPO) * len(EVENT_WINDOW)}; "
             f"2부 신호 {len(FUND_SIGNALS)}(현금>2·3·5%, 1주 현금 +1.5·+3%p, 1주 반도체 −2·−4·−6%p) × 펀드 {FUNDS} × "
             f"노출 {FUND_EXPO} × 최소 유지 {FUND_HOLD}거래일 = "
             f"{len(FUND_SIGNALS) * len(FUNDS) * len(FUND_EXPO) * len(FUND_HOLD)}. 합계 120. "
             "보유 기간(hold)은 KoAct PDF가 없어 KoAct·둘다 변형을 TIME 단독으로 대체해 잰다.")

EV_CACHE = H.CACHE / "H_events.json"

# 3부 사건(고점, 저점): 기본 경로 종가 평가 기준
EVENTS = {"2026": [("26-02 1차", "2026-02-03", "2026-02-05"), ("26-02 2차", "2026-02-25", "2026-03-06"),
                   ("26-03", "2026-03-19", "2026-03-30"), ("26 AI쇼크", "2026-06-22", "2026-07-29")],
          "hold": [("25 딥시크", "2025-01-23", "2025-01-27"), ("25 관세", "2025-02-18", "2025-04-04"),
                   ("25 가을", "2025-10-13", "2025-11-21"), ("24 여름", "2024-06-18", "2024-08-05"),
                   ("23 가을", "2023-08-01", "2023-10-26")]}


# ----------------------------------------------------------------------------- 1부 자료
def fetch_events() -> dict:
    if EV_CACHE.exists():
        return json.loads(EV_CACHE.read_text(encoding="utf-8"))
    import yfinance as yf
    out, fails = {}, {}
    for s in EARN_SYMS:
        try:
            d = yf.Ticker(s).get_earnings_dates(limit=40)
            rows = []
            for ts in d.index:
                ts = pd.Timestamp(ts).tz_convert("America/New_York")
                if not ("2023-01-01" <= ts.strftime("%Y-%m-%d") <= "2026-10-09"):
                    continue
                hr = ts.hour + ts.minute / 60
                if hr == 0:                                   # 시각 미상: TSM 장전, 나머지 장후가 관행
                    when = "BMO" if s == "TSM" else "AMC"
                else:
                    when = "BMO" if hr < 12 else "AMC"
                rows.append({"date": ts.strftime("%Y-%m-%d"), "time": ts.strftime("%H:%M"), "when": when})
            out[s] = sorted(rows, key=lambda r: r["date"])
        except Exception as e:                               # noqa: BLE001
            fails[s] = repr(e)
    payload = {"fetched": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "source": "yfinance get_earnings_dates(limit=40)",
               "earnings": out, "fails": fails, "fomc": FOMC}
    EV_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def reaction_days(ev: dict, syms: list, days: list) -> list:
    """사건마다 반응 첫 시가 거래일 R의 index. 장전(BMO) D → R = D 이후 첫 거래일(D 포함), 장후·FOMC D → R = D 다음 거래일."""
    arr = np.asarray(days)
    out = []
    for s in syms:
        items = [{"date": d, "when": "AMC"} for d in ev["fomc"]] if s == "FOMC" else ev["earnings"].get(s, [])
        for it in items:
            side = "left" if it["when"] == "BMO" else "right"
            i = int(np.searchsorted(arr, it["date"], side=side))
            if 1 <= i < len(arr):
                out.append(i)
    return sorted(set(out))


def event_exposure(b: H.Base, ev: dict, syms: list, lvl: float, win: str) -> pd.Series:
    e = np.ones(len(b.days))
    for i in reaction_days(ev, syms, b.days):
        e[i - 1] = min(e[i - 1], lvl)                       # R−1 시가 → R 시가(반응 갭)
        if win == "gap+day":
            e[i] = min(e[i], lvl)                           # R 시가 → R+1 시가(반응일 장중)
    return pd.Series(e, index=b.days)


# ----------------------------------------------------------------------------- 2부 자료
def _semis_classifier():
    import watchlist as W
    from desk import config, strategies as S
    from gbrc.etf.market import yahoo_ticker
    semi_layers = {"칩 설계", "파운드리·후공정", "메모리·스토리지", "장비·소재·테스트"}
    layer_of = {sym: L for L, subs in W.AI_UNIVERSE.items() for syms in subs.values() for sym in syms}
    cache: dict = {}

    def is_semi(code: str) -> bool:
        if code in cache:
            return cache[code]
        key = S.research_key(code)
        t = yahoo_ticker(key)
        sec = S.sector_of(key, t or key)
        ok = sec in config.SEMIS_SECTORS
        if not ok and t:
            ok = layer_of.get(W.AI_UNIVERSE_ALIASES.get(t, t)) in semi_layers
        cache[code] = ok
        return ok
    return is_semi


def load_fund_series() -> dict:
    """{'TIME26','KoAct26','TIMEhold'}: DataFrame(index=PDF 날짜, cash %, semi %)."""
    is_semi = _semis_classifier()

    def row_of(d, cash_key=None):
        hs = d["holdings"]
        cash = sum(float(h["weight"]) for h in hs if h.get("asset_type") == "cash" or (not h.get("code") and h.get("name") in ("현금", "Cash")))
        semi = sum(float(h["weight"]) for h in hs if h.get("code") and h.get("asset_type", "security") != "cash" and is_semi(h["code"]))
        if cash_key is not None:
            cash = float(d["metrics"]["cash_weight_pct"])
        return cash, semi

    out = {}
    norm = ROOT / "data/normalized/etf_holdings"
    for fid, nm in (("456600", "TIME26"), ("2ETFQ1", "KoAct26")):
        rec = {}
        for p in sorted(norm.glob(f"*_{fid}.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            rec[d["actual_date"]] = row_of(d, "metrics")
        out[nm] = pd.DataFrame.from_dict(rec, orient="index", columns=["cash", "semi"]).sort_index()
    rec = {}
    for p in sorted((ROOT / "data/derived/analysis/strategy_lab/holdout/time_weekly").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rec[d["actual_date"]] = row_of(d)
    out["TIMEhold"] = pd.DataFrame.from_dict(rec, orient="index", columns=["cash", "semi"]).sort_index()
    return out


def fund_signal(df: pd.DataFrame, kind: str, thr: float, lag: int) -> pd.Series:
    if kind == "cashL":
        return df["cash"] > thr
    if kind == "cashD":
        return (df["cash"] - df["cash"].shift(lag)) >= thr
    if kind == "semiD":
        return (df["semi"] - df["semi"].shift(lag)) <= -thr
    raise ValueError(kind)


def hold_min(on: pd.Series, n: int) -> pd.Series:
    """거래일 단위 on(0/1)을 마지막 켜진 날부터 n거래일 더 유지."""
    if n <= 0:
        return on
    return on.rolling(n + 1, min_periods=1).max()


def fund_exposure(b: H.Base, sig: pd.Series, lvl: float, n: int) -> pd.Series:
    on = H.at_open(sig.astype(float), b, fill=0.0)
    on = hold_min(on, n)
    return pd.Series(np.where(on > 0.5, lvl, 1.0), index=b.days)


# ----------------------------------------------------------------------------- 행동 보고
def behavior(b: H.Base, e: pd.Series, lo: str, hi: str) -> dict:
    """구간 [lo, hi]에서 줄인 구간들과 아낀 손실(기본이 빠진 날의 덜 잃은 몫)·놓친 반등(기본이 오른 날 덜 번 몫)."""
    r = b.ret[(b.ret.index >= lo) & (b.ret.index <= hi)]
    ee = e.reindex(r.index)
    ov = float((1 + ee * r).prod() - 1)
    bs = float((1 + r).prod() - 1)
    segs, cur = [], None
    for d, v in ee.items():
        if v < 0.999:
            if cur is None or cur["e"] != round(v, 3):
                if cur:
                    segs.append(cur)
                cur = {"from": d, "to": d, "e": round(float(v), 3), "n": 0, "_p": 1.0, "_q": 1.0}
            cur["to"] = d
            cur["n"] += 1
            cur["_p"] *= 1 + r[d]
            cur["_q"] *= 1 + v * r[d]
        elif cur:
            segs.append(cur)
            cur = None
    if cur:
        segs.append(cur)
    for s in segs:                                       # 효과 = 줄인 구간 안 (오버레이 − 기본) 복리 수익: +면 아낀 손실, −면 놓친 반등
        p, q = s.pop("_p"), s.pop("_q")
        s["base_ret"] = round(float(p - 1), 4)
        s["effect"] = round(float(q - p), 4)
    saved = sum(s["effect"] for s in segs if s["effect"] > 0)
    missed = sum(s["effect"] for s in segs if s["effect"] < 0)
    return {"window": [lo, hi], "base": round(bs, 4), "overlay": round(ov, 4), "saved": round(float(saved), 4),
            "missed": round(float(missed), 4), "exits": int(((ee.shift(1).fillna(1.0) >= 0.999) & (ee < 0.999)).sum()),
            "segments": segs}


# ----------------------------------------------------------------------------- 3부 시장 신호
def market_signals(b: H.Base) -> dict:
    """b.days 위의 종가 신호(t일 종가에 알 수 있는 값, bool)."""
    days = b.days
    sox = H.px("SOXX", "Close")
    s1 = (sox < sox.rolling(50).mean()).reindex(days).ffill().fillna(False).astype(bool)
    vr = (H.px("^VIX") / H.px("^VIX3M")).reindex(days).ffill()
    s2 = (vr > 1.0)
    f = H.fred_known()["BAMLH0A0HYM2"].dropna()
    hy_d = (f - f.shift(20))
    # fred_known 날짜(알게 된 날)를 거래일에 맞춤: 그날까지 알려진 최신 값
    hy = hy_d.reindex(sorted(set(hy_d.index) | set(days))).ffill().reindex(days)
    s3 = (hy >= 0.5)
    jpy = H.px("JPY=X", "Close").reindex(days).ffill()
    s4 = (jpy / jpy.shift(10) - 1) <= -0.03
    br = H.breadth()
    sma = br.rolling(50, min_periods=50).mean()
    above = (br > sma).astype(float).where(sma.notna() & br.notna())
    cnt = above.count(axis=1).astype(float)
    frac = above.sum(axis=1) / cnt.where(cnt > 0)
    s5 = (frac.reindex(days).ffill() < 0.40)
    cr = b.close_s.pct_change()
    s6 = (cr.rolling(20).std() * np.sqrt(252)) > 0.60
    s7 = (b.close_s / b.close_s.cummax() - 1) <= -0.10
    out = {"SOXX<SMA50": s1, "VIX/VIX3M>1": s2, "HY OAS 20일 +0.5%p": s3, "엔 10일 강세 3%": s4,
           "AI폭 50일선 위<40%": s5, "그림자 20일 변동성>60%": s6, "그림자 낙폭 −10%": s7}
    meta = {"HY OAS 20일 +0.5%p": f.index.min()}
    return {k: v.reindex(days).fillna(False).astype(bool) for k, v in out.items()}, meta


def event_table(b: H.Base, sigs: dict, events: list, pre: int = 10) -> list:
    days = b.days
    pos = {d: i for i, d in enumerate(days)}
    c = b.close_s
    op = pd.Series(b.open, index=days)
    rows = []
    for nm, P, T in events:
        iP, iT = pos[P], pos[T]
        peak = c.iloc[iP]
        trough = float(c.iloc[iP:iT + 1].min() / peak - 1)
        row = {"사건": nm, "고점": P, "저점": T, "하락": round(trough, 3)}
        for sn, s in sigs.items():
            lo = max(0, iP - pre)
            v = s.to_numpy()
            if not v[lo:iT + 1].any():
                row[sn] = "안 켜짐"
                continue
            # 고점 날 이후 처음 켜져 있는 날(고점 날에 이미 켜져 있으면 +0). 고점 전 pre거래일 안에 켜진 적이 있으면 덧붙인다
            pre_on = bool(v[lo:iP].any())
            post = [j for j in range(iP, iT + 1) if v[j]]
            tag = " ·고점 전에도 켜짐" if pre_on else ""
            if not post:
                row[sn] = "고점 전만 켜짐" if pre_on else "안 켜짐"
                continue
            j = post[0]
            act = op.iloc[j + 1] if j + 1 < len(op) else np.nan
            done = (act / peak - 1) / trough if trough < 0 else np.nan
            row[sn] = f"고점{j - iP:+d}·저점{j - iT:+d}(진행 {done:.0%}){tag}"
        rows.append(row)
    return rows


def false_alarms(b: H.Base, sigs: dict, events: list, pre: int = 10, gap: int = 5) -> dict:
    """새로 켜진 횟수(꺼진 지 gap거래일 이상 뒤 켜짐)와 그중 사건 창(고점 pre거래일 전~저점) 밖에서 켜진 횟수."""
    pos = {d: i for i, d in enumerate(b.days)}
    win = [(pos[P] - pre, pos[T]) for _, P, T in events]
    out = {}
    for sn, s in sigs.items():
        v = s.to_numpy()
        ons, last_on = [], -10 ** 9
        for i in range(1, len(v)):
            if v[i] and not v[i - 1] and i - last_on > gap:
                ons.append(i)
            if v[i]:
                last_on = i
        outside = [i for i in ons if not any(a <= i <= z for a, z in win)]
        out[sn] = {"켜짐": len(ons), "평온기": len(outside), "켜진날비율": round(float(v.mean()), 3),
                   "평온기날짜": [b.days[i] for i in outside][:12]}
    return out


# ----------------------------------------------------------------------------- 실행
def main():
    ev = fetch_events()
    print("실적일 수:", {k: len(v) for k, v in ev["earnings"].items()}, "실패:", ev["fails"])
    b26, bh = H.base("2026"), H.base("hold")
    rows, expo = [], {}

    # 1부
    for g, syms in EVENT_GROUPS.items():
        for lvl in EVENT_EXPO:
            for win in EVENT_WINDOW:
                nm = f"사건[{g}] {win} 노출{lvl}"
                e26 = event_exposure(b26, ev, syms, lvl, win)
                eh = event_exposure(bh, ev, syms, lvl, win)
                rows.append(H.evaluate(nm, {"2026": e26, "hold": eh}, None,
                                       {"part": 1, "group": g, "expo": lvl, "window": win}))
                expo[nm] = (e26, eh)

    # 2부
    fs = load_fund_series()
    print("펀드 자료:", {k: (len(v), v.index.min(), v.index.max()) for k, v in fs.items()})
    desc = {k: {"cash_med": round(float(v.cash.median()), 2), "cash_p90": round(float(v.cash.quantile(0.9)), 2),
                "cash_max": round(float(v.cash.max()), 2), "semi_med": round(float(v.semi.median()), 1),
                "semi_min": round(float(v.semi.min()), 1), "semi_max": round(float(v.semi.max()), 1)} for k, v in fs.items()}
    print("펀드 기술통계(보고용, 문턱 결정에 쓰지 않음):", desc)
    sig_on = {}
    for kind, thr in FUND_SIGNALS:
        t26 = fund_signal(fs["TIME26"], kind, thr, 5)
        k26 = fund_signal(fs["KoAct26"], kind, thr, 5)
        th = fund_signal(fs["TIMEhold"], kind, thr, 1)
        both26 = (t26 & k26.reindex(t26.index).fillna(False))
        for fund, s26 in (("TIME", t26), ("KoAct", k26), ("둘다", both26)):
            sig_on[f"{fund} {kind}{thr}"] = (float(H.at_open(s26.astype(float), b26, 0.0).mean()),
                                             float(H.at_open(th.astype(float), bh, 0.0).mean()))
            for lvl in FUND_EXPO:
                for n in FUND_HOLD:
                    nm = f"펀드[{fund} {kind}{thr}] 노출{lvl} 유지{n}"
                    e26 = fund_exposure(b26, s26, lvl, n)
                    eh = fund_exposure(bh, th, lvl, n)
                    rows.append(H.evaluate(nm, {"2026": e26, "hold": eh}, None,
                                           {"part": 2, "fund": fund, "kind": kind, "thr": thr, "expo": lvl, "hold": n,
                                            "hold_base": "TIME 단독" if fund != "TIME" else "TIME"}))
                    expo[nm] = (e26, eh)
    assert len(rows) == 120, len(rows)

    # 고르기(하네스 규칙과 같음) → 행동 보고 대상
    br = H.base_result()
    picks = {k: H.select_front(rows, br, kp) for k, kp in (("keep90", .9), ("keep80", .8), ("keep70", .7))}
    par = sorted(H.pareto_2026(rows, br, 0.9), key=lambda r: -r["2026"]["전체"]["수익"])
    target = picks["keep90"] or picks["keep80"] or (par[0] if par else None)
    tlabel = "keep90" if picks["keep90"] else ("keep80" if picks["keep80"] else "사후 파레토 1위")
    beh = {}
    names = {r["name"] for r in [target] + [picks["keep80"], picks["keep70"]] + par[:3] if r}
    for nm in names:
        e26 = expo[nm][0]
        beh[nm] = {"trace": H.trace(e26),
                   "AI쇼크": behavior(b26, e26, "2026-06-22", "2026-07-29"),
                   "7~9월": behavior(b26, e26, "2026-07-01", "2026-09-30"),
                   "AI쇼크+반등(6/22~8/31)": behavior(b26, e26, "2026-06-22", "2026-08-31"),
                   "2~3월(2/2~4/10)": behavior(b26, e26, "2026-02-02", "2026-04-10"),
                   "2/3~2/5": behavior(b26, e26, "2026-02-03", "2026-02-05"),
                   "2/25~3/6": behavior(b26, e26, "2026-02-25", "2026-03-06"),
                   "3/19~3/30": behavior(b26, e26, "2026-03-19", "2026-03-30")}

    # 비교 기준: 같은 평균 노출을 늘 유지했을 때(타이밍 가치 = 변형 − 이것). 격자 밖 참고값.
    flat = {}
    for nm in names:
        out = {}
        for bn, bb in (("2026", b26), ("hold", bh)):
            ex = expo[nm][0 if bn == "2026" else 1]
            avg = float(ex.reindex(bb.ret.index).mean())
            rr = avg * bb.ret
            out[bn] = {"avg": round(avg, 3), **{k: {kk: round(vv, 4) for kk, vv in H.stats(rr, *bb.periods[k]).items()}
                                                 for k in (("전체", "앞", "뒤", "AI쇼크") if bn == "2026" else ("2025", "2024"))}}
        flat[nm] = out

    # 사건 창 날의 평균 수익 vs 모든 날(기제 확인). 앞 구간 값은 고르기 전에 알 수 있는 근거.
    win_mean = {}
    for g, syms in EVENT_GROUPS.items():
        row = {}
        for tag, bb, lo, hi in (("2026앞", b26, "2026-01-02", "2026-05-29"), ("2026뒤", b26, "2026-06-01", "2026-10-09"),
                                ("2025", bh, "2025-01-01", "2025-12-31"), ("2024", bh, "2024-01-01", "2024-12-31"),
                                ("2023", bh, "2023-05-22", "2023-12-31")):
            for win in EVENT_WINDOW:
                ex = event_exposure(bb, ev, syms, 0.0, win).reindex(bb.ret.index)
                rr = bb.ret[(bb.ret.index >= lo) & (bb.ret.index <= hi)]
                m = ex.reindex(rr.index) < 0.5
                row[f"{tag} {win}"] = {"n": int(m.sum()), "창평균": round(float(rr[m].mean()), 4) if m.any() else None,
                                       "전체평균": round(float(rr.mean()), 4), "창합": round(float(rr[m].sum()), 3)}
        win_mean[g] = row

    # 3부
    sig26, meta = market_signals(b26)
    sigh, _ = market_signals(bh)
    # 펀드 신호도 표에 둘(대표 2개)
    t26 = fs["TIME26"]
    sig26["TIME 반도체 1주 −4%p"] = H.at_open(fund_signal(t26, "semiD", 4.0, 5).astype(float), b26, 0.0).shift(-1).fillna(0) > 0.5
    sig26["TIME 현금 1주 +1.5%p"] = H.at_open(fund_signal(t26, "cashD", 1.5, 5).astype(float), b26, 0.0).shift(-1).fillna(0) > 0.5
    sigh["TIME 반도체 1주 −4%p"] = H.at_open(fund_signal(fs["TIMEhold"], "semiD", 4.0, 1).astype(float), bh, 0.0).shift(-1).fillna(0) > 0.5
    sigh["TIME 현금 1주 +1.5%p"] = H.at_open(fund_signal(fs["TIMEhold"], "cashD", 1.5, 1).astype(float), bh, 0.0).shift(-1).fillna(0) > 0.5
    et = event_table(b26, sig26, EVENTS["2026"]) + event_table(bh, sigh, EVENTS["hold"])
    fa = {"2026": false_alarms(b26, sig26, EVENTS["2026"]), "hold": false_alarms(bh, sigh, EVENTS["hold"])}

    extra = {"events_file": str(EV_CACHE), "earn_counts": {k: len(v) for k, v in ev["earnings"].items()},
             "earn_fails": ev["fails"], "fund_desc": desc,
             "fund_signal_on_ratio(2026,hold)": sig_on, "behavior_target": tlabel, "behavior": beh,
             "event_table": et, "false_alarms": fa, "flat_same_exposure": flat, "event_window_mean": win_mean,
             "notes": ["HY OAS는 2023-10-10부터라 23 가을 사건은 판단 불가", "펀드 신호 표의 날짜는 PDF 날짜(그날 이전 바스켓) 기준이며 체결은 다음 미국 시가",
                       "진행 = 신호 다음 시가까지 이미 진행된 하락 / 고점→저점 하락(종가 평가)"]}
    payload = H.summarize_family("H_events_funds", rows, GRID_DESC, extra)

    print("\n== 행동 대상:", tlabel, target["name"] if target else None)
    for nm, d in beh.items():
        print("\n##", nm)
        print(" trace:", d["trace"])
        for k, v in d.items():
            if k == "trace":
                continue
            print(f"  {k}: 기본 {v['base']:+.3f} 오버레이 {v['overlay']:+.3f} 아낌 {v['saved']:+.3f} 놓침 {v['missed']:+.3f} 빠짐 {v['exits']}")
            if k in ("AI쇼크", "2~3월(2/2~4/10)"):
                for s in v["segments"]:
                    print("     ", s)
    print("\n== 사건 표")
    for r in et:
        print(r)
    print("\n== 헛경보")
    for k, v in fa.items():
        for sn, x in v.items():
            print(k, sn, x)
    print("\n== 펀드 신호 켜진 비율(2026, hold):", {k: (round(a, 3), round(b, 3)) for k, (a, b) in sig_on.items()})
    print("\n== 같은 평균 노출 고정")
    for nm, v in flat.items():
        print(nm, v)
    print("\n== 사건 창 평균 수익")
    for g, v in win_mean.items():
        print(g, {k: (x["n"], x["창평균"], x["전체평균"]) for k, x in v.items()})
    return payload


if __name__ == "__main__":
    main()
