"""breakout 계열(2026-10-10): 신고가 돌파에 사고, 추적 손절 또는 20일선 이탈에 판다. 사용자 예: PLTR 10/9 전고점 돌파.

미리 정한 격자 3 매수 × 3 매도 = 9개(뒤에 보고 추가한 변형 없음).
  매수 A  k-1일 종가 > k-2일까지 252거래일 고가 최고(52주 신고가 종가 돌파)
  매수 B  k-1일 종가 > k-2일까지 756거래일 고가 최고(약 3년, 2023-01부터 자료라 전고점 대용)
  매수 C  A 이고 k-1일 마지막 160분봉 RSI > 70
  매도    추적 10%(k-1일 종가 < 보유 중 종가 최고 × 0.90) / 추적 15% / k-1일 종가 < k-1일 20일 종가 평균
  순위    빈자리보다 신호가 많으면 k-1일 QQQ 대비 20일 상대수익 큰 순. 7자리, 체결 k일 시가.
시점      k일 시가 결정은 k-1일 이전 행만 쓴다(신고가 기준은 k-2일까지 → 표에서 hi.shift(1)의 k-1행).
선택      앞 1/2~5/29 수익 최고(앞 거래 20건 미만·앞 평균 보유 3 미만 제외) → 뒤·9/1~·10/1~ 은 확인용.
실행      cd data/derived/analysis/rsi_breakout/code && py variants/breakout.py
결과      results/variants/breakout.json"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from harness import Harness, table, months_table  # noqa: E402
from rsi_breakout import RES  # noqa: E402

FRONT, BACK = "앞 1/2~5/29", "뒤 6/1~10/9"


def build(H):
    cls = H.cls[H.syms]
    r160 = H.r160.reindex(columns=H.syms)
    ea = cls > H.hi(252)[H.syms].shift(1)          # 행 j: j일 종가 > j-1일까지 52주 고가 → k일 결정은 행 k-1
    eb = cls > H.hi(756)[H.syms].shift(1)
    ec = ea & (r160 > 70)
    col = {s: i for i, s in enumerate(H.syms)}
    C = cls.to_numpy()
    MA = H.ma(20)[H.syms].to_numpy()
    E = {"A": ea.to_numpy(), "B": eb.to_numpy(), "C": ec.to_numpy()}
    entries = {
        "A": ("52주 신고가", lambda s, k, e=E["A"]: bool(e[k - 1, col[s]])),
        "B": ("3년 신고가", lambda s, k, e=E["B"]: bool(e[k - 1, col[s]])),
        "C": ("52주+160RSI>70", lambda s, k, e=E["C"]: bool(e[k - 1, col[s]])),
    }

    def trail(x):
        return lambda s, k, p: bool(C[k - 1, col[s]] < (1 - x) * p["peak"])
    exits = {
        "T10": ("추적10%", trail(0.10)),
        "T15": ("추적15%", trail(0.15)),
        "MA20": ("종가<MA20", lambda s, k, p: bool(C[k - 1, col[s]] < MA[k - 1, col[s]])),
    }
    return entries, exits, {"A": ea, "B": eb, "C": ec}


def trade_list(H, df):
    """보유 열에서 거래(연속 보유 구간)를 뽑는다. 수익은 산 시가→판 시가(끝까지 들고 있으면 마지막 시가)."""
    days = df.day.tolist()
    held = [set(h.split(",")) if h else set() for h in df.held]
    pos = {d: i for i, d in enumerate(H.days)}
    start, out = {}, []
    for i in range(len(held) + 1):
        cur = held[i] if i < len(held) else set()
        prev = held[i - 1] if i else set()
        for s in cur - prev:
            start[s] = i
        for s in prev - cur:
            i0 = start.pop(s)
            k_in = pos[days[i0]]
            k_out = pos[days[i]] if i < len(held) else len(H.days) - 1
            a, b = H.opn[s].iloc[k_in], H.opn[s].iloc[k_out]
            out.append(dict(sym=s, buy=days[i0], sell=H.days[k_out] if i < len(held) else None,
                            days=i - i0, ret=float(b / a - 1)))
    return pd.DataFrame(out)


def oct_contrib(H, df):
    """10/1~ 구간 종목별 기여(매일 비중 × 시가→다음 시가, 단순합)."""
    pos = {d: i for i, d in enumerate(H.days)}
    c = {}
    for _, r in df[df.day >= "2026-10-01"].iterrows():
        hs = r.held.split(",") if r.held else []
        k = pos[r.day]
        for s in hs:
            a, b = H.opn[s].iloc[k], H.opn[s].iloc[k + 1]
            if np.isfinite(a) and np.isfinite(b):
                c[s] = c.get(s, 0.0) + (b / a - 1) / max(7, len(hs))
    return sorted(((s, round(float(v), 4)) for s, v in c.items()), key=lambda kv: -abs(kv[1]))


def next_decisions(H, pick, df, ef, xf, prio):
    """표시만(수익 없음): 10/9 시가 결정은 같은 picker로 한 번 더, 10/12 시가 결정은 10/9 종가까지로 같은 규칙을 손으로 잰다."""
    n = len(H.days)
    last = df.held.iloc[-1].split(",") if df.held.iloc[-1] else []
    h9 = pick(n - 1, last)                                   # 10/9 시가 보유(10/8 종가까지)
    runs = [set(h.split(",")) if h else set() for h in df.held] + [set(h9)]
    keep, sold = [], []
    for s in h9:
        i = len(runs) - 1
        while i > 0 and s in runs[i - 1]:
            i -= 1
        k0 = n - len(runs) + i                              # runs[-1] ↔ 10/9(행 n-1)
        p = {"k0": k0, "px": float(H.opn[s].iloc[k0]),
             "peak": float(max(H.opn[s].iloc[k0], np.nanmax(H.cls[s].iloc[k0:n].to_numpy())))}
        (sold if xf(s, n, p) else keep).append(s)
    elig = H.eligible("2026-10-12")
    new = [s for s in H.syms if s not in keep and s in elig and ef(s, n)]
    pr = prio(n)
    new.sort(key=lambda s: -(pr.get(s) if np.isfinite(pr.get(s, np.nan)) else -1e9))
    return dict(held_1009=sorted(h9), sell_1012=sorted(sold), signals_1012=new,
                buy_1012=new[:max(0, 7 - len(keep))], pltr_eligible_1012=("PLTR" in elig))


def pltr_dates(H, sig):
    """PLTR: 9/1~10/9 종가 기준 신호가 선 날(그 다음 거래일 시가에 산다)과 그날 담을 수 있었나."""
    s = "PLTR"
    if s not in H.syms:
        return {"note": "PLTR이 대상 종목에 없음"}
    out = {}
    nxt = {d: (H.days[i + 1] if i + 1 < len(H.days) else "2026-10-12(자료 밖)") for i, d in enumerate(H.days)}
    for e, tbl in sig.items():
        rows = []
        for d in tbl.index[(tbl.index >= "2026-09-01") & (tbl.index <= "2026-10-09")]:
            if bool(tbl.at[d, s]):
                buy = nxt[d]
                rows.append(dict(signal_close=d, buy_open=buy,
                                 eligible=(s in H.eligible(buy)) if buy in H.days else None))
        out[e] = rows
    i = H.days.index("2026-10-09")
    out["10/9 값"] = dict(close=float(H.cls[s].iloc[i]), hi252_prev=float(H.hi(252)[s].iloc[i - 1]),
                         hi756_prev=float(H.hi(756)[s].iloc[i - 1]), r160=float(H.r160[s].iloc[i]),
                         rs20=float(H.rs(20)[s].iloc[i]))
    return out


def main():
    H = Harness()
    entries, exits, sig = build(H)
    base = {r["name"]: r for r in json.load(open(os.path.join(RES, "variants", "baselines.json"), encoding="utf-8"))}
    qqq_b, desk_b = base["QQQ"][BACK]["수익"], base["데스크 7종목(#774)"][BACK]["수익"]
    prio = lambda k: H.rs(20).iloc[k - 1]   # noqa: E731
    pl = pltr_dates(H, sig)
    results = []
    for ek, (en, ef) in entries.items():
        for xk, (xn, xf) in exits.items():
            name = f"돌파 {ek}:{en} / {xn}"
            pick = H.entry_exit_picker(ef, xf, priority=prio)
            df = H.simulate(pick)
            r = H.evaluate(name, df)
            tr = trade_list(H, df)
            fr = df[(df.day >= "2026-01-02") & (df.day <= "2026-05-29")]
            ftr = tr[(tr.buy >= "2026-01-02") & (tr.buy <= "2026-05-29")] if len(tr) else tr
            r["params"] = dict(family="breakout", entry=ek, entry_desc=en, exit=xk, exit_desc=xn,
                               priority="rs(20) at k-1", slots=7, grid="3 entries x 3 exits = 9")
            r["거래수"] = int(len(tr))
            r["앞거래수"] = int(len(ftr))
            r["앞평균보유"] = float(fr.n.mean())
            r["평균보유일"] = float(tr.days.mean()) if len(tr) else None
            r["승률"] = float((tr.ret > 0).mean()) if len(tr) else None
            r["거래평균수익"] = float(tr.ret.mean()) if len(tr) else None
            r["최악거래"] = (tr.loc[tr.ret.idxmin(), ["sym", "buy", "ret"]].to_dict() if len(tr) else None)
            r["10월기여"] = oct_contrib(H, df)
            r["10월보유"] = df[df.day >= "2026-10-01"].held.iloc[-1] if (df.day >= "2026-10-01").any() else ""
            r["PLTR거래"] = tr[tr.sym == "PLTR"].to_dict("records") if len(tr) else []
            r["PLTR신호"] = pl.get(ek)
            r["10/12결정(표시만)"] = next_decisions(H, pick, df, ef, xf, lambda k: H.rs(20).iloc[k - 1])
            r["앞뒤모두+"] = bool(r[FRONT]["수익"] > 0 and r[BACK]["수익"] > 0)
            r["뒤>QQQ"] = bool(r[BACK]["수익"] > qqq_b)
            r["뒤>데스크"] = bool(r[BACK]["수익"] > desk_b)
            results.append(r)

    ok = [r for r in results if r["앞거래수"] >= 20 and r["앞평균보유"] >= 3]
    best = max(ok, key=lambda r: r[FRONT]["수익"]) if ok else None
    os.makedirs(os.path.join(RES, "variants"), exist_ok=True)
    for r in results:
        r["선택"] = bool(best is not None and r["name"] == best["name"])
    json.dump(results,                       # baselines.json과 같은 목록 형식, 변형마다 params·PLTR신호
              open(os.path.join(RES, "variants", "breakout.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)

    print(table(results))
    print(months_table(results))
    print("\n  이름 | 거래(앞) | 앞 평균보유 | 보유일 | 승률 | 거래평균 | 최악 | 앞뒤+ | 뒤>QQQ | 뒤>데스크")
    for r in results:
        w = r["최악거래"] or {}
        print(f"  {r['name'][:30].ljust(30)} {r['거래수']:4d}({r['앞거래수']:3d}) {r['앞평균보유']:5.2f} "
              f"{r['평균보유일'] or 0:5.1f} {r['승률'] or 0:5.0%} {r['거래평균수익'] or 0:+6.1%} "
              f"{w.get('sym', '')} {w.get('ret', 0):+.0%} | {r['앞뒤모두+']} {r['뒤>QQQ']} {r['뒤>데스크']}")
    print("\n  선택(앞 기준):", best["name"] if best else "없음")
    if best:
        print("   ", {p: round(best[p]["수익"], 4) for p in ("앞 1/2~5/29", "뒤 6/1~10/9", "9/1~10/9", "10/1~10/9", "전체")},
              "낙폭", round(best["전체"]["낙폭"], 4))
    print("\n  10월 보유·기여(상위)")
    for r in results:
        print(f"  {r['name'][:30].ljust(30)} 보유[{r['10월보유']}] 기여 {r['10월기여'][:5]}")
    print("\n  PLTR 9/1~10/9 신호(종가일→매수 시가일, 담을 수 있었나)")
    for e in ("A", "B", "C"):
        print("  ", e, [(x["signal_close"], x["buy_open"], x["eligible"]) for x in pl.get(e, [])])
    print("   10/9 값", pl.get("10/9 값"))
    print("\n  10/12 시가 결정(10/9 종가까지, 표시만·수익 없음): 팔 것 / 신호(rs20 순) / 살 것")
    for r in results:
        d = r["10/12결정(표시만)"]
        print(f"  {r['name'][:30].ljust(30)} 팔 {d['sell_1012']} 신호 {d['signals_1012'][:8]} 살 {d['buy_1012']}"
              f" PLTR 대상 {d['pltr_eligible_1012']}")
    for r in results:
        if r["PLTR거래"]:
            print("  ", r["name"], "PLTR 거래", [(t["buy"], t["sell"], round(t["ret"], 3)) for t in r["PLTR거래"]])
    bad = [r["name"] for r in results if any(not np.isfinite(r[p]["수익"]) for p in ("앞 1/2~5/29", "뒤 6/1~10/9", "전체"))
           or r["평균보유"] <= 0]
    print("\n  점검: NaN·보유0 변형", bad or "없음")


if __name__ == "__main__":
    main()
