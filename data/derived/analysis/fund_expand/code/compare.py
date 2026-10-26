"""KoAct 다른 펀드 3개의 보유 종목을 넣었을 때 성과가 좋아지는가 (2026-10-12 사용자 요청).

범위(사전 등록, 결과를 보기 전에 고정):
  U0 지금(TIME + KoAct 성장기업 2ETFQ1) / U1 +로봇피지컬AI(2ETFU7) / U2 +바이오헬스케어(2ETFS9)
  U3 +치매&뇌질환(2ETFO5) / U4 네 KoAct 모두 + TIME.
  각 펀드는 그 펀드 PDF가 있는 날부터만 넣는다(실제로는 월말 PDF 11개·로봇 7개, 다음 PDF까지 이어 씀). 미국 상장만.
  주의: 기본 범위는 매일 보유라 주기가 섞인다 — 같은 조건 비교는 lag_check2.py·verify_noise.py.
전략(같은 비중, 매주 첫 거래일 시가 교체, 결정은 직전 거래일 종가, 다 채운 주만, 낙폭은 매일 종가):
  추세 등급 최종식(E14 10일 평균 + 관문) S / S+A / 이력 띠(S로 들어와 A까지 유지),
  120일 수익률 상위 7(7종목 규칙의 선택 방식만 흉내, 비중·상한 없음), 범위 전체 같은 비중, QQQ.
구간: 2026 1~5월(front)·6~10월(back)·전체. 2025는 2026 보유 합집합으로 잰 보조(생존 편향).
가격: rsi_breakout 일봉 패널(150종목) + 새 종목은 yfinance(같은 방식, 분할만 반영) — cache/prices_extra.parquet.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TG = HERE.parents[1] / "trend_grade" / "code"
sys.path.insert(0, str(TG))
import fam_composite as FC  # noqa: E402
import fam_ladder as FL  # noqa: E402
import harness as H  # noqa: E402

PDFS = HERE.parent / "pdfs"
CACHE = HERE.parent / "cache"
EXTRA = {"rob": "2ETFU7", "bio": "2ETFS9", "dem": "2ETFO5"}
VARIANTS = {"U0": [], "U1": ["rob"], "U2": ["bio"], "U3": ["dem"], "U4": ["rob", "bio", "dem"]}
WINDOWS = {"front": ("2026-01-02", "2026-05-29"), "back": ("2026-06-01", "2026-10-09"),
           "y2026": ("2026-01-02", "2026-10-09"), "y2025": ("2025-01-02", "2025-12-31")}


def us_symbol(code: str) -> str | None:
    p = str(code).split()
    if len(p) >= 3 and p[1] == "US" and p[-1].upper() == "EQUITY":
        return H.ALIASES.get(p[0], p[0]).replace("/", "-")
    return None


def fund_members(fid: str) -> dict[str, set[str]]:
    out = {}
    for f in sorted((PDFS / fid).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        syms = {us_symbol(h.get("code")) for h in d.get("holdings", [])} - {None}
        if syms:
            out[f.stem] = syms
    return out


def load_panel(extra_syms: set[str]) -> dict:
    P = H.load_panel()
    have = set(P["C"].columns) | {"QQQ"}
    need = sorted(extra_syms - have)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "prices_extra.parquet"
    got = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ts", "sym"])
    missing = [s for s in need if s not in set(got.get("sym", []))]
    if missing:
        import yfinance as yf
        frame = yf.download(missing, start="2022-06-01", end="2026-10-10", auto_adjust=False, progress=False,
                            threads=True, group_by="column")
        rows = []
        for s in missing:
            try:
                b = pd.DataFrame({k: frame[k][s] for k in ("Open", "High", "Low", "Close", "Volume")}).dropna(subset=["Close"])
            except KeyError:
                continue
            b = b[b["High"] > 0]
            b["ts"] = pd.DatetimeIndex(b.index).tz_localize(None).normalize()
            b["sym"] = s
            rows.append(b.reset_index(drop=True))
        if rows:
            got = pd.concat([got, *rows], ignore_index=True) if len(got) else pd.concat(rows, ignore_index=True)
            got.to_parquet(path)
    out = {k: v.copy() for k, v in P.items() if k != "bench"}
    out["bench"] = P["bench"]
    if len(got):
        got["ts"] = pd.to_datetime(got["ts"])
        cal = P["C"].index
        for k, col in (("O", "Open"), ("H", "High"), ("L", "Low"), ("C", "Close"), ("V", "Volume")):
            w = got.pivot(index="ts", columns="sym", values=col).reindex(cal)
            out[k] = pd.concat([out[k], w[[c for c in w.columns if c not in out[k].columns]]], axis=1)
    return out


def membership(P: dict, funds: list[str], extra: dict[str, dict[str, set[str]]], aux_all: bool) -> pd.DataFrame:
    """날짜 × 종목. 2026: TIME·2ETFQ1(정규화 스냅샷, 매일) + 고른 펀드(월말 PDF, 그 펀드 첫 PDF부터). 2026 이전: 그 범위의
    2026 보유 합집합 전체(생존 편향, aux_all)."""
    cal, cols = P["C"].index, P["C"].columns
    base = H.membership().reindex(index=cal, columns=cols).fillna(False).astype(bool)
    m = base.copy()
    first2026 = pd.Timestamp("2026-01-02")
    if aux_all:
        union = set(H.load_panel()["C"].columns)
        for f in funds:
            union |= set().union(*extra[f].values()) if extra[f] else set()
        m.loc[cal < first2026, :] = False
        m.loc[cal < first2026, [c for c in cols if c in union]] = True
    for f in funds:
        days = sorted(extra[f])
        for i, d in enumerate(days):
            a = pd.Timestamp(d)
            b = pd.Timestamp(days[i + 1]) if i + 1 < len(days) else cal[-1] + pd.Timedelta(days=1)
            rows = (cal >= a) & (cal < b)
            m.loc[rows, [c for c in extra[f][d] if c in cols]] = True
    return m


def grades(P: dict, mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    um = mask & (P["C"].notna().cumsum() >= H.MIN_BARS) & P["C"].notna()
    orig = H.universe_mask
    H.universe_mask = lambda _P: um  # 점수의 단면 백분위 모집단을 이 범위로
    try:
        sc = FC.make(FC.GRID["E14"], 10)(P)
    finally:
        H.universe_mask = orig
    return FL.hyb_grades(sc, P, um), um


def weekly(P, G, um, a, b, rule):
    O, C = P["O"], P["C"]
    cal = C.index
    d = cal[(cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))]
    wk = pd.Series(d.to_period("W-SUN"), index=d)
    firsts = [x for x, w in zip(d, wk) if x == wk[wk == w].index[0]]
    mom = C / C.shift(120) - 1
    eq, curve, held, turn, n_list, picks, span = 1.0, [], set(), 0.0, [], [], []
    for i in range(len(firsts) - 1):
        w0, w1 = firsts[i], firsts[i + 1]
        prev = cal[cal.get_loc(w0) - 1]
        if prev < pd.Timestamp(a):
            continue
        g, ok = G.loc[prev], um.loc[prev]
        if rule == "S":
            new = {s for s in g.index if ok[s] and g[s] == "S"}
        elif rule == "SA":
            new = {s for s in g.index if ok[s] and g[s] in ("S", "A")}
        elif rule == "BUF":
            new = {s for s in g.index if ok[s] and (g[s] == "S" or (s in held and g[s] in ("S", "A")))}
        elif rule == "TOP7":
            r = mom.loc[prev][ok].dropna()
            new = set(r.sort_values(ascending=False).index[:7])
        else:  # ALL
            new = {s for s in g.index if ok[s]}
        turn += len(new ^ held) / max(len(new), 1) / 2
        held = new
        names = [s for s in held if np.isfinite(O.at[w0, s]) and np.isfinite(O.at[w1, s])]
        if not names:
            continue
        n_list.append(len(names))
        picks.append(names)
        span.append((w0, w1))
        for x in d[(d >= w0) & (d < w1)]:
            curve.append(eq * float((C.loc[x, names] / O.loc[w0, names]).mean()))
        eq *= float((O.loc[w1, names] / O.loc[w0, names]).mean())
    c = pd.Series(curve)
    weeks = len(n_list)
    return {"ret": round((eq - 1) * 100, 1), "mdd": round(float((c / c.cummax() - 1).min()) * 100, 1) if len(c) else None,
            "avg_n": round(float(np.mean(n_list)), 1) if n_list else 0, "weeks": weeks,
            "turn_yr": round(turn / max(weeks, 1) * 52 * 100), "picks": picks,
            "span": [span[0][0].date().isoformat(), span[-1][1].date().isoformat()] if span else None}


def qqq(P, span):
    """전략과 같은 기간(첫 주 시가 → 마지막 주 다음 주 시가)의 QQQ 수익."""
    if not span:
        return None
    O = P["bench"]["O"]
    return round((float(O[pd.Timestamp(span[1])] / O[pd.Timestamp(span[0])]) - 1) * 100, 1)


def main():
    extra = {k: fund_members(fid) for k, fid in EXTRA.items()}
    allx = set()
    for k, v in extra.items():
        for syms in v.values():
            allx |= syms
    P = load_panel(allx)
    out = {"funds": {k: {"id": EXTRA[k], "pdf_weeks": len(v), "first": min(v) if v else None,
                         "us_names": len(set().union(*v.values())) if v else 0} for k, v in extra.items()},
           "variants": {}}
    for name, funds in VARIANTS.items():
        m = membership(P, funds, extra, aux_all=True)
        G, um = grades(P, m)
        res = {}
        for w, (a, b) in WINDOWS.items():
            res[w] = {r: weekly(P, G, um, a, b, r) for r in ("S", "SA", "BUF", "TOP7", "ALL")}
            res[w]["QQQ"] = qqq(P, res[w]["ALL"]["span"])
        # 추가 펀드에서 온 종목이 S에 든 비율(2026)
        base_m = H.membership().reindex(index=P["C"].index, columns=P["C"].columns).fillna(False).astype(bool)
        s26 = (G == "S") & um
        s26 = s26.loc["2026-01-02":]
        from_extra = (s26 & ~base_m.loc["2026-01-02":]).sum().sum()
        res["S_from_added_pct"] = round(100 * from_extra / max(int(s26.sum().sum()), 1), 1)
        res["universe_n_2026"] = round(float(um.loc["2026-01-02":].sum(axis=1).mean()), 1)
        added_days = (s26 & ~base_m.loc["2026-01-02":]).sum()
        res["S_added_top"] = {k: int(v) for k, v in added_days[added_days > 0].sort_values(ascending=False).head(12).items()}
        out["variants"][name] = res
        line = " | ".join(f"{w}: S {res[w]['S']['ret']:+.1f} SA {res[w]['SA']['ret']:+.1f} BUF {res[w]['BUF']['ret']:+.1f} "
                          f"TOP7 {res[w]['TOP7']['ret']:+.1f} ALL {res[w]['ALL']['ret']:+.1f}" for w in WINDOWS)
        print(name, f"n {res['universe_n_2026']} S_added {res['S_from_added_pct']}% |", line, flush=True)
    for v in out["variants"].values():
        for w in WINDOWS:
            for r in ("S", "SA", "BUF", "TOP7", "ALL"):
                v[w][r].pop("picks", None)
    (HERE.parent / "results").mkdir(exist_ok=True)
    (HERE.parent / "results" / "compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
