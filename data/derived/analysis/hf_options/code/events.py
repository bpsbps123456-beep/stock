"""갈래 events(설명용, 판정 없음): QQQ·SOXX 7% 하락 직전 고점과 그 바닥에서 24개 신호가 어땠나.

- 사건: 2025-01-02부터 종가 기준 누적 고점 → 다음 새 고점 전 최저 종가가 −7% 이하면 (고점, 바닥) 한 쌍.
- 사건일 행 = 그날 시가에 알 수 있던 정보(harness 패널은 진입일 색인). 창 = 사건일 포함 직전 10행(d−9 … d).
- 기준: 2025-01-02 ~ 마지막 날 아무 날의 꼬리 빈도(≥0.8 / ≤0.2), 아무 10행 창의 꼬리 빈도,
  그리고 '사건 아닌 20일 종가 고점·저점일'(같은 자산, 사건 ±5거래일 밖)의 같은 빈도.
- 무리(군집): 날짜가 3거래일 안에 겹치는 같은 종류 사건은 하나로(첫날 대표).
- 묶음 점수: 개인 환호(8개 방향 맞춘 평균), 큰손 위험 선호(11개). 정의는 아래 상수(결과 보기 전 정함).
- 같은 구성 재발: 사건일에 가장 치우친 큰손 신호 1개 + 개인 신호 1개가 같은 꼬리에 함께 있던 날을 2025~2026에서
  모두 찾아 묶음(5거래일 넘게 끊기면 새 묶음)으로 세고, 그 묶음 뒤 10거래일 안에 같은 종류 사건이 있었는지 센다.
결과: results/events.json, results/events.md(표 자동 + 서술은 NARR 상수).
"""
from __future__ import annotations

import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import binom

import harness as H

DD, START, WIN, HI, LO = -7.0, "2025-01-02", 10, 0.8, 0.2
MERGE, EXCL, AHEAD, GAP = 3, 5, 10, 5
OFFS = (-20, -15, -10, -5, 0)
N_NULL, SEED = 2000, 23

# 묶음과 '위험 선호·환호' 방향(+1 = 원값 높을수록 환호/위험 선호). 결과를 보기 전에 정함.
RETAIL = {"aaii_spread": 1, "lev_qqq_bull5": 1, "lev_soxx_bull5": 1, "lev_qqq_flow20p": 1,
          "eq_pcr5": -1, "ai_cpcr5": -1, "ai_breadth5": 1, "nq_nonrept_net_oi": 1}
BIG = {"nq_lev_net_oi": 1, "nq_lev_chg4": 1, "nq_am_net_oi": 1, "nq_am_chg4": 1, "es_lev_net_oi": 1,
       "vx_lev_net_oi": -1, "dix": 1, "gvip_rel20": 1, "si_ai_chg": -1, "qai_beta60": 1, "naaim": 1}
OTHER = ["skew", "vix_ts", "gex", "nq_dealer_net_oi", "shvr_ai5"]  # 방향이 한쪽 주체로 읽히지 않는 것
ORDER = list(RETAIL) + list(BIG) + OTHER

KO = {"aaii_spread": "AAII 강세−약세", "lev_qqq_bull5": "TQQQ 거래대금 몫", "lev_soxx_bull5": "SOXL 거래대금 몫",
      "lev_qqq_flow20p": "TQQQ−SQQQ 20일 순유입", "eq_pcr5": "CBOE 주식 풋/콜", "ai_cpcr5": "OCC AI 고객 풋/콜",
      "ai_breadth5": "AI 콜 쏠림 종목 폭", "nq_nonrept_net_oi": "NQ 소액 순포지션",
      "nq_lev_net_oi": "NQ 레버리지 펀드 순", "nq_lev_chg4": "NQ 레버리지 4주 변화", "nq_am_net_oi": "NQ 자산운용 순",
      "nq_am_chg4": "NQ 자산운용 4주 변화", "es_lev_net_oi": "ES 레버리지 펀드 순", "vx_lev_net_oi": "VIX 레버리지 순",
      "dix": "DIX(다크풀)", "gvip_rel20": "GVIP/QQQ 20일", "si_ai_chg": "AI 공매도 잔고 변화", "qai_beta60": "QAI 베타",
      "naaim": "NAAIM 노출", "skew": "SKEW", "vix_ts": "VIX/VIX3M", "gex": "GEX(딜러 감마)",
      "nq_dealer_net_oi": "NQ 딜러 순", "shvr_ai5": "AI 공매도 거래량 비율"}


# ---------- 사건 ----------
def swings(c: pd.Series) -> list[dict]:
    """options_sentiment/evaluate.tops와 같은 규칙(누적 고점 → 새 고점 전 최저가 DD 이하)."""
    out, peak, pdate, tr, tdate = [], c.iloc[0], c.index[0], c.iloc[0], c.index[0]
    for d, v in c.items():
        if v > peak:
            if (tr / peak - 1) * 100 <= DD:
                out.append({"peak": pdate, "trough": tdate, "dd": round((tr / peak - 1) * 100, 1), "ongoing": False})
            peak, pdate, tr, tdate = v, d, v, d
        elif v < tr:
            tr, tdate = v, d
    if (tr / peak - 1) * 100 <= DD:
        out.append({"peak": pdate, "trough": tdate, "dd": round((tr / peak - 1) * 100, 1), "ongoing": True})
    return out


def event_list(px: pd.DataFrame) -> list[dict]:
    ev = []
    for a in ("QQQ", "SOXX"):
        c = px[f"{a}_c"]
        for s in swings(c[c.index >= START]):
            ev.append({"kind": "top", "asset": a, "date": s["peak"], "pair": s["trough"], "dd": s["dd"], "ongoing": s["ongoing"]})
            ev.append({"kind": "bottom", "asset": a, "date": s["trough"], "pair": s["peak"], "dd": s["dd"], "ongoing": s["ongoing"]})
    return sorted(ev, key=lambda e: (e["kind"], e["date"], e["asset"]))


def clusters(ev: list[dict], pos: dict) -> list[dict]:
    out = []
    for kind in ("top", "bottom"):
        for e in sorted([e for e in ev if e["kind"] == kind], key=lambda e: e["date"]):
            if out and out[-1]["kind"] == kind and pos[e["date"]] - pos[out[-1]["date"]] <= MERGE:
                out[-1]["members"].append(f"{e['asset']} {e['date']}")
                continue
            out.append({"kind": kind, "date": e["date"], "members": [f"{e['asset']} {e['date']}"],
                        "assets": e["asset"], "dd": e["dd"], "pair": e["pair"]})
    for c in out:
        c["assets"] = "+".join(sorted({m.split()[0] for m in c["members"]}, key=["QQQ", "SOXX"].index))
    return out


# ---------- 원값(패널과 같은 시점 규칙, 서술용) ----------
def raw_panel(dd: pd.Index) -> pd.DataFrame:
    import fetch as osf  # options_sentiment 수집기(harness가 경로를 넣음)
    out = {}
    cb = osf.load_cboe()
    eq = cb["equity_pv"].rolling(5).sum() / cb["equity_cv"].rolling(5).sum()
    out["eq_pcr5"] = H._place(eq, pd.Series(eq.index, index=eq.index), dd, H.DAILY_AGE)
    sq = pd.read_parquet(H.CACHE / "squeeze.parquet").set_index("asof")
    ws = pd.Series(sq.index, index=sq.index)
    out["dix"] = H._place(sq["dix"] * 100, ws, dd, H.DAILY_AGE)
    out["gex_bn"] = H._place(sq["gex_bn"], ws, dd, H.DAILY_AGE)
    hp = pd.read_parquet(H.CACHE / "hf_proxies.parquet")
    wk = pd.Series(hp.index, index=hp.index)
    for c, k in (("lev_qqq_bull5", 100), ("lev_soxx_bull5", 100), ("gvip_rel20", 100), ("qai_beta60", 1)):
        out[c] = H._place(hp[c] * k, wk, dd, H.DAILY_AGE)
    fl = hp.dropna(subset=["lev_qqq_flow20"]).set_index("asof")
    wf = pd.Series(fl.index, index=fl.index)
    out["lev_qqq_flow20_bn"] = H._place(fl["lev_qqq_flow20"] / 1e9, wf, dd, H.DAILY_AGE)
    out["lev_qqq_flow20p"] = H._place(fl["lev_qqq_flow20"] / (fl["aum_tqqq"] + fl["aum_sqqq"]) * 100, wf, dd, H.DAILY_AGE)
    cf = pd.read_parquet(H.CACHE / "cftc_tff.parquet")
    wc = pd.Series(cf.index, index=cf.index)
    for nm, num, den in (("nq_lev_net_oi", "nq_lev_net", "nq_oi"), ("nq_am_net_oi", "nq_am_net", "nq_oi"),
                         ("es_lev_net_oi", "es_lev_net", "es_oi"), ("vx_lev_net_oi", "vx_lev_net", "vx_oi"),
                         ("nq_nonrept_net_oi", "nq_nonrept_net", "nq_oi")):
        out[nm] = H._place(cf[num] / cf[den] * 100, wc, dd, H.WEEKLY_AGE)
    cfa = pd.Series(pd.to_datetime(cf["asof"]).map(pd.Timestamp.toordinal).astype(float).to_numpy(), index=cf.index)
    out["cftc_asof_ord"] = H._place(cfa, wc, dd, H.WEEKLY_AGE)
    si = pd.read_parquet(H.CACHE / "finra_si.parquet")
    out["si_ai_chg"] = H._place(si["si_ai_chg"] * 100, pd.Series(si.index, index=si.index), dd, H.BIMONTH_AGE)
    sv = pd.read_parquet(H.CACHE / "surveys.parquet")
    a = sv["aaii_spread"].dropna()
    out["aaii_spread"] = H._place(a, pd.Series(a.index, index=a.index), dd, H.WEEKLY_AGE)
    n = sv["naaim"].dropna()
    out["naaim"] = H._place(n, pd.Series(n.index, index=n.index), dd, H.WEEKLY_AGE)
    R = pd.DataFrame(out, index=dd)
    R["cftc_asof"] = R.pop("cftc_asof_ord").map(lambda v: pd.Timestamp.fromordinal(int(v)).strftime("%Y-%m-%d") if v == v else None)
    return R


def cftc_by_asof() -> pd.DataFrame:
    """CFTC 8개 신호의 52주 백분위를 기준일(asof)로 색인. 사건 당시 공개 전이었을 수 있는 '사후' 값(설명용)."""
    cf = pd.read_parquet(H.CACHE / "cftc_tff.parquet")
    nq_lev, nq_am = cf["nq_lev_net"] / cf["nq_oi"], cf["nq_am_net"] / cf["nq_oi"]
    ser = {"nq_lev_net_oi": nq_lev, "nq_lev_chg4": nq_lev.diff(4), "nq_am_net_oi": nq_am, "nq_am_chg4": nq_am.diff(4),
           "nq_dealer_net_oi": cf["nq_dealer_net"] / cf["nq_oi"], "es_lev_net_oi": cf["es_lev_net"] / cf["es_oi"],
           "vx_lev_net_oi": cf["vx_lev_net"] / cf["vx_oi"], "nq_nonrept_net_oi": cf["nq_nonrept_net"] / cf["nq_oi"]}
    out = pd.DataFrame({k: H.rolling_pct(v, 52) for k, v in ser.items()})
    out["known"] = cf.index
    out["asof"] = cf["asof"].to_numpy()
    return out.set_index("asof")


# ---------- 점수·창 ----------
def group_score(P: pd.DataFrame, spec: dict) -> pd.Series:
    parts = pd.DataFrame({s: (P[s] if d > 0 else 1 - P[s]) for s, d in spec.items()})
    return parts.mean(axis=1).where(parts.notna().sum(axis=1) >= (len(spec) + 1) // 2)


def episodes(flag: pd.Series, gap: int = GAP) -> list[tuple[int, int]]:
    idx = np.flatnonzero(flag.fillna(False).to_numpy(bool))
    eps = []
    for i in idx:
        if eps and i - eps[-1][1] <= gap:
            eps[-1][1] = i
        else:
            eps.append([i, i])
    return [tuple(e) for e in eps]


def main() -> None:
    P = H.panel()
    idx = P.index
    pos = {d: i for i, d in enumerate(idx)}
    px = H.prices().reindex(idx)
    base = np.asarray(idx >= START)
    Wmax = P.rolling(WIN, min_periods=1).max()
    Wmin = P.rolling(WIN, min_periods=1).min()
    R = raw_panel(idx)
    ev = event_list(H.prices())
    cl = clusters(ev, pos)
    re_s, bg_s = group_score(P, RETAIL), group_score(P, BIG)
    cft = [k for k in BIG if k.startswith(("nq_", "es_", "vx_"))]
    bg_c = group_score(P, {k: BIG[k] for k in cft})  # CFTC 6개(2025 셧다운 때 빔)
    bg_o = group_score(P, {k: v for k, v in BIG.items() if k not in cft})  # 나머지 5개

    # ---- 기준 빈도: 아무 날 / 아무 10행 창 / 사건 아닌 20일 고점·저점일 ----
    loc_hi, loc_lo = {}, {}
    for a in ("QQQ", "SOXX"):
        c = px[f"{a}_c"]
        loc_hi[a] = (c >= c.rolling(20, min_periods=20).max()).to_numpy()
        loc_lo[a] = (c <= c.rolling(20, min_periods=20).min()).to_numpy()

    def away(kind: str, asset: str) -> np.ndarray:
        m = np.ones(len(idx), bool)
        for e in ev:
            if e["kind"] == kind and e["asset"] == asset:
                p = pos[e["date"]]
                m[max(0, p - EXCL):p + EXCL + 1] = False
        return m

    ctrl = {"top": np.zeros(len(idx), bool), "bottom": np.zeros(len(idx), bool)}
    for a in ("QQQ", "SOXX"):
        ctrl["top"] |= loc_hi[a] & away("top", a)
        ctrl["bottom"] |= loc_lo[a] & away("bottom", a)
    ctrl = {k: v & base for k, v in ctrl.items()}

    conds = {"day_hi": (P >= HI), "day_lo": (P <= LO), "win_hi": (Wmax >= HI), "win_lo": (Wmin <= LO)}
    notna = {"day_hi": P.notna(), "day_lo": P.notna(), "win_hi": Wmax.notna(), "win_lo": Wmin.notna()}

    def rate(mask: np.ndarray) -> dict:
        return {s: {c: float(conds[c][s][mask & notna[c][s].to_numpy()].mean()) for c in conds} for s in ORDER}

    base_rates = {"random": rate(base), "ctrl_top": rate(ctrl["top"]), "ctrl_bottom": rate(ctrl["bottom"])}
    n_ctrl = {k: int(v.sum()) for k, v in ctrl.items()}

    # ---- 사건별 값 ----
    def snap(d: str) -> dict:
        p = pos[d]
        rec = {}
        for s in ORDER:
            rec[s] = {"day": P[s].iloc[p], "max10": Wmax[s].iloc[p], "min10": Wmin[s].iloc[p],
                      "n10": int(P[s].iloc[max(0, p - WIN + 1):p + 1].notna().sum()),
                      "traj": {str(o): (P[s].iloc[p + o] if p + o >= 0 else np.nan) for o in OFFS}}
        return rec

    def raw_snap(d: str) -> dict:
        p = pos[d]
        r = {c: (R[c].iloc[p] if c != "cftc_asof" else R[c].iloc[p]) for c in R.columns}
        r["traj"] = {c: {str(o): R[c].iloc[p + o] for o in OFFS} for c in R.columns if c != "cftc_asof"}
        if isinstance(r["cftc_asof"], str):
            r["cftc_age_td"] = int(p - np.searchsorted(idx.to_numpy(), r["cftc_asof"]))
        return r

    for e in ev:
        e["signals"] = snap(e["date"])
    for c in cl:
        d, p = c["date"], pos[c["date"]]
        c["signals"] = snap(d)
        c["raw"] = raw_snap(d)
        c["retail_score"] = {str(o): re_s.iloc[p + o] for o in OFFS}
        c["big_score"] = {str(o): bg_s.iloc[p + o] for o in OFFS}
        c["big_cftc_score"] = {str(o): bg_c.iloc[p + o] for o in OFFS}
        c["big_other_score"] = {str(o): bg_o.iloc[p + o] for o in OFFS}
        cba = cftc_by_asof()
        last = cba[cba.index < d].iloc[-1]
        c["cftc_hindsight"] = {"asof": last.name, "known": last["known"], **{k: float(last[k]) for k in cba.columns if k != "known"}}
        c["ret20_into"] = {a: float((px[f"{a}_c"].iloc[p] / px[f"{a}_c"].iloc[p - 20] - 1) * 100) for a in ("QQQ", "SOXX")}
        c["span_td"] = abs(pos[c["pair"]] - p)
        c["retail_score_pct"] = float((re_s[base].dropna() <= re_s.iloc[p]).mean())
        c["big_score_pct"] = float((bg_s[base].dropna() <= bg_s.iloc[p]).mean())

    # ---- 사건 무리 적중 수 vs 기준 ----
    hits = {}
    for kind in ("top", "bottom"):
        ps = [pos[c["date"]] for c in cl if c["kind"] == kind]
        for s in ORDER:
            for cnd in conds:
                v = conds[cnd][s].to_numpy()[ps]
                ok = notna[cnd][s].to_numpy()[ps]
                k, n = int(v[ok].sum()), int(ok.sum())
                b = base_rates["random"][s][cnd]
                bc = base_rates[f"ctrl_{kind}"][s][cnd]
                hits[f"{kind}|{s}|{cnd}"] = {"k": k, "n": n, "base": b, "ctrl": bc,
                                             "p_random": float(binom.sf(k - 1, n, b)) if n else np.nan,
                                             "p_ctrl": float(binom.sf(k - 1, n, bc)) if n else np.nan}

    # 우연 기준: 같은 수의 날짜를 무작위로(아무 날 / 사건 아닌 고점·저점일) 뽑았을 때 p<0.05 칸 수
    rng = np.random.default_rng(SEED)
    pool_rand = np.flatnonzero(base)
    obs_cnt, null_cnt = {}, {}
    for kind in ("top", "bottom"):
        npick = sum(c["kind"] == kind for c in cl)
        for ref in ("random", "ctrl"):
            obs_cnt[f"{kind}|{ref}"] = int(sum(hits[f"{kind}|{s}|{c}"][f"p_{ref}"] < 0.05 for s in ORDER for c in conds))
            pool = pool_rand if ref == "random" else np.flatnonzero(ctrl[kind])
            cnts = []
            for _ in range(N_NULL):
                ps = rng.choice(pool, npick, replace=False)
                c_ = 0
                for s in ORDER:
                    for cnd in conds:
                        v, ok = conds[cnd][s].to_numpy()[ps], notna[cnd][s].to_numpy()[ps]
                        k, n = int(v[ok].sum()), int(ok.sum())
                        b = base_rates["random" if ref == "random" else f"ctrl_{kind}"][s][cnd]
                        c_ += bool(n and binom.sf(k - 1, n, b) < 0.05)
                cnts.append(c_)
            cnts = np.array(cnts)
            null_cnt[f"{kind}|{ref}"] = {"mean": float(cnts.mean()), "p95": float(np.percentile(cnts, 95)),
                                         "p": float(((cnts >= obs_cnt[f"{kind}|{ref}"]).sum() + 1) / (N_NULL + 1))}

    # ---- 같은 구성 재발: 사건일 가장 치우친 큰손 1 + 개인 1이 같은 꼬리 ----
    ev_pos = {k: sorted(pos[e["date"]] for e in ev if e["kind"] == k) for k in ("top", "bottom")}

    def followed(kind: str, a: int, b: int) -> bool:
        return any(a <= q <= b + AHEAD for q in ev_pos[kind])

    for c in cl:
        p = pos[c["date"]]
        pick = {}
        for g, spec in (("big", BIG), ("retail", RETAIL)):
            vals = {s: P[s].iloc[p] for s in spec if P[s].iloc[p] == P[s].iloc[p]}
            s = max(vals, key=lambda k: abs(vals[k] - 0.5))
            pick[g] = (s, vals[s])
        flag = pd.Series(True, index=idx)
        desc = []
        in_tail = True
        for g in ("big", "retail"):
            s, v = pick[g]
            if v >= HI:
                flag &= P[s] >= HI
                desc.append(f"{s}>={HI}")
            elif v <= LO:
                flag &= P[s] <= LO
                desc.append(f"{s}<={LO}")
            else:
                in_tail = False
                hi = v >= 0.5
                flag &= (P[s] >= 0.7) if hi else (P[s] <= 0.3)
                desc.append(f"{s}{'>=0.7' if hi else '<=0.3'}")
        flag = flag & pd.Series(base, index=idx)
        eps = episodes(flag)
        with_ev = [e for e in eps if followed(c["kind"], *e)]
        c["config"] = {"big": pick["big"][0], "big_v": pick["big"][1], "retail": pick["retail"][0], "retail_v": pick["retail"][1],
                       "rule": " & ".join(desc), "both_in_0.8_tail": in_tail, "n_days": int(flag.sum()),
                       "n_episodes": len(eps), "n_episodes_with_event_10d": len(with_ev),
                       "n_episodes_without": len(eps) - len(with_ev),
                       "episodes": [{"start": idx[a], "end": idx[b], "n": int(flag.iloc[a:b + 1].sum()),
                                     "event_10d": followed(c["kind"], a, b),
                                     **{f"{x}_fwd10": (float((px[f"{x}_c"].iloc[b + AHEAD] / px[f"{x}_c"].iloc[b] - 1) * 100)
                                                       if b + AHEAD < len(idx) else None) for x in ("QQQ", "SOXX")}}
                                    for a, b in eps]}

    # 묶음 점수 3×3 상태(2025~2026 삼분위)
    q = {g: np.nanquantile(s[base].dropna(), [1 / 3, 2 / 3]) for g, s in (("retail", re_s), ("big", bg_s))}

    def lvl(v, g):
        return np.where(np.isnan(v), np.nan, np.where(v <= q[g][0], 0, np.where(v >= q[g][1], 2, 1)))

    st_r, st_b = lvl(re_s.to_numpy(), "retail"), lvl(bg_s.to_numpy(), "big")
    nxt = {k: np.array([followed(k, i, i) for i in range(len(idx))]) for k in ("top", "bottom")}
    state_tab = {}
    for r_, b_ in product(range(3), range(3)):
        m = base & (st_r == r_) & (st_b == b_)
        state_tab[f"r{r_}b{b_}"] = {"n_days": int(m.sum()), "top_10d": float(nxt["top"][m].mean()) if m.sum() else np.nan,
                                    "bottom_10d": float(nxt["bottom"][m].mean()) if m.sum() else np.nan}
    for g, st in (("r", st_r), ("b", st_b)):
        for k_ in range(3):
            m = base & (st == k_)
            state_tab[f"{g}{k_}"] = {"n_days": int(m.sum()), "top_10d": float(nxt["top"][m].mean()), "bottom_10d": float(nxt["bottom"][m].mean())}
    allm = base & ~np.isnan(st_r) & ~np.isnan(st_b)
    state_tab["all"] = {"n_days": int(allm.sum()), "top_10d": float(nxt["top"][allm].mean()), "bottom_10d": float(nxt["bottom"][allm].mean())}
    for c in cl:
        p = pos[c["date"]]
        c["state"] = f"r{int(st_r[p])}b{int(st_b[p])}" if not (np.isnan(st_r[p]) or np.isnan(st_b[p])) else None

    res = {"rules": {"DD": DD, "start": START, "win": WIN, "hi": HI, "lo": LO, "merge_td": MERGE, "ctrl_excl_td": EXCL,
                     "ahead_td": AHEAD, "episode_gap_td": GAP, "panel_last": idx[-1], "retail": RETAIL, "big": BIG, "other": OTHER},
           "events": ev, "clusters": cl, "base_rates": base_rates, "n_ctrl_days": n_ctrl, "n_base_days": int(base.sum()),
           "hits": hits, "multiple": {"observed_p05": obs_cnt, "null": null_cnt, "n_cells_per_kind": len(ORDER) * len(conds)},
           "state_cut": {g: v.tolist() for g, v in q.items()}, "state_table": state_tab}
    H.save("events", res)
    write_md(res)
    return res


# ---------- 보고서 ----------
MEAN = {"aaii_spread": "개인 낙관", "lev_qqq_bull5": "TQQQ 쪽 거래 몰림", "lev_soxx_bull5": "SOXL 쪽 거래 몰림",
        "lev_qqq_flow20p": "TQQQ로 돈 들어옴", "eq_pcr5": "풋 많음(겁)", "ai_cpcr5": "AI 종목 풋 많음(겁)",
        "ai_breadth5": "콜 몰린 AI 종목 많음", "nq_nonrept_net_oi": "소액 선물 롱",
        "nq_lev_net_oi": "헤지펀드 NQ 롱", "nq_lev_chg4": "4주간 NQ 롱 늘림", "nq_am_net_oi": "자산운용 NQ 롱",
        "nq_am_chg4": "4주간 NQ 롱 늘림", "es_lev_net_oi": "헤지펀드 ES 숏 작음", "vx_lev_net_oi": "VIX 선물 롱(헤지)",
        "dix": "다크풀 매수 많음(회사 주장)", "gvip_rel20": "헤지펀드 인기주가 QQQ보다 강함", "si_ai_chg": "AI 공매도 잔고 늘어남",
        "qai_beta60": "헤지펀드 복제 베타 큼", "naaim": "액티브 매니저 노출 큼", "skew": "꼬리 풋 비쌈",
        "vix_ts": "단기 공포 큼", "gex": "딜러 양의 감마(잔잔)", "nq_dealer_net_oi": "딜러 NQ 롱",
        "shvr_ai5": "AI 공매도 거래량 비율 큼(주로 시장조성)"}
GRP = {**{k: "개인" for k in RETAIL}, **{k: "큰손" for k in BIG}, **{k: "기타" for k in OTHER}}
NUM = "①②③④⑤⑥⑦⑧"

NARR = {
    ("top", "2025-01-22"): "큰손: 헤지펀드 인기주(GVIP)가 20일간 QQQ보다 +4.0% 강해 0.97(10일 전 0.22)이었고, NQ 레버리지 펀드 순포지션은 10일 전 0.37 → 0.71로 늘었다. VIX 선물 레버리지 펀드는 0.15(변동성 매도 쪽), NAAIM은 0.17로 낮았다. "
                           "개인: AAII 강세−약세 −15.1%p(0.02)로 비관이 강했고 TQQQ·SOXL 거래대금 몫도 0.27·0.33으로 낮았다. 주식 풋/콜만 0.18(콜 쪽). OCC AI 자료는 2025-04-14부터라 없다. "
                           "'개미 환호' 모양이 아니었다. 같은 구성(GVIP ≥0.8 & AAII ≤0.2)은 7묶음 중 4묶음 뒤에 고점이 없었다(그중 하나는 4/7~4/8 바닥).",
    ("top", "2025-02-19"): "큰손: 그날 시가에 보인 CFTC(2/11 기준)에서 NQ 레버리지 펀드는 0.23(20일 전 0.71)으로 롱을 줄이던 중이었다(사후 공개된 2/18 기준은 0.04). VIX 선물 레버리지 0.12(변동성 매도 많음), AI 공매도 잔고는 줄어드는 쪽(0.08), NAAIM 0.29. SKEW는 1.00(꼬리 헤지 수요 1년 최고). "
                           "개인: AAII −18.9%p(0.02), TQQQ−SQQQ 20일 순유입 −22.2억 달러(0.04)로 개인 레버리지 돈이 빠지고 있었다. 개인 묶음 점수 0.26(2025~26 하위 4%), 큰손 0.42(하위 19%). "
                           "둘 다 이미 조심스러운 상태의 고점이었다. 같은 구성(AI 공매도 감소 ≤0.2 & AAII ≤0.2)은 3묶음 중 2묶음이 고점이 아니었다(2025-04-11~28 바닥 직후, 2026-07-28~30 바닥).",
    ("top", "2025-10-29"): "큰손: 정부 셧다운으로 CFTC가 공개를 멈춰 그날 볼 수 있던 선물 포지션이 없다(패널 빈칸, 10/21~11/19). 마지막으로 보인 9/23 기준은 NQ 레버리지 0.65, NQ 자산운용 0.96, VIX 레버리지 0.08. 사후 공개(12/5)된 10/28 기준으로는 NQ 레버리지 펀드가 1년 중 가장 롱(1.00)이었지만 당시엔 알 수 없었다. 볼 수 있던 것: 헤지펀드 인기주가 20일간 QQQ보다 −3.8%(0.04), NAAIM 90.4(0.81), QAI 베타 0.80. "
                           "개인: SOXL 거래대금 몫 1.00(1년 최고), OCC AI 고객 풋/콜 0.05·주식 풋/콜 0.10(콜 쏠림), 개인 묶음 0.75(상위 1%). AAII는 0.50(중립). "
                           "같은 구성(GVIP ≤0.2 & SOXL ≥0.8)은 2025~26에 2묶음(28일)뿐이고 다른 하나가 2026-06-01~05(6/2 QQQ 고점)였다. 사건을 보고 고른 조합이고 2번이라 통계가 아니다.",
    ("top", "2025-12-10"): "그날 보인 CFTC는 셧다운 밀린 공개라 11/4 기준(25거래일 묵음)이었다: NQ 레버리지 0.98, NQ 자산운용 0.79(4주 변화 0.23, 줄이는 중), ES 레버리지 0.15. QAI 베타 0.97, NAAIM 98.6(0.94), AI 공매도 잔고는 늘어나는 쪽(0.83). "
                           "개인: AAII +13.5%p(0.98), SOXL 몫 0.98, TQQQ 순유입 0.80, 주식 풋/콜 0.14. 개인·큰손 모두 위험 선호(상태 r2b2). "
                           "같은 구성(NQ 레버리지 ≥0.8 & SOXL ≥0.8)은 11/20~12/17 한 묶음뿐이었고, 그 안에 바닥(11/20)·고점(12/10)·바닥(12/17)이 다 있었다.",
    ("top", "2026-01-29"): "큰손: NQ 자산운용사 순포지션 0.98(4주 동안 1년 최고 쪽으로 늘림), NQ 레버리지 펀드는 0.35·4주 변화 0.21로 줄이는 쪽이라 둘이 엇갈렸다. ES 레버리지 0.15, VIX 레버리지 0.71(헤지 쪽), QAI 베타 1.00. "
                           "개인: SOXL 몫 0.99, AAII 0.87(10일 내내 0.87~1.00), 주식 풋/콜 0.16. TQQQ 몫은 0.10으로, 나스닥보다 반도체 레버리지에 몰렸다. "
                           "같은 구성(QAI ≥0.8 & SOXL ≥0.8)은 80일 켜져 있었고 한 묶음이 2025-10-29~2026-03-02 넉 달 동안 이어져 고점 4개·바닥 3개를 다 품었다. 그 시기의 평상시 모습이었다.",
    ("top", "2026-02-25"): "큰손: ES 레버리지 펀드 0.04(순 −23.5%, 숏 큼), NQ 자산운용 0.65(20일 전 0.98, 4주 변화 0.19로 줄임), NQ 레버리지 0.56, DIX 0.18, QAI 베타 1.00. "
                           "개인: 이미 겁먹은 쪽이었다. 주식 풋/콜 0.86(5일 전 0.99), OCC AI 풋/콜 0.82, AI 콜 쏠림 폭 0.16, TQQQ 몫 0.01. SOXL 몫만 0.94. 개인 묶음 0.40(하위 26%). "
                           "'환호 끝 고점'이 아니라 '불안 속 고점'이었다. 같은 구성(QAI ≥0.8 & TQQQ 몫 ≤0.2)은 131일·4묶음이었고, 2026-03-30 바닥도 같은 구성이었다.",
    ("top", "2026-06-02"): "큰손: NQ 레버리지 펀드 순 −21.4%(0.04, 1년 중 가장 숏 쪽)였고 20일 내내 0.02~0.15였다. NQ 자산운용 0.58(5일 전 0.94에서 줄임), 헤지펀드 인기주가 QQQ보다 −3.1%(0.12, 20일 최고 0.33), DIX 0.05, AI 공매도 잔고 증가(0.88). 반면 NAAIM은 98.4(0.92). "
                           "개인: 주식 풋/콜 0.00·OCC AI 풋/콜 0.00(1년 중 가장 콜 쪽), AI 콜 쏠림 폭 0.92, SOXL 0.90, TQQQ 순유입 20일 전 0.00 → 0.54. 개인 묶음 0.74(상위 1%), 큰손 0.38(하위 11%)로 사전 등록한 '개미 환호 + 큰손 방어'에 가장 가까운 날이다(엄격 기준 T1은 2026에 0일). "
                           "같은 구성(NQ 레버리지 ≤0.2 & AI 풋/콜 ≤0.2)은 3묶음 중 2묶음(2025-05-19~21, 2026-04-20~22) 뒤에 고점이 없었고 10거래일 뒤 QQQ는 +2.3%·+6.2%였다. 3거래일 뒤 SOXX 바닥(6/5)도 같은 그림이었다.",
    ("top", "2026-06-22"): "큰손: 헤지펀드 인기주가 10일 만에 0.13 → 0.96(QQQ보다 +4.0%). NQ 레버리지 펀드는 아직 숏 쪽(0.21)이지만 4주 변화 0.88로 숏을 줄이는 중, NQ 자산운용은 4주 변화 0.13(줄임), VIX 레버리지 0.75. "
                           "개인: SOXL 몫 0.94, TQQQ 순유입 0.82. 그러나 OCC AI 풋/콜도 0.74(5일 전 0.88)로 풋이 많았다. "
                           "같은 구성(GVIP ≥0.8 & SOXL ≥0.8)은 6묶음 중 3묶음 뒤에 고점이 없었다. 이 하락(−29.0%)은 10/12까지 6/22 종가를 회복하지 못했다.",
    ("bottom", "2025-04-08"): "큰손: NQ 레버리지 펀드 순은 백분위로 아직 0.88이지만 원값은 15일 전 미결제의 −1.2% → −12.6%로 줄었고 4주 변화는 0.10, NQ 자산운용은 0.10이지만 4주 변화 0.87(늘림), NAAIM 49.4(0.02), AI 공매도 잔고 증가(0.92), DIX 0.03, QAI 베타 1.00, GEX 0.01(딜러 음의 감마). "
                              "개인: AAII −40.2%p(0.04), 주식 풋/콜 0.89, TQQQ·SOXL 거래대금 몫 0.01·0.00. 그런데 TQQQ−SQQQ 20일 순유입은 +38.7억 달러(0.98)로, 개인 돈은 내리는 동안 레버리지 상승 ETF로 들어갔다. "
                              "같은 구성(QAI ≥0.8 & SOXL 몫 ≤0.2)은 3묶음 중 1묶음(2025-05-22~06-09)이 바닥과 무관했다.",
    ("bottom", "2025-11-20"): "그날 보인 CFTC는 9/30 기준(37거래일 묵음, 11/19 공개)이라 당시 포지션이 아니다. 볼 수 있던 것: DIX 0.06, QAI 베타 0.97, NAAIM 0.75, GVIP 0.60(5일 전 0.13에서 반등). "
                              "개인: 레버리지 매수(TQQQ 몫 0.97, SOXL 0.92, TQQQ 순유입 +14.8억 달러·0.85)와 풋 매수(주식 풋/콜 0.96, AI 풋/콜 0.77)가 함께 많았다. AAII 0.27. "
                              "같은 구성(NQ 레버리지 ≥0.8 & TQQQ 몫 ≥0.8)은 4묶음 중 3묶음(2025-07~09 상승장)이 바닥과 무관했다.",
    ("bottom", "2025-12-17"): "CFTC는 11/25 기준(15거래일 묵음): NQ 레버리지 0.88(4주 변화 0.19). QAI 베타 0.97, NAAIM 97.1(0.92), SKEW 0.82. "
                              "개인: AAII +14.0%p(1.00, 1년 최고), SOXL 0.89, TQQQ 순유입 0.77. 바닥인데 개인 설문은 1년 중 가장 낙관이었다. "
                              "같은 구성(QAI ≥0.8 & AAII ≥0.8)의 긴 묶음(2025-12-05~2026-02-12)이 고점 2개와 바닥 2개를 함께 품었다.",
    ("bottom", "2026-02-04"): "4거래일 전 고점(1/29)과 그림이 거의 같다. 큰손: NQ 자산운용 1.00, VIX 레버리지 0.92(VIX 선물 롱 쪽, 헤지), QAI 0.98, GVIP 0.88, ES 레버리지 0.13. 개인: AAII 0.94, SOXL 0.92, TQQQ 몫 0.04. "
                              "같은 구성(NQ 자산운용 ≥0.8 & TQQQ 몫 ≤0.2)은 3묶음 중 1묶음(2026-01-12~14)이 10거래일 안에 바닥이 없었다.",
    ("bottom", "2026-03-30"): "큰손: NQ 레버리지 0.12·4주 변화 0.10(줄임), NQ 자산운용 0.33·4주 변화 0.13(줄임), 헤지펀드 인기주가 QQQ보다 −2.3%(0.11), DIX 0.02, NAAIM 0.15, GEX −72억 달러(0.00, 딜러 음의 감마). 큰손 묶음 0.31(하위 3%). "
                              "개인: TQQQ 몫 0.00, OCC AI 풋/콜 0.92, AI 콜 쏠림 폭 0.05, AAII −17.7%p(0.17), 주식 풋/콜 0.48(5일 전 1.00). 그래도 TQQQ 순유입은 0.77. 개인·큰손 모두 방어(r0b0). "
                              "r0b0 상태 56일 중 10거래일 안에 바닥이 온 날은 30%(아무 날 16%). 같은 구성(QAI ≥0.8 & TQQQ 몫 ≤0.2)은 2026-02-25 고점과 같았다.",
    ("bottom", "2026-06-05"): "6/2 고점과 같은 그림이었다. 개인 환호(주식 풋/콜 0.02, AI 풋/콜 0.01, AI 콜 쏠림 폭 0.81, SOXL 0.88)와 큰손 방어(NQ 레버리지 0.04, GVIP 0.13, DIX 0.16, AI 공매도 잔고 증가 0.88). "
                              "이틀 만의 −12.3% 뒤 6/22에 새 고점이 나왔다. '개미 환호 + 큰손 방어' 그림이 고점에도 바닥에도 있었다.",
    ("bottom", "2026-07-29"): "큰손: NQ 레버리지 펀드 순 −29.1%(0.02, 1년 중 가장 숏). VIX 선물 레버리지 펀드는 순 +0.8%로 롱이 됐고(0.96, 20일 내내 0.96~1.00) ES 레버리지 0.92, DIX 39.7%(0.01), 헤지펀드 인기주가 QQQ보다 −3.2%(0.12), AI 공매도 잔고는 줄어드는 쪽(0.17). "
                              "개인: 주식 풋/콜 0.98, OCC AI 풋/콜 0.88, AAII 0.19(5일 전 0.88에서 급락)로 겁먹었다. 그래도 TQQQ 순유입은 +21.9억 달러(0.96). "
                              "같은 구성(DIX ≤0.2 & 주식 풋/콜 ≥0.8)은 9묶음(26일) 중 4묶음 뒤에 바닥이 없었다.",
}


def _f(v, nd=2):
    return "–" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def _cell(sig: dict) -> str:
    d, lo, hi = sig["day"], sig["min10"], sig["max10"]
    core = _f(d)
    if d == d and (d >= HI or d <= LO):
        core = f"**{core}**"
    return f"{core} ({_f(lo)}–{_f(hi)})" if lo == lo else core


def write_md(r: dict) -> None:
    cl, br, hits, mult, st = r["clusters"], r["base_rates"], r["hits"], r["multiple"], r["state_table"]
    tops = [c for c in cl if c["kind"] == "top"]
    bots = [c for c in cl if c["kind"] == "bottom"]
    lab = {("top", c["date"]): f"고점{NUM[i]}" for i, c in enumerate(tops)}
    lab.update({("bottom", c["date"]): f"바닥{NUM[i]}" for i, c in enumerate(bots)})
    nt = {k: sum(1 for c in grp if c["retail_score"]["0"] >= r["state_cut"]["retail"][1]) for k, grp in (("top", tops), ("bottom", bots))}
    L = []
    w = L.append
    w("# 고점·바닥 사건 설명: 옵션·헤지펀드·개인 신호 (갈래 events, 2026-10-14)")
    w("")
    w("설명용이다. 판정·매수 추천은 하지 않는다. 코드 `code/events.py`, 전체 수치 `results/events.json`. 패널 마지막 진입일 " + r["rules"]["panel_last"] + ".")
    w("")
    w("## 결론")
    w("")
    mt, mb = mult["null"]["top|ctrl"], mult["null"]["bottom|ctrl"]
    w(f"- **고점 날의 신호는 '그냥 많이 오른 날'과 구분되지 않았다.** 사건 아닌 20일 종가 고점일({r['n_ctrl_days']['top']}일)과 견줘 우연을 넘는 칸은 96칸 중 "
      f"{mult['observed_p05']['top|ctrl']}칸이었다(대조일에서 무작위로 뽑으면 평균 {mt['mean']:.1f}칸, p {mt['p']:.2f}).")
    w(f"- **개인 환호는 고점에도 바닥에도 있었다.** 고점 {len(tops)}번 중 {nt['top']}번, 바닥 {len(bots)}번 중 {nt['bottom']}번이 개인 환호 점수 상위 삼분위였다. 개인 환호 상위 삼분위 날 뒤 10거래일 안에 고점이 온 비율 "
      f"{st['r2']['top_10d']:.0%}, 바닥 {st['r2']['bottom_10d']:.0%}(아무 날 {st['all']['top_10d']:.0%}·{st['all']['bottom_10d']:.0%}). 방향보다 '큰 움직임이 가깝다'에 가까웠다.")
    w("- **같은 그림이 고점과 바닥에 함께 나온 쌍이 넷이다**: 2026-06-02 고점과 06-05 바닥('개미 환호 + 큰손 방어'), 2025-12-10 고점과 12-17 바닥, 2026-01-29 고점과 02-04 바닥, 2026-02-25 고점과 03-30 바닥.")
    w("- **헤지펀드(NQ 레버리지 펀드)의 방향은 QQQ 고점마다 달랐다**: 2026-06-02엔 1년 중 가장 숏(0.04), 2025-10-29엔 사후 공개 값으로 가장 롱(1.00, 당시엔 셧다운으로 미공개), 2025-02-19엔 롱을 줄이던 중(0.23).")
    hd, hq = hits["bottom|dix|day_lo"], hits["bottom|qai_beta60|day_hi"]
    w(f"- 바닥 날은 다른 20일 저점일({r['n_ctrl_days']['bottom']}일)보다 조금 더 달랐다({mult['observed_p05']['bottom|ctrl']}칸, 우연 평균 {mb['mean']:.1f}, p {mb['p']:.2f}). "
      f"주로 DIX ≤0.2(바닥 {hd['k']}/{hd['n']} 대 대조 {hd['ctrl']:.0%})와 QAI 베타 ≥0.8({hq['k']}/{hq['n']} 대 {hq['ctrl']:.0%})인데, 둘 다 매도가 길어질수록 같이 움직이는 값이고 바닥인지는 지나고서야 안다. 비교 4번(고점·바닥 × 아무 날·대조) 중 하나라 약하다.")
    w("")
    w("## 사건 (종가 기준 −7% 이상 하락 직전 고점과 그 바닥, 2025-01-02부터)")
    w("")
    w("| 사건 | 날짜 | 자산 | 낙폭 | 짝 | 사이 거래일 | 그 전 20일 QQQ / SOXX |")
    w("|---|---|---|---|---|---|---|")
    for c in cl:
        w(f"| {lab[(c['kind'], c['date'])]} | {c['date']} | {', '.join(c['members'])} | {c['dd']}% | {c['pair']} | {c['span_td']} | "
          f"{c['ret20_into']['QQQ']:+.1f}% / {c['ret20_into']['SOXX']:+.1f}% |")
    w("")
    w("- 3거래일 안에 겹친 같은 종류 사건은 하나로 셌다(2026-06-02 QQQ·06-03 SOXX). 낙폭·짝은 대표 자산 기준. SOXX 2026-06-22 → 07-29 하락은 10/12까지 회복 전이라 바닥이 바뀔 수 있다.")
    w("- 숫자는 지난 1년(일별 250, 주간 52, 월 2회 24관측) 안 백분위다. 1이면 1년 중 가장 높다. '높으면' 뜻은 2절 표에 적었다.")
    w("- 사건일 값 = 그날 **시가 전에 공개된** 값. 괄호 = 사건일 포함 10거래일(d−9…d)의 최저–최고. **굵게** = 0.8 이상 또는 0.2 이하.")
    w("")
    w("### 언제 알 수 있었나(harness 규칙)")
    w("")
    w("| 자료 | 패널에 들어가는 날(그날 시가) |")
    w("|---|---|")
    w("| CBOE 주식 풋/콜, OCC AI 고객 풋/콜, DIX·GEX, FINRA 공매도 거래량, ProShares 흐름 | 측정일 저녁 공개 → 다음 거래일 |")
    w("| TQQQ·SOXL 거래대금 몫, GVIP/QQQ, QAI 베타, SKEW, VIX/VIX3M | 그날 종가 → 다음 거래일 |")
    w("| CFTC(화요일 포지션, 금 15:30 ET 공개) | 공개 다음 거래일. 기준일보다 보통 4~8거래일 늦다. **2025-10-21~11-19 빈칸(셧다운), 11-19~2026-01-05엔 밀린 14주가 몰려 공개돼 그 사이 값은 최대 37거래일 묵었다**(패널의 비우기 규칙은 공개일 기준이라 이 묵음을 거르지 않는다) |")
    w("| AAII·NAAIM(주간, 목 공개) | 다음 거래일. NAAIM은 2026-07-29 측정분까지만(유료 전환) |")
    w("| AI 공매도 잔고(월 2회) | 기준일 + 약 8거래일 공개 → 다음 거래일 |")
    w("| OCC AI 풋/콜·AI 콜 쏠림 폭 | 2025-04-14부터(그 전 사건은 빈칸) |")
    w("")
    w("## 1. 묶음 점수: 개인 환호 vs 큰손 위험 선호")
    w("")
    w("개인 환호 = 개인 8개, 큰손 위험 선호 = 큰손 11개를 방향 맞춰 평균(정의는 결과 전 고정, 코드 `RETAIL`·`BIG`). 괄호는 2025-01~2026-10 안 순위. 큰손은 CFTC 6개와 그 밖 5개로도 나눴다. "
      f"상태: r = 개인, b = 큰손, 0/1/2 = 2025~26 삼분위(경계 개인 {r['state_cut']['retail'][0]:.2f}·{r['state_cut']['retail'][1]:.2f}, 큰손 {r['state_cut']['big'][0]:.2f}·{r['state_cut']['big'][1]:.2f}).")
    w("")
    w("| 사건 | 개인 d−20 → d−10 → d (순위) | 큰손 d−20 → d−10 → d (순위) | 큰손 d: CFTC / 그 밖 | 상태 |")
    w("|---|---|---|---|---|")
    for c in cl:
        rs, bs = c["retail_score"], c["big_score"]
        brank = _f(c["big_score_pct"]) if bs["0"] == bs["0"] else "–"
        w(f"| {lab[(c['kind'], c['date'])]} {c['date']} | {_f(rs['-20'])} → {_f(rs['-10'])} → {_f(rs['0'])} ({c['retail_score_pct']:.2f}) | "
          f"{_f(bs['-20'])} → {_f(bs['-10'])} → {_f(bs['0'])} ({brank}) | "
          f"{_f(c['big_cftc_score']['0'])} / {_f(c['big_other_score']['0'])} | {c['state'] or '–(CFTC 빈칸)'} |")
    w("")
    w("상태별로 그날 뒤 10거래일 안(그날 포함)에 고점·바닥이 온 비율(2025-01-02~, 두 점수가 다 있는 날):")
    w("")
    w("| 개인 \\ 큰손 | 큰손 하 | 큰손 중 | 큰손 상 |")
    w("|---|---|---|---|")
    for r_, nm in ((2, "개인 상"), (1, "개인 중"), (0, "개인 하")):
        row = [f"{st[f'r{r_}b{b_}']['n_days']}일 · 고점 {st[f'r{r_}b{b_}']['top_10d']:.0%} · 바닥 {st[f'r{r_}b{b_}']['bottom_10d']:.0%}" for b_ in range(3)]
        w(f"| {nm} | " + " | ".join(row) + " |")
    w(f"| 전체 | {st['all']['n_days']}일 · 고점 {st['all']['top_10d']:.0%} · 바닥 {st['all']['bottom_10d']:.0%} | | |")
    w("")
    w("## 2. 사건 날 꼬리 빈도 vs 기준 빈도")
    w("")
    w(f"기준: 아무 날 = 2025-01-02~ {r['n_base_days']}일. 대조 = 사건 아닌 20일 종가 고점일(같은 자산 사건 ±5거래일 밖) {r['n_ctrl_days']['top']}일 / 20일 종가 저점일 {r['n_ctrl_days']['bottom']}일. "
      "칸은 ≥0.8 비율 / ≤0.2 비율. 사건은 k/n(n = 값이 있던 사건 수). * = 아무 날 대비 이항 p < 0.05, † = 대조 대비 p < 0.05(여러 칸 비교 보정 전).")
    for kind, nm, ctrl_nm in (("top", "고점", "대조(고점일)"), ("bottom", "바닥", "대조(저점일)")):
        nr, nc = mult["null"][f"{kind}|random"], mult["null"][f"{kind}|ctrl"]
        w("")
        w(f"**{nm}** — 96칸 중 p<0.05: 아무 날 대비 {mult['observed_p05'][f'{kind}|random']}칸(무작위 날짜 {N_NULL}번 평균 {nr['mean']:.1f}, 95% {nr['p95']:.0f}, p {nr['p']:.2f}), "
          f"대조 대비 {mult['observed_p05'][f'{kind}|ctrl']}칸(대조일에서 뽑은 평균 {nc['mean']:.1f}, 95% {nc['p95']:.0f}, p {nc['p']:.2f})")
        w("")
        w(f"| 신호 | 높으면 | 아무 날 | 아무 10일 창 | {nm} 당일 | {nm} 10일 창 | {ctrl_nm} 당일 | {ctrl_nm} 10일 창 |")
        w("|---|---|---|---|---|---|---|---|")
        cb = br[f"ctrl_{kind}"]
        for s in ORDER:
            def kk(c, s=s):
                h = hits[f"{kind}|{s}|{c}"]
                mk = ("*" if h["p_random"] < 0.05 else "") + ("†" if h["p_ctrl"] < 0.05 else "")
                return f"{h['k']}/{h['n']}{mk}"
            w(f"| {KO[s]} ({GRP[s]}) | {MEAN[s]} | {br['random'][s]['day_hi']:.2f} / {br['random'][s]['day_lo']:.2f} | {br['random'][s]['win_hi']:.2f} / {br['random'][s]['win_lo']:.2f} | "
              f"{kk('day_hi')} / {kk('day_lo')} | {kk('win_hi')} / {kk('win_lo')} | {cb[s]['day_hi']:.2f} / {cb[s]['day_lo']:.2f} | {cb[s]['win_hi']:.2f} / {cb[s]['win_lo']:.2f} |")
    for kind, group, nm in (("top", tops, "고점"), ("bottom", bots, "바닥")):
        w("")
        w(f"## 3. {nm}별 24개 신호: 사건일 (10일 최저–최고)")
        w("")
        w("| 신호 | " + " | ".join(f"{lab[(kind, c['date'])]} {c['date'][2:]}" for c in group) + " |")
        w("|---|" + "---|" * len(group))
        for s in ORDER:
            w(f"| {KO[s]} | " + " | ".join(_cell(c["signals"][s]) for c in group) + " |")
        cells = []
        for c in group:
            a = c["raw"].get("cftc_asof")
            cells.append(f"{a}({c['raw']['cftc_age_td']})" if isinstance(a, str) else "없음")
        w("| CFTC 기준일(묵은 거래일) | " + " | ".join(cells) + " |")
    w("")
    w("## 4. 같은 구성은 사건 없이도 나왔나")
    w("")
    w("사건일에 가장 치우친 큰손 신호 1개와 개인 신호 1개를 골라(사건을 보고 고른 것), 둘이 같은 꼬리(≥0.8 / ≤0.2)에 함께 있던 날을 2025-01-02~에서 모두 찾았다. "
      "5거래일 넘게 끊기면 새 묶음. '있음' = 묶음 시작~끝+10거래일 안에 같은 종류 사건(QQQ·SOXX 아무거나)이 있었다. 사건 자신이 든 묶음도 '있음'에 들어간다.")
    w("")
    w("| 사건 | 구성(사건일 값) | 일수 | 묶음 | 있음 | 없음 | 없던 묶음(끝난 뒤 10거래일 QQQ / SOXX) |")
    w("|---|---|---|---|---|---|---|")
    for c in cl:
        g = c["config"]
        miss = "; ".join(f"{e['start']}~{e['end'][5:]} ({_f(e['QQQ_fwd10'], 1)}% / {_f(e['SOXX_fwd10'], 1)}%)" for e in g["episodes"] if not e["event_10d"]) or "–"
        rule = g["rule"].replace(">=", " ≥").replace("<=", " ≤")
        w(f"| {lab[(c['kind'], c['date'])]} {c['date']} | {rule} ({g['big_v']:.2f} / {g['retail_v']:.2f}) | {g['n_days']} | {g['n_episodes']} | "
          f"{g['n_episodes_with_event_10d']} | {g['n_episodes_without']} | {miss} |")
    w("")
    w(f"참고: 아무 날 뒤 10거래일 안에 고점이 올 확률 {st['all']['top_10d']:.0%}, 바닥 {st['all']['bottom_10d']:.0%}. 여러 날 이어진 묶음은 이보다 '있음'이 나오기 쉽다.")
    w("")
    w("## 5. 사건별 서술 (큰손 vs 개인, 사건 전 몇 주)")
    w("")
    w("숫자는 백분위(0~1) 또는 원값. '사후'라고 적은 값은 그날엔 공개 전이었다.")
    for c in cl:
        w("")
        w(f"**{lab[(c['kind'], c['date'])]} {c['date']} {c['assets']}** ({c['dd']}%, 짝 {c['pair']})  ")
        w(NARR[(c["kind"], c["date"])])
    w("")
    w("## 6. 한계")
    w("")
    w("- 사건 수가 고점 8·바닥 7(겹침 합침)이다. 2절의 이항 p는 사건이 서로 독립이라는 가정이고, 신호끼리·날끼리 상관이 커서 보정 전 값이다. 고점·바닥 각각 96칸(24신호 × 4조건)을 봤다.")
    w("- 사건 날짜는 사후에 정한 날이다. 그날 시가 정보만 썼지만 '그날이 고점'이라는 사실은 그때 알 수 없다. 4절의 구성도 사건을 보고 고른 것이다.")
    w("- 많은 신호가 가격과 같이 움직인다(오른 뒤 콜 많음·레버리지 몰림, 내린 뒤 DIX 낮음·풋 많음·VIX 역전·60일 베타 큼). 그래서 '아무 날'과 함께 '20일 고점·저점일'과도 비교했다.")
    w("- CFTC: 2025-10-21~11-19 빈칸, 2025-11-20~2026-01-05 값은 기준일이 3~7주 늦다(고점④·바닥②③의 CFTC 칸). 레버리지 펀드에는 CTA·상대가치 거래도 섞인다. ES 레버리지 순은 구조적으로 크게 음수라 수준보다 백분위로 읽는다.")
    w("- DIX와 AI 공매도 거래량 비율은 FINRA 공매도 거래량에서 나오며 대부분 시장조성자 몫이다. 'DIX 높음 = 기관 매수'는 회사 주장이다.")
    w("- OCC AI 자료는 2025-04-14부터라 고점①②·바닥①에 없다. NAAIM은 2026-07-29 측정분 뒤 없다. 13F는 받지 못했다(SEC 403).")
    w("- 이 설명은 2026 앞(1~5월) → 뒤(6~10월) 확인 판정(`grid.json`: 고점 쪽 확인 3칸 대 우연 평균 3.5, 바닥 쪽 1칸 대 3.1)을 바꾸지 않는다.")
    (H.RES / "events.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r = main()
    for c in r["clusters"]:
        print(c["kind"], c["date"], c["members"], c["dd"], "state", c["state"], "re", round(c["retail_score"]["0"], 2),
              "big", round(c["big_score"]["0"], 2), "cfg", c["config"]["rule"], c["config"]["n_episodes"],
              c["config"]["n_episodes_with_event_10d"])
    print(r["multiple"])
    print(r["n_ctrl_days"], r["n_base_days"])
