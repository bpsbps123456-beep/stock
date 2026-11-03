"""SEIBRO 바닥 연구 — 상품 갈래: SOXX 바닥 전후 한국인이 사고 판 상품(설명용, 사후 표지).

실행(연구 폴더에서): py code/products.py
읽는 것: derived/flows_trade_product.parquet·measures_daily.parquet·price_feats.parquet·events.json(고치지 않음),
        ../seibro_flows/raw/daily/d1~d4(50위 문턱·이름, 읽기만).
출력(results/products/): top_lists.csv, recurrence.csv, rotation.csv, rotation_events.csv, run.json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True
STUDY = Path(__file__).resolve().parents[1]
PRIOR = STUDY.parent / "seibro_flows"
DER = STUDY / "derived"
OUT = STUDY / "results" / "products"

WINS = {"W1": (-20, -6), "W2": (-5, -1), "W3": (0, 0), "W4": (1, 5), "W5": (6, 20)}
TOP_WINS = ["decline", "W2", "W3", "W4", "W5"]  # 순위 표·반복 표
ROT_WINS = ["decline", "W1", "W2", "W3", "W4", "W5"]  # 범주 회전 표
TOPN, RECN = 10, 5
REC_MIN, BASE_MIN = 5, 0.20  # 반복 = B15 6개 중 5개 이상 / 후보에 넣을 평소 상위 5 비율
# 회전 범주: 겹치지 않게 위에서부터 먼저 맞는 범주에 넣는다
CATS = [
    ("SOXL", "SOXL"),
    ("EQ3L_oth", "기타 3배 롱(지수)"),
    ("L3_single", "단일종목 레버리지 롱(L3)"),
    ("LEV_oth", "기타 레버리지 롱(2배 지수 등)"),
    ("H4_inv", "주식 인버스(H4)"),
    ("R5_inc", "인컴·커버드콜(R5)"),
    ("R1_semi", "1배 반도체(R1)"),
    ("R2_idx", "1배 지수 ETF(R2)"),
    ("R4_stock", "그 밖 1배 개별주(R4)"),
    ("R3_bond", "채권·현금(R3)"),
    ("OTH", "기타(원자재·코인·VIX·FX·OTHER·인버스 채권)"),
]
CKEYS = [c[0] for c in CATS]
CLAB = dict(CATS)


def iso(d) -> str | None:
    return None if d is None or pd.isna(d) else pd.Timestamp(d).date().isoformat()


def category(r) -> str:
    if r["product"] == "SOXL":
        return "SOXL"
    if r["LSGL"]:
        return "L3_single"
    if r["EQ3L"]:
        return "EQ3L_oth"
    if r["lev_long"]:
        return "LEV_oth"
    if r["EQ_INV"]:  # DIPS·FIAT(인컴 숏)은 H4 정의대로 인버스
        return "H4_inv"
    if r["R5"]:  # 반도체 커버드콜(NVDY·NVYY·CHPY)은 R1과 겹쳐 R5로만 센다
        return "R5_inc"
    if r["R1"]:
        return "R1_semi"
    if r["R2"]:
        return "R2_idx"
    if r["R4"]:
        return "R4_stock"
    if r["R3"]:
        return "R3_bond"
    return "OTH"


# ---------- 1. 자료 ----------
def load_raw_lists():
    """SEIBRO 날짜별 50위 문턱과 ISIN 이름. 목록 줄 수 < 50이면 그 목록 문턱 = 0(값 있는 상품이 다 나옴)."""
    th, names = {}, {}
    for d in "1234":
        for p in sorted((PRIOR / "raw" / "daily" / f"d{d}").glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            s = pd.Timestamp(rec["start"])
            rows = rec["rows"]
            for x in rows:
                names[x["isin"]] = x["name"]
            last = rows[-1] if len(rows) >= 50 else None
            key = {"1": "buy", "2": "sell", "3": "tot", "4": "net"}[d]
            th.setdefault(s, {})[f"{key}50"] = float(last[key]) if last else 0.0
    t = pd.DataFrame.from_dict(th, orient="index").sort_index()
    t = t.fillna(0.0)
    # 목록 밖 상품: net ≤ min(net50, buy50, tot50), −net ≤ min(sell50, tot50)
    t["hi"] = t[["net50", "buy50", "tot50"]].min(axis=1).clip(lower=0)
    t["lo"] = t[["sell50", "tot50"]].min(axis=1).clip(lower=0)
    return t, names


def load():
    f = pd.read_parquet(DER / "flows_trade_product.parquet")
    f = f[~f["dummy"].fillna(False).astype(bool)].copy()
    md = pd.read_parquet(DER / "measures_daily.parquet").set_index("date").sort_index()
    pf = pd.read_parquet(DER / "price_feats.parquet").set_index("date").sort_index()
    units = pd.read_parquet(DER / "units.parquet")
    ev = json.loads((DER / "events.json").read_text(encoding="utf-8"))
    return f, md, pf, units, ev


def build_mats(f, md, units, th):
    dates = md.index
    prods = sorted(f["product"].unique())
    pidx = {p: i for i, p in enumerate(prods)}
    di = dates.get_indexer(f["date"])
    if (di < 0).any():
        raise SystemExit("흐름 날짜가 측정 날짜표 밖")
    pi = f["product"].map(pidx).to_numpy()
    shape = (len(dates), len(prods))
    mats = {}
    for col in ("net", "buy", "sell"):
        m = np.zeros(shape)
        np.add.at(m, (di, pi), f[col].to_numpy(dtype=float))
        mats[col] = m
    listed = np.zeros(shape, bool)
    listed[di, pi] = f["in_any_list"].to_numpy(bool)
    full = np.zeros(shape, bool)
    full[di, pi] = f["full_obs"].to_numpy(bool) & f["in_any_list"].to_numpy(bool)
    ipo = np.zeros(shape, bool)
    ipo[di, pi] = f["ipo_excl"].to_numpy(bool)
    # 거래일별 숨을 수 있는 최대 순매수(hi)·순매도(lo): 묶음 SEIBRO 날짜 문턱 합 ÷ 묶음 거래일 수
    u = units[units["unit_n_trades"] > 0][["seibro_date", "unit_id", "unit_n_trades"]].copy()
    allu = units[["seibro_date", "unit_id"]].merge(th[["hi", "lo"]], left_on="seibro_date", right_index=True,
                                                   how="left").fillna({"hi": 0.0, "lo": 0.0})
    us = allu.groupby("unit_id")[["hi", "lo"]].sum()
    ntr = u.groupby("unit_id")["unit_n_trades"].first()
    us = us.loc[ntr.index].div(ntr, axis=0)
    hi = md["unit_id"].map(us["hi"]).to_numpy(float)
    lo = md["unit_id"].map(us["lo"]).to_numpy(float)
    if np.isnan(hi).any() or np.isnan(lo).any():
        raise SystemExit("문턱 대응 실패")
    pt = f.sort_values("date").drop_duplicates("product", keep="last").set_index("product")
    flags = ["utype", "L", "sign", "semi", "single", "equity", "lev_long", "EQ3L", "IDX", "EQ_INV", "spot", "LSGL",
             "LIDX", "R1", "R2", "R3", "R4", "R5"]
    pt = pt.loc[prods, flags].copy()
    for c in flags[3:]:
        pt[c] = pt[c].fillna(False).astype(bool)
    pt["product"] = pt.index
    pt["cat"] = pt.apply(category, axis=1)
    return dates, prods, mats, listed, full, ipo, hi, lo, pt


def product_names(f, names, prods):
    big = f.sort_values("tot").drop_duplicates("product", keep="last").set_index("product")["isins"]
    out = {}
    for p in prods:
        isins = [x for x in str(big.get(p, "")).split(";") if x]
        out[p] = (names.get(isins[0], "") if isins else "")[:48]
    return out


# ---------- 2. 창 ----------
def windows(ev, sess, fdates):
    """사건·F 날의 창(흐름 행 번호 a~b, 양끝 포함). 흐름 범위 밖이 끼면 None."""
    spos = {d: i for i, d in enumerate(sess)}
    f0, f1 = spos[fdates[0]], spos[fdates[-1]]

    def mk(k0, top):
        out = {}
        for w, (lo, hi) in WINS.items():
            a, b = k0 + lo, k0 + hi
            out[w] = (a - f0, b - f0) if (a >= f0 and b <= f1) else None
        a, b = spos[top] + 1, k0
        out["decline"] = (a - f0, b - f0) if (a >= f0 and b <= f1) else None
        return out

    evs = []
    for e in ev["B10_events"]:
        d = pd.Timestamp(e["bottom"])
        evs.append({"kind": "B10", "date": d, "B15": bool(e["B15"]), "top": pd.Timestamp(e["top10"]),
                    "episode": e["bottom"], "win": mk(spos[d], pd.Timestamp(e["top10"]))})
    ep_start = {x["episode"]: pd.Timestamp(x["start"]) for x in ev["episodes"]}
    fds = []
    for x in ev["F"]:
        d = pd.Timestamp(x["date"])
        top = ep_start[x["episode"]]
        fds.append({"kind": "F", "date": d, "B15": False, "top": top, "episode": x["episode"],
                    "win": mk(spos[d], top)})
    return evs, fds


def wlen(ab):
    return ab[1] - ab[0] + 1


# ---------- 3. 평소(같은 해 같은 길이 굴린 창) ----------
class Base:
    """길이 L 굴린 창(끝 = 흐름 행 L−1 … 끝)의 상품 상위 5·범주 값. 해(끝 날짜)마다 모은다.
    가격 맞춤: 같은 해 같은 길이 창을 SOXX 창 수익 5분위로 나눠 같은 분위 창끼리 본다(메아리 대조)."""

    def __init__(self, net, catm, a1, years, cl):
        self.net, self.catm, self.a1, self.years = net, catm, a1, years
        self.cl = cl  # 길이 n+1: cl[0] = 흐름 첫날 앞 세션 종가, cl[i+1] = 흐름 i행 종가
        self.cn = np.vstack([np.zeros((1, net.shape[1])), np.cumsum(net, 0)])
        self.cc = np.vstack([np.zeros((1, catm.shape[1])), np.cumsum(catm, 0)])
        self.ca = np.concatenate([[0.0], np.cumsum(a1)])
        self.cache = {}

    def get(self, L):
        if L in self.cache:
            return self.cache[L]
        n = self.net.shape[0]
        ends = np.arange(L - 1, n)
        S = self.cn[ends + 1] - self.cn[ends + 1 - L]
        k = RECN
        ib = np.argpartition(-S, k, axis=1)[:, :k]
        vb = np.take_along_axis(S, ib, 1) > 0
        isl = np.argpartition(S, k, axis=1)[:, :k]
        vs = np.take_along_axis(S, isl, 1) < 0
        yrs = self.years[ends]
        ret = self.cl[ends + 1] / self.cl[ends + 1 - L] - 1
        q = np.full(len(ends), -1)
        pi, pq = {}, {}
        for y in np.unique(yrs):
            m = yrs == y
            nb = np.bincount(ib[m][vb[m]], minlength=S.shape[1]) / m.sum()
            ns = np.bincount(isl[m][vs[m]], minlength=S.shape[1]) / m.sum()
            pi[int(y)] = {"buy": nb, "sell": ns, "n": int(m.sum())}
            edges = np.quantile(ret[m], [0.2, 0.4, 0.6, 0.8])
            q[m] = np.searchsorted(edges, ret[m], side="right")
            for qq in range(5):
                mq = m & (q == qq)
                pq[(int(y), qq)] = {
                    "buy": np.bincount(ib[mq][vb[mq]], minlength=S.shape[1]) / max(mq.sum(), 1),
                    "sell": np.bincount(isl[mq][vs[mq]], minlength=S.shape[1]) / max(mq.sum(), 1),
                    "n": int(mq.sum())}
        C = self.cc[ends + 1] - self.cc[ends + 1 - L]
        A = self.ca[ends + 1] - self.ca[ends + 1 - L]
        pos, neg = np.clip(S, 0, None).sum(1), np.clip(S, None, 0).sum(1)
        rot = rot_values(C, A, pos, neg)
        self.cache[L] = {"pi": pi, "pq": pq, "q": q, "ret": ret, "years": yrs, "rot": rot}
        return self.cache[L]

    def at(self, ab):
        """창 (a, b)의 해·분위와 평소 비율(같은 해, 같은 해·같은 분위)."""
        L = wlen(ab)
        bl = self.get(L)
        i = ab[1] - (L - 1)
        y, qq = int(bl["years"][i]), int(bl["q"][i])
        return bl, i, bl["pi"][y], bl["pq"][(y, qq)], qq


def rot_values(C, A, pos, neg):
    """범주 값: 몫 = 범주 순매수 ÷ Σ|범주 순매수|, 거래 대비 = 순매수 ÷ A1. ALL = 전체 순매수 ÷ A1, FUND = |Σ음| ÷ Σ양(상품)."""
    C = np.atleast_2d(C)
    absum = np.abs(C).sum(1, keepdims=True)
    share = np.where(absum > 0, C / np.where(absum > 0, absum, 1), np.nan)
    A = np.atleast_1d(A).astype(float)
    turn = C / A[:, None]
    allnet = C.sum(1) / A
    fund = np.abs(np.atleast_1d(neg)) / np.atleast_1d(pos)
    return {"share": share, "turn": turn, "net": C, "all_turn": allnet, "fund": fund}


def pct_of(x, ref):
    ref = np.sort(ref[np.isfinite(ref)])
    if not np.isfinite(x) or len(ref) == 0:
        return np.nan
    lt = np.searchsorted(ref, x, "left")
    le = np.searchsorted(ref, x, "right")
    return 100.0 * (lt + 0.5 * (le - lt)) / len(ref)


def pb_tail(ps, k):
    """포아송 이항 P(X ≥ k)."""
    dp = np.zeros(len(ps) + 1)
    dp[0] = 1.0
    for p in ps:
        dp[1:] = dp[1:] * (1 - p) + dp[:-1] * p
        dp[0] *= 1 - p
    return float(dp[k:].sum())


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(len(pv))
    qq[o] = np.minimum(r, 1.0)
    q[ok] = qq
    return q


# ---------- 4. 순위 표 ----------
def window_sums(mats, listed, full, ipo, hi, lo, ab):
    a, b = ab
    sl = slice(a, b + 1)
    s = mats["net"][sl].sum(0)
    nfull = (~full[sl])
    return {"net": s, "buy": mats["buy"][sl].sum(0), "sell": mats["sell"][sl].sum(0),
            "n_listed": listed[sl].sum(0), "hid_hi": (nfull * hi[sl, None]).sum(0),
            "hid_lo": (nfull * lo[sl, None]).sum(0), "ipo": ipo[sl].any(0),
            "never_hi": hi[sl].sum(), "never_lo": lo[sl].sum()}


def ranks(s):
    """순매수 순위(양수만, 1 = 가장 큼)·순매도 순위(음수만, 1 = 가장 작음)."""
    order_b = np.argsort(-s, kind="stable")
    rb = np.empty(len(s))
    rb[order_b] = np.arange(1, len(s) + 1)
    rb[s <= 0] = np.nan
    order_s = np.argsort(s, kind="stable")
    rs = np.empty(len(s))
    rs[order_s] = np.arange(1, len(s) + 1)
    rs[s >= 0] = np.nan
    return rb, rs


def safe_k(ws, k, side):
    """절단에 안전한 상위 k인가: 상위 k의 하한 최솟값 ≥ 나머지(목록에 한 번도 없던 상품 포함)의 상한 최댓값."""
    s = ws["net"]
    if side == "buy":
        o = np.argsort(-s)[:k]
        if (s[o] <= 0).any():
            return False
        lo_top = (s[o] - ws["hid_lo"][o]).min()
        rest = np.ones(len(s), bool)
        rest[o] = False
        hi_rest = max((s[rest] + ws["hid_hi"][rest]).max(), ws["never_hi"])
        return bool(lo_top >= hi_rest)
    o = np.argsort(s)[:k]
    if (s[o] >= 0).any():
        return False
    hi_top = (s[o] + ws["hid_hi"][o]).max()
    rest = np.ones(len(s), bool)
    rest[o] = False
    lo_rest = min((s[rest] - ws["hid_lo"][rest]).min(), -ws["never_lo"])
    return bool(hi_top <= lo_rest)


def wret(c, fdates, ab):
    """창 가격 수익: 창 끝 종가 ÷ 창 시작 전 세션 종가 − 1."""
    a, b = fdates[ab[0]], fdates[ab[1]]
    i = c.index.get_loc(a)
    return float(c.loc[b] / c.iloc[i - 1] - 1)


def soxx_ret(pf, fdates, ab):
    return wret(pf["SOXX_Close"], fdates, ab)


def top_lists(evs, mats, listed, full, ipo, hi, lo, pt, pname, prods, fdates, pf, merged):
    rows, safe = [], []
    for e in evs:
        for w in TOP_WINS:
            ab = e["win"][w]
            if ab is None:
                continue
            ws = window_sums(mats, listed, full, ipo, hi, lo, ab)
            s = ws["net"]
            P, M = s[s > 0].sum(), s[s < 0].sum()
            rb, rs = ranks(s)
            base = {"event": iso(e["date"]), "B15": e["B15"], "year": e["date"].year, "window": w,
                    "start": iso(fdates[ab[0]]), "end": iso(fdates[ab[1]]), "n_sessions": wlen(ab),
                    "n_merged_days": int(merged[ab[0]:ab[1] + 1].sum()), "soxx_ret": round(soxx_ret(pf, fdates, ab), 4),
                    "pos_total_usd": round(P), "neg_total_usd": round(M),
                    "never_listed_max_buy_usd": round(ws["never_hi"]), "never_listed_max_sell_usd": round(ws["never_lo"])}
            for side, order in (("buy", np.argsort(-s)), ("sell", np.argsort(s))):
                sf = {k: safe_k(ws, k, side) for k in (RECN, TOPN)}
                safe.append({"event": base["event"], "B15": e["B15"], "window": w, "side": side,
                             "safe5": sf[RECN], "safe10": sf[TOPN]})
                for j in order[:TOPN]:
                    v = s[j]
                    if (side == "buy" and v <= 0) or (side == "sell" and v >= 0):
                        break
                    p = prods[j]
                    rows.append({**base, "side": side, "rank": int(rb[j] if side == "buy" else rs[j]), "product": p,
                                 "name": pname.get(p, ""), "category": pt.at[p, "cat"], "utype": pt.at[p, "utype"],
                                 "L": pt.at[p, "L"], "sign": int(pt.at[p, "sign"]), "net_usd": round(v),
                                 "share": round(abs(v) / (P if side == "buy" else -M), 4),  # 순매도 몫도 양수
                                 "buy_usd": round(ws["buy"][j]), "sell_usd": round(ws["sell"][j]),
                                 "n_listed_days": int(ws["n_listed"][j]),
                                 "hidden_max_buy_usd": round(ws["hid_hi"][j]),
                                 "hidden_max_sell_usd": round(ws["hid_lo"][j]),
                                 "ipo_28d": bool(ws["ipo"][j]), "list_safe5": sf[RECN], "list_safe10": sf[TOPN]})
    return pd.DataFrame(rows), pd.DataFrame(safe)


# ---------- 5. 반복 표 ----------
def top5_sets(net, ab):
    s = net[ab[0]:ab[1] + 1].sum(0)
    rb, rs = ranks(s)
    return {"buy": set(np.where(rb <= RECN)[0]), "sell": set(np.where(rs <= RECN)[0])}, rb, rs


def recurrence(evs, fds, net, base, prods, pt, pname, fdates):
    rows = []
    for w in TOP_WINS:
        einfo, finfo = [], []
        for grp, items, dst in (("e", evs, einfo), ("f", fds, finfo)):
            for x in items:
                ab = x["win"][w]
                if ab is None:
                    continue
                sets, rb, rs = top5_sets(net, ab)
                _, _, pi, pq, _ = base.at(ab)
                dst.append((x, sets, {"buy": rb, "sell": rs}, pi, pq))
        for side in ("buy", "sell"):
            cand = set()
            for _, sets, *_ in einfo + finfo:
                cand |= sets[side]
            pim = np.mean([pi[side] for *_, pi, _ in einfo], axis=0)
            cand |= set(np.where(pim >= BASE_MIN)[0])
            for j in sorted(cand):
                r = {"window": w, "side": side, "product": prods[j], "name": pname.get(prods[j], ""),
                     "category": pt.at[prods[j], "cat"]}
                for grp, sel in (("B15", lambda e: e["B15"]), ("B10", lambda e: True)):
                    sub = [t for t in einfo if sel(t[0])]
                    hit = [j in t[1][side] for t in sub]
                    ps = [float(t[3][side][j]) for t in sub]
                    pqs = [float(t[4][side][j]) for t in sub]
                    k = int(sum(hit))
                    r[f"n_{grp}"] = len(sub)
                    r[f"hit_{grp}"] = k
                    r[f"base_rate_{grp}"] = round(float(np.mean(ps)), 4)
                    r[f"base_exp_{grp}"] = round(float(np.sum(ps)), 3)
                    r[f"p_{grp}"] = pb_tail(ps, k) if k > 0 else 1.0
                    r[f"basepx_rate_{grp}"] = round(float(np.mean(pqs)), 4)
                    r[f"basepx_exp_{grp}"] = round(float(np.sum(pqs)), 3)
                    r[f"ppx_{grp}"] = pb_tail(pqs, k) if k > 0 else 1.0
                    if grp == "B15":
                        r["B15_events_hit"] = ";".join(iso(t[0]["date"]) for t, h in zip(sub, hit) if h)
                        rk = [t[2][side][j] for t in sub]
                        r["B15_median_rank"] = float(np.nanmedian(rk)) if np.isfinite(rk).any() else np.nan
                        r["B15_ranks"] = ";".join("" if not np.isfinite(x) else str(int(x)) for x in rk)
                fh = [j in t[1][side] for t in finfo]
                r["n_F"] = len(finfo)
                r["hit_F"] = int(sum(fh))
                r["F_rate"] = round(float(np.mean(fh)), 4) if fh else np.nan
                r["F_base_rate"] = round(float(np.mean([t[3][side][j] for t in finfo])), 4) if fh else np.nan
                r["F_basepx_rate"] = round(float(np.mean([t[4][side][j] for t in finfo])), 4) if fh else np.nan
                eps = {}
                for t, h in zip(finfo, fh):
                    eps.setdefault(t[0]["episode"], []).append(h)
                r["n_F_episodes"] = len(eps)
                r["F_rate_ep"] = round(float(np.mean([np.mean(v) for v in eps.values()])), 4) if eps else np.nan
                # 국면 짝: 같은 하락 국면의 바닥(0/1) 대 그 국면 F 날 비율
                for grp, sel in (("B10", lambda e: True), ("B15", lambda e: e["B15"])):
                    dd = [float(j in t[1][side]) - float(np.mean(eps[t[0]["episode"]]))
                          for t in einfo if sel(t[0]) and t[0]["episode"] in eps]
                    r[f"pair_n_{grp}"] = len(dd)
                    r[f"pair_gt_{grp}"] = int(sum(x > 0 for x in dd))
                    r[f"pair_lt_{grp}"] = int(sum(x < 0 for x in dd))
                r["recurring_B15"] = r["hit_B15"] >= REC_MIN
                r["recurring_B10"] = r["hit_B10"] >= 9
                rows.append(r)
    rec = pd.DataFrame(rows)
    for c in ("B15", "B10"):
        rec[f"q_{c}"] = bh(rec[f"p_{c}"].where(rec[f"hit_{c}"] > 0)).round(4)
        rec[f"qpx_{c}"] = bh(rec[f"ppx_{c}"].where(rec[f"hit_{c}"] > 0)).round(4)
        rec[f"p_{c}"] = rec[f"p_{c}"].round(5)
        rec[f"ppx_{c}"] = rec[f"ppx_{c}"].round(5)
    rec["lift_B15"] = (rec["hit_B15"] / rec["base_exp_B15"].replace(0, np.nan)).round(3)
    rec["liftpx_B15"] = (rec["hit_B15"] / rec["basepx_exp_B15"].replace(0, np.nan)).round(3)
    return rec.sort_values(["window", "side", "hit_B15", "hit_B10", "base_rate_B15"],
                           ascending=[True, True, False, False, False]).reset_index(drop=True)


# ---------- 5b. 메아리 점검(엄격) ----------
def echo_check(rec, evs, net, prods, fdates, closes, min_hit=3):
    """B15에서 3번 이상 상위 5인 상품: 같은 해·같은 길이 창 가운데 가격이 바닥 창만큼 이상 움직인 창
    (바닥 창 수익 < 0이면 ≤, > 0이면 ≥, 바닥 창과 겹치는 창 제외)에서 상위 5 비율. SOXX·QQQ 따로."""
    cn = np.vstack([np.zeros((1, net.shape[1])), np.cumsum(net, 0)])
    pidx = {p: i for i, p in enumerate(prods)}
    years = fdates.year.to_numpy()
    rows, summ = [], []
    cache = {}
    for _, rr in rec[rec["hit_B15"] >= min_hit].iterrows():
        w, side, p = rr["window"], rr["side"], rr["product"]
        j = pidx[p]
        rates = {"SOXX": [], "QQQ": []}
        hits = []
        for e in evs:
            if not e["B15"]:
                continue
            a, b = e["win"][w]
            L = b - a + 1
            if L not in cache:
                ends = np.arange(L - 1, net.shape[0])
                cache[L] = {"ends": ends, "S": cn[ends + 1] - cn[ends + 1 - L]}
            c = cache[L]
            ends, S = c["ends"], c["S"]
            order = np.argsort(-S, 1)[:, :RECN] if side == "buy" else np.argsort(S, 1)[:, :RECN]
            inn = (order == j).any(1) & ((S[:, j] > 0) if side == "buy" else (S[:, j] < 0))
            hit = bool(inn[b - (L - 1)])
            hits.append(hit)
            rec_e = {"window": w, "side": side, "product": p, "event": iso(e["date"]), "L": L, "hit": hit}
            for t, cl in closes.items():
                r_all = cl[ends + 1] / cl[ends + 1 - L] - 1
                r_e = cl[b + 1] / cl[a] - 1
                m = (years[ends] == years[b]) & (np.abs(ends - b) >= L)
                m &= (r_all <= r_e) if r_e < 0 else (r_all >= r_e)
                rate = float(inn[m].mean()) if m.sum() else np.nan
                rates[t].append(rate)
                rec_e.update({f"{t.lower()}_ret": round(float(r_e), 4), f"n_{t.lower()}": int(m.sum()),
                              f"rate_{t.lower()}": round(rate, 3) if np.isfinite(rate) else np.nan})
            rows.append(rec_e)
        k = int(sum(hits))
        s = {"window": w, "side": side, "product": p, "hit_B15": k, "n_B15": len(hits),
             "basepx_rate_B15": rr["basepx_rate_B15"], "ppx_B15": rr["ppx_B15"], "F_rate": rr["F_rate"]}
        for t in closes:
            ps = np.array(rates[t], float)
            ok = np.isfinite(ps)
            s[f"n_ev_{t.lower()}"] = int(ok.sum())
            s[f"exp_{t.lower()}"] = round(float(ps[ok].sum()), 2)
            # 비교 창 없는 사건은 빼고, 남은 사건의 맞음 수로 꼬리 확률
            s[f"p_{t.lower()}"] = round(pb_tail(ps[ok], int(np.array(hits)[ok].sum())), 4) if ok.any() else np.nan
        summ.append(s)
    return pd.DataFrame(summ), pd.DataFrame(rows)


# ---------- 6. 회전 표 ----------
def rotation(evs, fds, net, catm, a1, base, fdates, px):
    years = fdates.year
    keys = CKEYS + ["ALL", "FUND"]
    rec = []
    for grp, items in (("bottom", evs), ("F", fds)):
        for x in items:
            for w in ROT_WINS:
                ab = x["win"][w]
                if ab is None:
                    continue
                a, b = ab
                rets = {f"{t.lower()}_ret": wret(px[t], fdates, ab) for t in ("SOXX", "QQQ")}
                S = net[a:b + 1].sum(0)
                v = rot_values(catm[a:b + 1].sum(0)[None, :], np.array([a1[a:b + 1].sum()]),
                               np.array([S[S > 0].sum()]), np.array([S[S < 0].sum()]))
                bl, i, _, _, qq = base.at(ab)
                m = bl["years"] == years[b]
                mq = m & (bl["q"] == qq)  # 같은 해·같은 SOXX 수익 5분위
                br = bl["rot"]
                for ci, k in enumerate(keys):
                    if k == "ALL":
                        val, ser, net_usd, turn = v["all_turn"][0], br["all_turn"], np.nan, v["all_turn"][0]
                    elif k == "FUND":
                        val, ser, net_usd, turn = v["fund"][0], br["fund"], np.nan, np.nan
                    else:
                        val, ser = v["share"][0, ci], br["share"][:, ci]
                        net_usd, turn = v["net"][0, ci], v["turn"][0, ci]
                    if abs(ser[i] - val) > 1e-9 * max(1.0, abs(val)):
                        raise SystemExit(f"굴린 창 값 불일치 {x['date']} {w} {k}")
                    ref, refq = ser[m], ser[mq]
                    rec.append({"group": grp, "date": iso(x["date"]), "B15": x["B15"], "episode": x["episode"],
                                "window": w, "L": wlen(ab), "year": int(years[b]), **rets, "soxx_q": qq, "cat": k,
                                "value": val, "net_usd": net_usd, "net_over_turnover": turn,
                                "P_year": pct_of(val, ref), "base_median": float(np.nanmedian(ref)),
                                "base_frac_neg": float(np.mean(ref < 0)) if k not in ("ALL", "FUND") else np.nan,
                                "n_base": int(np.isfinite(ref).sum()),
                                "P_px": pct_of(val, refq), "basepx_median": float(np.nanmedian(refq)),
                                "n_basepx": int(np.isfinite(refq).sum())})
    det = pd.DataFrame(rec)
    det["label"] = det["cat"].map({**CLAB, "ALL": "전체 순매수 ÷ 거래(A4/A1)", "FUND": "상품 순매도 합 ÷ 순매수 합"})
    rows = []
    for (w, k), g in det.groupby(["window", "cat"], sort=False):
        bt = g[g["group"] == "bottom"]
        b15 = bt[bt["B15"]].sort_values("date")
        F = g[g["group"] == "F"]
        r = {"window": w, "cat": k, "label": g["label"].iloc[0]}
        for nm, sub in (("B15", b15), ("B10", bt)):
            r[f"n_{nm}"] = len(sub)
            r[f"{nm}_median"] = sub["value"].median()
            r[f"{nm}_n_neg"] = int((sub["value"] < 0).sum())
            r[f"{nm}_n_above_base"] = int((sub["value"] > sub["base_median"]).sum())
            r[f"{nm}_median_P"] = sub["P_year"].median()
            r[f"{nm}_n_P80"] = int((sub["P_year"] >= 80).sum())
            r[f"{nm}_n_P20"] = int((sub["P_year"] <= 20).sum())
            r[f"{nm}_median_Ppx"] = sub["P_px"].median()
            r[f"{nm}_n_Ppx80"] = int((sub["P_px"] >= 80).sum())
            r[f"{nm}_n_Ppx20"] = int((sub["P_px"] <= 20).sum())
        r["B15_values"] = ";".join(f"{v:.3f}" for v in b15["value"])
        r["B15_P"] = ";".join(f"{v:.0f}" for v in b15["P_year"])
        r["B15_Ppx"] = ";".join(f"{v:.0f}" for v in b15["P_px"])
        r["base_median_B15yrs"] = b15["base_median"].median()
        r["basepx_median_B15"] = b15["basepx_median"].median()
        r["base_frac_neg_B15yrs"] = b15["base_frac_neg"].mean()
        r["n_F"] = len(F)
        r["F_median"] = F["value"].median()
        r["F_n_neg"] = int((F["value"] < 0).sum())
        r["F_median_P"] = F["P_year"].median()
        r["F_median_Ppx"] = F["P_px"].median()
        r["F_n_above_base"] = int((F["value"] > F["base_median"]).sum())
        epm = F[F["episode"] != "current"].groupby("episode")["value"].mean()
        d = (bt.set_index("episode")["value"] - epm).dropna()
        d15 = d[d.index.isin(b15["episode"])]
        r["n_pairs"] = len(d)
        r["pair_median_d"] = d.median() if len(d) else np.nan
        r["pair_n_pos"] = int((d > 0).sum())
        r["pair_sign_p"] = float(stats.binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue) if (d != 0).any() else np.nan
        r["n_pairs_B15"] = len(d15)
        r["pair_B15_n_pos"] = int((d15 > 0).sum())
        for t in ("soxx", "qqq"):  # 메아리 점검용 창 가격 수익(중앙)
            r[f"B15_{t}_ret_med"] = b15[f"{t}_ret"].median()
            r[f"F_{t}_ret_med"] = F[f"{t}_ret"].median()
        rows.append(r)
    summ = pd.DataFrame(rows)
    order = {w: i for i, w in enumerate(ROT_WINS)}
    corder = {k: i for i, k in enumerate(keys)}
    summ = summ.sort_values(["window", "cat"], key=lambda s: s.map(order) if s.name == "window" else s.map(corder))
    return summ.reset_index(drop=True), det


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    f, md, pf, units, ev = load()
    th, names = load_raw_lists()
    fdates, prods, mats, listed, full, ipo, hi, lo, pt = build_mats(f, md, units, th)
    pname = product_names(f, names, prods)
    sess = pf.index
    evs, fds = windows(ev, sess, fdates)
    onehot = np.zeros((len(prods), len(CKEYS)))
    onehot[np.arange(len(prods)), pt["cat"].map({k: i for i, k in enumerate(CKEYS)}).to_numpy()] = 1
    catm = mats["net"] @ onehot
    a1 = md["A1"].to_numpy(float)
    merged = md["merged"].to_numpy(bool)
    sc = pf["SOXX_Close"]
    i0 = sc.index.get_loc(fdates[0])
    cl = np.concatenate([[sc.iloc[i0 - 1]], sc.reindex(fdates).to_numpy(float)])
    if not np.isfinite(cl).all():
        raise SystemExit("SOXX 종가 빠짐")
    base = Base(mats["net"], catm, a1, fdates.year.to_numpy(), cl)

    tl, safe = top_lists(evs, mats, listed, full, ipo, hi, lo, pt, pname, prods, fdates, pf, merged)
    tl.to_csv(OUT / "top_lists.csv", index=False, encoding="utf-8-sig")
    rec = recurrence(evs, fds, mats["net"], base, prods, pt, pname, fdates)
    rec.to_csv(OUT / "recurrence.csv", index=False, encoding="utf-8-sig")
    pxl = pd.read_csv(DER / "prices.csv", parse_dates=["date"])
    px = {t: pxl[pxl["ticker"] == t].set_index("date")["Close"].sort_index() for t in ("SOXX", "QQQ")}
    closes = {}
    for t, c in px.items():
        k0 = c.index.get_loc(fdates[0])
        closes[t] = np.concatenate([[c.iloc[k0 - 1]], c.reindex(fdates).to_numpy(float)])
    ech, ech_ev = echo_check(rec, evs, mats["net"], prods, fdates, closes)
    ech.to_csv(OUT / "echo_check.csv", index=False, encoding="utf-8-sig")
    ech_ev.to_csv(OUT / "echo_check_events.csv", index=False, encoding="utf-8-sig")
    rot, det = rotation(evs, fds, mats["net"], catm, a1, base, fdates, px)
    rot.to_csv(OUT / "rotation.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    det.to_csv(OUT / "rotation_events.csv", index=False, encoding="utf-8-sig", float_format="%.5g")

    # 점검·범위
    tested = rec[rec["hit_B15"] > 0]
    cat_net = pd.Series(catm.sum(0), index=CKEYS)
    run = {
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {"flows_rows": int(len(f)), "trade_days": int(len(fdates)), "first": iso(fdates[0]),
                   "last": iso(fdates[-1]), "products": len(prods)},
        "events": {"B10": len(evs), "B15": int(sum(e["B15"] for e in evs)), "F": len(fds),
                   "F_valid_by_window": {w: int(sum(x["win"][w] is not None for x in fds)) for w in ROT_WINS},
                   "decline_len": {iso(e["date"]): wlen(e["win"]["decline"]) for e in evs}},
        "cells": {"top_list_rows": int(len(tl)), "event_windows": int(len(safe) // 2),
                  "recurrence_rows": int(len(rec)), "recurrence_rows_B15_hit": int(len(tested)),
                  "recurrence_B15_p05": int((tested["p_B15"] < 0.05).sum()),
                  "recurrence_B15_p05_expected_upper": round(0.05 * len(tested), 1),
                  "recurrence_B15_q10": int((tested["q_B15"] <= 0.10).sum()),
                  "recurrence_B15_ppx05": int((tested["ppx_B15"] < 0.05).sum()),
                  "recurrence_B15_qpx10": int((tested["qpx_B15"] <= 0.10).sum()),
                  "recurrence_recurring_B15": rec.loc[rec["recurring_B15"], ["window", "side", "product"]]
                  .astype(str).agg(":".join, axis=1).tolist(),
                  "rotation_cells": int(len(rot)), "rotation_pair_p05": int((rot["pair_sign_p"] < 0.05).sum()),
                  "rotation_pair_p05_cells": rot.loc[rot["pair_sign_p"] < 0.05, ["window", "cat"]]
                  .astype(str).agg(":".join, axis=1).tolist(),
                  "rotation_pair_p05_null_expect": round(len(rot) * 2 / 2 ** 7, 1)},
        "censor": {
            "safe5_share": {s: round(float(safe.loc[safe["side"] == s, "safe5"].mean()), 3) for s in ("buy", "sell")},
            "safe10_share": {s: round(float(safe.loc[safe["side"] == s, "safe10"].mean()), 3) for s in ("buy", "sell")},
            "safe_by_window": safe.groupby(["window", "side"])[["safe5", "safe10"]].mean().round(3)
            .reset_index().to_dict("records"),
            "daily_hidden_max_buy_median": round(float(np.median(hi))),
            "daily_hidden_max_sell_median": round(float(np.median(lo))),
            "top_rows_not_fully_listed": int((tl["n_listed_days"] < tl["n_sessions"]).sum()),
        },
        "categories": {k: {"label": CLAB[k], "n_products": int((pt["cat"] == k).sum()),
                           "net_sum_usd": round(float(cat_net[k]))} for k in CKEYS},
        "params": {"TOPN": TOPN, "RECN": RECN, "REC_MIN": REC_MIN, "BASE_MIN": BASE_MIN,
                   "decline": "(10% 지그재그 고점, 바닥] = 고점 다음 세션 ~ 바닥 날", "F_decline": "(국면 고점, F 날]",
                   "baseline": f"같은 길이 굴린 창, 창 끝이 같은 해(흐름 자료 {iso(fdates[0])} ~ {iso(fdates[-1])} 안, 사건 창 포함)",
                   "baseline_px": "위 창 가운데 SOXX 창 수익 5분위(그 해 안)가 같은 창(메아리 대조)",
                   "share": "범주 순매수 ÷ Σ|범주 순매수|(창 합)"},
    }
    (OUT / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(run["cells"], ensure_ascii=False), json.dumps(run["censor"]["safe5_share"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
