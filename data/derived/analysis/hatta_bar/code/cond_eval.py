"""2차 판정(BRIEF2.md): 원안 ①·② 신호에 조건 18개를 한·두 겹 얹은 칸과 깊이 3 나무를 앞 구간(7~8월)에서 고르고
뒤 구간(9~10월)·2025로 확인한다. 고르기 점수 = R1 비용 후(비용 모형)의 거래일 묶음 t값.
사용: py cond_eval.py stage1   → 한·두 조건 칸 전부, 나무, 순열 귀무, 주 가설 6개(E1 전)
      py cond_eval.py e1       → 주 가설 뒤 구간 신호의 실측 호가 손익"""
import os, sys, json
import numpy as np
import pandas as pd
import numba as nb
import engine as E
from cond_cost import tier_of, tod_of

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
SPLIT = 20260831
ELIG = dict(n=60, days=15, syms=10, share=0.20)
B_BOOT, N_PERM = 2000, 200
EXIT_TOD = {0: "pre", 1: "am", 2: "am", 3: "post"}

# (이름, 한국어, 열, 경계 또는 None(따로 계산), 구간 이름)
VARS = [
    ("cap", "시총", "cap_b", [5, 20, 100], ["<50억$", "50억~200억$", "200억~1000억$", "1000억$+"]),
    ("adv", "거래대금", "adv20", [1e8, 1e9], ["<1억$", "1억~10억$", "10억$+"]),
    ("price", "주가", "prevclose", [20, 100], ["<20$", "20~100$", "100$+"]),
    ("ai", "AI종목", None, None, ["아님", "AI"]),
    ("tod", "시간대", None, None, ["장전", "개장15분", "오전", "장후"]),
    ("dayret", "당일낙폭", "dayret", [-0.10, -0.05, -0.02], ["<-10%", "-10~-5%", "-5~-2%", ">-2%"]),
    ("qqq_day", "QQQ당일", "qqq_day", [-0.01, 0.0], ["<-1%", "-1~0%", ">0%"]),
    ("qqq_30", "QQQ30분", "qqq_30", [-0.003, 0.0], ["<-0.3%", "-0.3~0%", ">0%"]),
    ("rel", "상대낙폭", "rel", [-0.05, -0.02], ["<-5%", "-5~-2%", ">-2%"]),
    ("vol_tod", "시간대거래량", "vol_tod", [2, 5], ["<2배", "2~5배", ">5배"]),
    ("vwap", "VWAP이격", "vwap_gap", [-0.03, -0.01], ["<-3%", "-3~-1%", ">-1%"]),
    ("depth", "120분하락", "dd", [0.03, 0.05], ["2~3%", "3~5%", ">5%"]),
    ("ret5d", "5일수익률", "ret5d", [-0.05, 0.05], ["<-5%", "-5~5%", ">5%"]),
    ("atrd", "일봉ATR", "atrd", None, ["낮음", "중간", "높음"]),
    ("earn", "실적일", None, None, ["아님", "실적일", "모름"]),
    ("chain", "연쇄", None, None, ["첫 신호", "두번째+"]),
    ("kcnt", "지지선수", None, None, ["적음", "많음"]),
    ("recsize", "회복크기", "recsize", [0.003, 0.006], ["<0.3%", "0.3~0.6%", ">0.6%"]),
]


def codes(R, atr_edges):
    C = {}
    for name, _, col, edges, labels in VARS:
        if name == "ai":
            c = R.ai.astype(int).to_numpy()
        elif name == "tod":
            c = R.mins.map(lambda m: ["pre", "open15", "am", "post"].index(tod_of(m))).to_numpy()
        elif name == "earn":
            c = R.earn.map({0: 0, 1: 1, -1: 2}).to_numpy()
        elif name == "chain":
            c = (R.chain >= 1).astype(int).to_numpy()
        elif name == "kcnt":
            c = np.where(R.pat == "A", R.kcnt >= 3, R.kcnt >= 2).astype(int)
        else:
            x = R[col].to_numpy(dtype=float)
            e = atr_edges if name == "atrd" else edges
            c = np.where(np.isnan(x), -1, np.digitize(x, e))
        C[name] = np.asarray(c, dtype=np.int64)
    return pd.DataFrame(C, index=R.index)


def load(year, cost):
    R = pd.read_parquet(os.path.join(E.CACHE, f"cond_{year}.parquet"))
    R = R[R.arm == "sig"].copy()
    tier = R.cap_b.map(tier_of)
    tod = R.mins.map(tod_of)
    tab = cost["table"]
    c_in = np.array([tab.get(f"{a}|{b}", {"half": cost["by_tod"][b]})["half"] for a, b in zip(tier, tod)])
    c_out = np.array([tab.get(f"{a}|{EXIT_TOD[int(s)]}", {"half": cost["by_tod"][EXIT_TOD[int(s)]]})["half"]
                      for a, s in zip(tier, R.sess_out)])
    R["r1n"] = R.r1_e2 - c_in - np.where(R.hit_e2 > 0, 0.0, c_out)
    R["c_in"], R["c_out"] = c_in, c_out
    return R.sort_values(["sid", "ts", "pat"]).reset_index(drop=True)


@nb.njit(cache=True)
def cstats(sel, sid, ts, dcode, per, r1n, r1, bt, hit, race, nd, nsym):
    """선택 행 → 같은 종목 30분 겹침 빼기 → 기간(0 앞, 1 뒤)별 [n, 거래일, 종목, 하루 몫, R1비용후, t, 도달, 경주, R1, R1−같은분]."""
    out = np.full((2, 10), np.nan)
    dsum = np.zeros((2, nd))
    dcnt = np.zeros((2, nd))
    seen = np.zeros((2, nsym), np.bool_)
    acc = np.zeros((2, 8))
    last_s = -1
    last_t = -10 ** 12
    for q in range(len(sel)):
        i = sel[q]
        if sid[i] == last_s and ts[i] - last_t < 30:
            continue
        last_s = sid[i]
        last_t = ts[i]
        w = per[i]
        dsum[w, dcode[i]] += r1n[i]
        dcnt[w, dcode[i]] += 1
        seen[w, sid[i]] = True
        acc[w, 0] += 1
        acc[w, 1] += r1n[i]
        acc[w, 2] += hit[i]
        if not np.isnan(race[i]):
            acc[w, 3] += race[i]
            acc[w, 4] += 1
        acc[w, 5] += r1[i]
        if not np.isnan(bt[i]):
            acc[w, 6] += r1[i] - bt[i]
            acc[w, 7] += 1
    for w in range(2):
        n = acc[w, 0]
        out[w, 0] = n
        if n == 0:
            continue
        G = 0
        mx = 0.0
        for d in range(nd):
            if dcnt[w, d] > 0:
                G += 1
                mx = max(mx, dcnt[w, d])
        ns = 0
        for s in range(nsym):
            if seen[w, s]:
                ns += 1
        m = acc[w, 1] / n
        v = 0.0
        for d in range(nd):
            if dcnt[w, d] > 0:
                e = dsum[w, d] - m * dcnt[w, d]
                v += e * e
        out[w, 1] = G
        out[w, 2] = ns
        out[w, 3] = mx / n
        out[w, 4] = m
        out[w, 5] = m / (np.sqrt(v * G / (G - 1)) / n) if (G > 1 and v > 0) else np.nan
        out[w, 6] = acc[w, 2] / n
        out[w, 7] = acc[w, 3] / acc[w, 4] if acc[w, 4] > 0 else np.nan
        out[w, 8] = acc[w, 5] / n
        out[w, 9] = acc[w, 6] / acc[w, 7] if acc[w, 7] > 0 else np.nan
    return out


SN = ["n", "days", "syms", "share", "r1n", "t", "hit", "race", "r1", "vs_bt"]


class Group:
    """한 묶음(①·②·아무거나)의 배열과 조건 마스크."""
    def __init__(self, R, C, front_split=True):
        self.R, self.C = R, C
        days = np.sort(R.day.unique())
        self.a = dict(sid=R.sid.to_numpy().astype(np.int64), ts=R.ts.to_numpy().astype(np.int64),
                      dcode=np.searchsorted(days, R.day.to_numpy()).astype(np.int64),
                      per=((R.day.to_numpy() > SPLIT).astype(np.int64) if front_split else np.zeros(len(R), np.int64)),
                      r1n=R.r1n.to_numpy(), r1=R.r1_e2.to_numpy(), bt=R.bt2_r1.to_numpy(), hit=R.hit_e2.to_numpy(),
                      race=R.race_e2.to_numpy(), nd=len(days), nsym=int(R.sid.max()) + 1)

    def stats(self, mask):
        a = self.a
        return cstats(np.nonzero(mask)[0].astype(np.int64), a["sid"], a["ts"], a["dcode"], a["per"], a["r1n"], a["r1"],
                      a["bt"], a["hit"], a["race"], a["nd"], a["nsym"])


def cell_list():
    singles = [((v[0], b),) for v in VARS for b in range(len(v[4]))]
    pairs = []
    for i in range(len(VARS)):
        for j in range(i + 1, len(VARS)):
            for bi in range(len(VARS[i][4])):
                for bj in range(len(VARS[j][4])):
                    pairs.append(((VARS[i][0], bi), (VARS[j][0], bj)))
    return singles + pairs


def cell_mask(C, cell):
    m = np.ones(len(C), bool)
    for name, b in cell:
        m &= C[name].to_numpy() == b
    return m


def eval_cells(G, cells, masks=None):
    rows = []
    for cell in cells:
        mk = masks[cell] if masks is not None else cell_mask(G.C, cell)
        st = G.stats(mk)
        rows.append(st)
    return np.array(rows)


def eligible(F):
    return (F[:, 0] >= ELIG["n"]) & (F[:, 1] >= ELIG["days"]) & (F[:, 2] >= ELIG["syms"]) & (F[:, 3] <= ELIG["share"])


def describe(cell):
    lab = {v[0]: v for v in VARS}
    return " & ".join(f"{lab[n][1]} {lab[n][4][b]}" for n, b in cell)


def tree_rule(G, front_rows, seed=0):
    """앞 구간(겹침 뺀 기본 신호)으로 깊이 3 회귀 나무 → 앞 평균이 가장 높은 잎(80건 이상)."""
    from sklearn.tree import DecisionTreeRegressor
    X = tree_X(G.C)
    y = G.R.r1n.to_numpy()
    tr = DecisionTreeRegressor(max_depth=3, min_samples_leaf=80, random_state=seed).fit(X[front_rows], y[front_rows])
    leaf = tr.apply(X)
    lf = pd.Series(y[front_rows]).groupby(leaf[front_rows]).agg(["mean", "size"])
    lf = lf[lf["size"] >= 80]
    best = int(lf["mean"].idxmax())
    return tr, leaf == best, best, rule_text(tr, best, X.columns)


def tree_X(C):
    X = C.copy()
    for cat, k in (("tod", 4), ("earn", 3)):
        for v in range(k):
            X[f"{cat}={v}"] = (C[cat] == v).astype(int)
        X = X.drop(columns=cat)
    return X


def rule_text(tr, leaf_id, cols):
    t = tr.tree_
    path, node = [], 0
    parent = {}
    for nd in range(t.node_count):
        for ch, side in ((t.children_left[nd], "<="), (t.children_right[nd], ">")):
            if ch != -1:
                parent[ch] = (nd, side)
    while leaf_id in parent:
        nd, side = parent[leaf_id]
        path.append(f"{cols[t.feature[nd]]} {side} {t.threshold[nd]:.2f}")
        leaf_id = nd
    return " & ".join(reversed(path))


def front_dedup_rows(G):
    """앞 구간 기본 신호를 겹침 빼기 한 행 번호(나무 학습용)."""
    a = G.a
    keep = E.dedup(a["sid"], a["ts"], 30)
    return keep & (a["per"] == 0)


def boot_p(G, mask, per, rng):
    """거래일 묶음 부트스트랩 단측 p: R1 비용 후 > 0, R1 − 같은 분 > 0 (겹침 뺀 뒤)."""
    a = G.a
    sel = np.nonzero(mask & (a["per"] == per))[0]
    keep = E.dedup(a["sid"][sel], a["ts"][sel], 30)
    sel = sel[keep]
    if len(sel) < 10:
        return {"n": int(len(sel)), "p_r1n": 1.0, "p_bt": 1.0}
    d = a["dcode"][sel]
    days = np.unique(d)
    pos = np.searchsorted(days, d)
    s1 = np.bincount(pos, weights=a["r1n"][sel], minlength=len(days))
    x2 = a["r1"][sel] - a["bt"][sel]
    ok = ~np.isnan(x2)
    s2 = np.bincount(pos[ok], weights=x2[ok], minlength=len(days))
    c1 = np.bincount(pos, minlength=len(days)).astype(float)
    c2 = np.bincount(pos[ok], minlength=len(days)).astype(float)
    d1, d2 = [], []
    for _ in range(B_BOOT):
        w = np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days))
        d1.append((s1 * w).sum() / max((c1 * w).sum(), 1))
        d2.append((s2 * w).sum() / max((c2 * w).sum(), 1))
    d1, d2 = np.array(d1), np.array(d2)
    return {"n": int(len(sel)), "r1n": float(s1.sum() / c1.sum()), "vs_bt": float(s2.sum() / max(c2.sum(), 1)),
            "p_r1n": float((1 + np.sum(d1 <= 0)) / (B_BOOT + 1)), "p_bt": float((1 + np.sum(d2 <= 0)) / (B_BOOT + 1))}


def confirm_rows(G, mask, per, name, year):
    a = G.a
    sel = np.nonzero(mask & (a["per"] == per))[0]
    sel = sel[E.dedup(a["sid"][sel], a["ts"][sel], 30)]
    X = G.R.iloc[sel][["sym", "ts", "day", "t", "mins", "pat", "r1_e2", "r1n", "hit_e2", "c_in", "c_out"]].copy()
    X["hyp"], X["year"] = name, year
    return X


def e1():
    """주 가설 확인 구간 신호의 실측 호가 손익(신호 마감 + 10초 매도호가 진입, 창 끝 매수호가 청산)."""
    from concurrent.futures import ThreadPoolExecutor
    from fetch_alpaca import keys, Pace
    from fetch_quotes import last_quote
    S = pd.read_csv(os.path.join(RES, "cond_hyp_signals.csv"))
    U = S.drop_duplicates(["sym", "ts"])[["sym", "ts"]]
    head, pace = keys(), Pace(3.0)

    def one(r):
        t = pd.Timestamp(int(r[1]) * 60, unit="s", tz="UTC")
        ai, bi = last_quote(r[0], t + pd.Timedelta(seconds=70), head, pace)
        ao, bo = last_quote(r[0], t + pd.Timedelta(minutes=31), head, pace)
        return r[0], int(r[1]), ai, bi, ao, bo

    with ThreadPoolExecutor(6) as ex:
        Q = pd.DataFrame(list(ex.map(one, U.itertuples(index=False, name=None))),
                         columns=["sym", "ts", "ask_in", "bid_in", "ask_out", "bid_out"])
    S = S.merge(Q, on=["sym", "ts"], how="left")
    bars = {}
    r = []
    for x in S.itertuples():
        src = "alp1m" if x.year == 2026 or str(x.year) == "2026" else "alp1m_2025"
        if (x.sym, src) not in bars:
            bars[(x.sym, src)] = E.prep(E.load(x.sym, src))
        a = bars[(x.sym, src)]
        t = int(x.t)
        if not (x.ask_in > 0):
            r.append(np.nan)
            continue
        tlim = a["ts"][t] + 30
        z = t + 1
        while z + 1 < len(a["ts"]) and a["ts"][z + 1] <= tlim and a["day"][z + 1] == a["day"][t]:
            z += 1
        tg = x.ask_in * 1.01
        hit = a["c"][t + 1] >= tg or (z >= t + 2 and np.max(a["h"][t + 2:z + 1]) >= tg)
        ex_ = x.bid_out if x.bid_out > 0 else a["c"][z]
        r.append(0.01 if hit else ex_ / x.ask_in - 1)
    S["r1_e1"] = r
    S.to_csv(os.path.join(RES, "cond_hyp_signals_e1.csv"), index=False)
    res = {k: {"n": int(g.r1_e1.notna().sum()), "r1_e1": float(g.r1_e1.mean()), "r1n_model": float(g.r1n.mean()),
               "hit_e1": float((g.r1_e1 == 0.01).mean())} for k, g in S.groupby("hyp")}
    P = json.load(open(os.path.join(RES, "cond_primary.json"), encoding="utf-8"))
    for k, v in res.items():
        h = P["hyps"][k]
        v["accept"] = bool(h["pass_stat"] and v["r1_e1"] > 0 and (h["chain"] or h["y2025"].get("r1n", -1) > 0))
        h["e1"] = v
    json.dump(P, open(os.path.join(RES, "cond_primary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def holm(ps):
    order = sorted(ps, key=lambda k: ps[k])
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, ps[k] * (len(ps) - i)))
        out[k] = run
    return out


def stage1():
    cost = json.load(open(os.path.join(RES, "cost_model.json"), encoding="utf-8"))
    R26, R25 = load("2026", cost), load("2025", cost)
    fa = R26[(R26.day <= SPLIT)]
    atr_edges = list(np.nanpercentile(fa.atrd, [100 / 3, 200 / 3]))
    C26, C25 = codes(R26, atr_edges), codes(R25, atr_edges)
    cells = cell_list()
    rng = np.random.default_rng(20261030)
    out = {"atr_edges": atr_edges, "n_cells_per_group": len(cells), "groups": {}}
    single_rows, all_cells, hyp_rows = [], [], []
    hyps = {}
    for gname, sel26, sel25 in (("①", R26.pat == "A", R25.pat == "A"), ("②", R26.pat == "B", R25.pat == "B"),
                               ("아무거나", np.ones(len(R26), bool), np.ones(len(R25), bool))):
        G = Group(R26[sel26].reset_index(drop=True), C26[sel26].reset_index(drop=True))
        H = Group(R25[sel25].reset_index(drop=True), C25[sel25].reset_index(drop=True), front_split=False)
        masks = {c: cell_mask(G.C, c) for c in cells}
        M = eval_cells(G, cells, masks)                      # [cell, 2, 10]
        M25 = eval_cells(H, cells)
        F, Bk = M[:, 0, :], M[:, 1, :]
        el = eligible(F)
        score = np.where(el, F[:, 5], -np.inf)
        order = np.argsort(-score)
        base = G.stats(np.ones(len(G.R), bool))
        base25 = H.stats(np.ones(len(H.R), bool))
        for ci, cell in enumerate(cells):
            rec = {"group": gname, "cell": describe(cell), "k": len(cell), "elig": bool(el[ci]),
                   "rank": int(np.nonzero(order == ci)[0][0]) + 1 if el[ci] else None}
            for pi, pn in ((0, "f"), (1, "b")):
                for si, sn in enumerate(SN):
                    rec[f"{pn}_{sn}"] = float(M[ci, pi, si])
            for si, sn in enumerate(SN):
                rec[f"y25_{sn}"] = float(M25[ci, 0, si])
            all_cells.append(rec)
            if len(cell) == 1:
                single_rows.append(rec)
        top = [cells[i] for i in order[:10] if el[i]]
        # 나무
        fr = front_dedup_rows(G)
        tr, tmask, leaf, rule = tree_rule(G, fr)
        X25 = tree_X(H.C)
        tmask25 = tr.apply(X25) == leaf
        ts_ = G.stats(tmask)
        ts25 = H.stats(tmask25)
        # 그레이디언트 부스팅 상위 20%(설명용)
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xg = tree_X(G.C)
        gb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                                           random_state=0).fit(Xg[fr], G.R.r1n.to_numpy()[fr])
        pr = gb.predict(Xg)
        thr = np.percentile(pr[fr], 80)
        gbs = G.stats(pr >= thr)
        gbs25 = H.stats(gb.predict(X25) >= thr)
        # 순열 귀무: 조건 행을 신호끼리 섞어 같은 고르기
        perm = []
        for p in range(N_PERM):
            pi_ = rng.permutation(len(G.R))
            pm = {c: masks[c][pi_] for c in cells}
            Mp = eval_cells(G, cells, pm)
            elp = eligible(Mp[:, 0, :])
            sp = np.where(elp, Mp[:, 0, 5], -np.inf)
            op = np.argsort(-sp)[:10]
            op = [i for i in op if elp[i]]
            perm.append({"top1_b_r1n": float(Mp[op[0], 1, 4]) if op else np.nan,
                         "top1_f_t": float(Mp[op[0], 0, 5]) if op else np.nan,
                         "top10_pos": int(sum(Mp[i, 1, 4] > 0 for i in op))})
        perm = pd.DataFrame(perm)
        obs_top1 = float(M[order[0], 1, 4])
        obs_pos = int(sum(M[cells.index(c), 1, 4] > 0 for c in top))
        out["groups"][gname] = {
            "base_front": dict(zip(SN, map(float, base[0]))), "base_back": dict(zip(SN, map(float, base[1]))),
            "base_2025": dict(zip(SN, map(float, base25[0]))), "eligible": int(el.sum()),
            "top10": [{"cell": describe(c), "front": dict(zip(SN, map(float, M[cells.index(c), 0]))),
                       "back": dict(zip(SN, map(float, M[cells.index(c), 1]))),
                       "y2025": dict(zip(SN, map(float, M25[cells.index(c), 0])))} for c in top],
            "tree": {"rule": rule, "front": dict(zip(SN, map(float, ts_[0]))), "back": dict(zip(SN, map(float, ts_[1]))),
                     "y2025": dict(zip(SN, map(float, ts25[0])))},
            "gb_top20": {"front": dict(zip(SN, map(float, gbs[0]))), "back": dict(zip(SN, map(float, gbs[1]))),
                         "y2025": dict(zip(SN, map(float, gbs25[0])))},
            "perm": {"top1_b_r1n_obs": obs_top1, "top1_b_r1n_null_mean": float(perm.top1_b_r1n.mean()),
                     "top1_b_r1n_null_p": float((1 + (perm.top1_b_r1n >= obs_top1).sum()) / (N_PERM + 1)),
                     "top10_pos_obs": obs_pos, "top10_pos_null_mean": float(perm.top10_pos.mean()),
                     "top10_pos_null_p": float((1 + (perm.top10_pos >= obs_pos).sum()) / (N_PERM + 1)),
                     "top1_f_t_obs": float(F[order[0], 5]), "top1_f_t_null_mean": float(perm.top1_f_t.mean())}}
        # 주 가설: 앞 1위 칸, 나무 잎
        top1 = top[0]
        chain_cell = any(n == "chain" for n, _ in top1)
        m1 = masks[top1]
        hyps[f"{gname} 1위칸"] = {"cell": describe(top1), "chain": chain_cell,
                                  "back": boot_p(G, m1, 1, rng), "y2025": boot_p(H, cell_mask(H.C, top1), 0, rng)}
        chain_tree = "chain" in rule
        hyps[f"{gname} 나무"] = {"cell": rule, "chain": chain_tree, "back": boot_p(G, tmask, 1, rng),
                                 "y2025": boot_p(H, tmask25, 0, rng)}
        for hk, mk26, mk25, ch in ((f"{gname} 1위칸", m1, cell_mask(H.C, top1), chain_cell),
                                   (f"{gname} 나무", tmask, tmask25, chain_tree)):
            hyp_rows.append(confirm_rows(H if ch else G, mk25 if ch else mk26, 0 if ch else 1, hk,
                                         "2025" if ch else "2026"))
        print(gname, "자격", int(el.sum()), "1위", describe(top1), "뒤", hyps[f"{gname} 1위칸"]["back"], flush=True)
    ps = {}
    for k, h in hyps.items():
        src = h["y2025"] if h["chain"] else h["back"]
        ps[k] = max(src["p_r1n"], src["p_bt"])
    adj = holm(ps)
    for k in hyps:
        hyps[k]["p_hyp"] = ps[k]
        hyps[k]["p_holm"] = adj[k]
        hyps[k]["pass_stat"] = bool(adj[k] < 0.05)
    out["hyps"] = hyps
    json.dump(out, open(os.path.join(RES, "cond_primary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    pd.DataFrame(single_rows).to_csv(os.path.join(RES, "cond_single.csv"), index=False, encoding="utf-8-sig")
    pd.concat(hyp_rows, ignore_index=True).to_csv(os.path.join(RES, "cond_hyp_signals.csv"), index=False)
    pd.DataFrame(all_cells).to_parquet(os.path.join(E.CACHE, "cond_cells.parquet"))
    print("done")


if __name__ == "__main__":
    {"stage1": stage1, "e1": e1}[sys.argv[1]]()
