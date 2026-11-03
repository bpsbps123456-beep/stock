"""4차(BRIEF4.md): 하따봉 꼬리가 사용자 지표의 160분봉 지지·저항 구역(desk/zones.py)에 닿을 때.
사용: py zone_eval.py fetch    → cache/alp10m/<SYM>.parquet (Alpaca SIP 10분봉, 정규장, 2024-09-01~)
      py zone_eval.py flags    → cache/zone_flags.parquet (네 기간 ①·② 신호마다 구역 닿음 등)
      py zone_eval.py judge    → results/z4_primary.json, results/z4_tables.json, results/z4_signals.csv
      py zone_eval.py e1       → 2026 Z1·Z2 신호 실측 호가 손익"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
import engine as E
import cond_eval as CE
from fetch_alpaca import keys, Pace, URL, ROOT

sys.path.insert(0, ROOT)
from desk import zones as Z  # noqa: E402  사용자 지표 구역 규칙의 파이썬 정본

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
D10 = os.path.join(E.CACHE, "alp10m")
PERIODS = {"2025h1": "alp1m_2025h1", "2025": "alp1m_2025", "2026h1": "alp1m_2026h1", "2026": "alp1m"}
ENDS = (730, 890, 960)          # 160분봉 끝(12:10·14:50·16:00)
B_BOOT = 2000


def syms():
    return sorted(json.load(open(os.path.join(RES, "universe2.json"), encoding="utf-8")))


def fetch_one(sym, head, pace):
    path = os.path.join(D10, f"{sym}.parquet")
    if os.path.exists(path):
        return sym, -1
    rows, token = [], None
    p = {"symbols": sym, "timeframe": "10Min", "start": "2024-09-01T04:00:00Z", "end": "2026-10-29T04:00:00Z",
         "limit": 10000, "adjustment": "split", "feed": "sip", "sort": "asc"}
    while True:
        if token:
            p["page_token"] = token
        for attempt in range(8):
            pace.wait()
            try:
                r = requests.get(URL, headers=head, params=p, timeout=60)
            except requests.exceptions.RequestException:
                time.sleep(5 + 5 * attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 + 3 * attempt)
                continue
            r.raise_for_status()
            break
        j = r.json()
        rows += (j.get("bars") or {}).get(sym, [])
        token = j.get("next_page_token")
        if not token:
            break
    if not rows:
        return sym, 0
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d.pop("t"), utc=True).dt.tz_convert(E.ET)
    d = d.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})[["Open", "High", "Low", "Close", "Volume"]]
    m = d.index.hour * 60 + d.index.minute
    d = d[(m >= 570) & (m < 960)]
    d = d[~d.index.duplicated()].sort_index()
    d.to_parquet(path)
    return sym, len(d)


def fetch():
    os.makedirs(D10, exist_ok=True)
    head, pace = keys(), Pace(3.0)
    with ThreadPoolExecutor(8) as ex:
        for s, n in ex.map(lambda x: fetch_one(x, head, pace), syms()):
            print(s, n, flush=True)


def bars160_exact(d10):
    """정규장 10분봉 → 정확한 160분봉(09:30·12:10·14:50 시작) + 각 봉 끝 시각(유닉스 분)."""
    b = Z.bars160(d10, exact=True)
    day = pd.to_datetime(b["day"]).dt.tz_localize(E.ET)
    end = day + pd.to_timedelta([ENDS[int(g)] for g in b["g"]], unit="min")
    b["end_ts"] = ((end.dt.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy()
    return b


def touch(zs, L, C):
    """구역 아래×0.995 ≤ L ≤ 구역 위×1.001 이고 C ≥ 구역 아래×0.999인 구역(가장 위 구역), 없으면 None."""
    best = None
    for z in zs:
        lo, hi = z[0], z[1]
        if lo * 0.995 <= L <= hi * 1.001 and C >= lo * 0.999:
            if best is None or lo > best[0]:
                best = z
    return best


def sweep(zs, L, C):
    """구역 아래를 0.5~2% 뚫고 내려갔다가 구역 아래 위로 마감(보조)."""
    return any(z[0] * 0.98 <= L < z[0] * 0.995 and C >= z[0] * 0.999 for z in zs)


def flags():
    out = []
    t0 = time.time()
    for n, sym in enumerate(syms()):
        p10 = os.path.join(D10, f"{sym}.parquet")
        if not os.path.exists(p10):
            continue
        d10 = pd.read_parquet(p10)
        b = bars160_exact(d10)
        piv = Z.pivots(b)
        ends = b["end_ts"].to_numpy()
        # 전일 정규장 저가(10분봉): 날짜별 최저가
        dl = d10.groupby(d10.index.tz_localize(None).normalize()).Low.min()
        dl_days = (dl.index.year * 10000 + dl.index.month * 100 + dl.index.day).to_numpy()
        for per, src in PERIODS.items():
            cpath = os.path.join(E.CACHE, f"cond_{per}.parquet")
            R = pd.read_parquet(cpath, columns=["arm", "pat", "sym", "ts", "t", "k", "day"])
            R = R[(R.arm == "sig") & (R.sym == sym)]
            if len(R) == 0:
                continue
            d1 = E.load(sym, src)
            lo1, cl1 = d1.Low.to_numpy(), d1.Close.to_numpy()
            for r in R.itertuples():
                t, k = int(r.t), int(r.k)
                L = lo1[k]
                C = cl1[t]
                dec = int(np.searchsorted(ends, int(r.ts), side="right")) - 1
                rec = {"period": per, "sym": sym, "ts": int(r.ts), "pat": r.pat, "z_hit": False, "z_str": 0,
                       "z_near": False, "z_rank": -1, "z_sweep": False, "pdl_hit": False, "n_zones": 0}
                if dec >= 0:
                    ref = float(b["close"].iloc[dec])
                    sup, res, _ = Z.zones(piv, dec, ref)
                    zs = sup + res
                    rec["n_zones"] = len(zs)
                    z = touch(zs, L, C)
                    if z is not None:
                        rec["z_hit"] = True
                        rec["z_str"] = int(z[2])
                        rec["z_near"] = bool(sup) and z is sup[0]
                        rec["z_rank"] = next((i + 1 for i, s_ in enumerate(sup) if s_ is z), -1)   # 지지 몇 번째(보조)
                    rec["z_sweep"] = sweep(zs, L, C)
                i = int(np.searchsorted(dl_days, int(r.day))) - 1
                if i >= 0:
                    pdl = float(dl.iloc[i])
                    rec["pdl_hit"] = bool(pdl * 0.995 <= L <= pdl * 1.001 and C >= pdl * 0.999)
                out.append(rec)
        if n % 50 == 0:
            print(n, sym, f"{time.time() - t0:.0f}s", flush=True)
    F = pd.DataFrame(out)
    F.to_parquet(os.path.join(E.CACHE, "zone_flags.parquet"))
    print("done", len(F), F.groupby(["period", "pat"]).z_hit.mean().round(3).to_dict())


def load_all(cost):
    F = pd.read_parquet(os.path.join(E.CACHE, "zone_flags.parquet"))
    parts = []
    for per in PERIODS:
        R = CE.load(per, cost)
        R["period"] = per
        parts.append(R.merge(F[F.period == per].drop(columns="period"), on=["sym", "ts", "pat"], how="left"))
    A = pd.concat(parts, ignore_index=True)
    for c in ("z_hit", "z_near", "z_sweep", "pdl_hit"):
        A[c] = A[c].fillna(False).astype(bool)
    A["year"] = A.period.str[:4]
    return A


def dedup(X):
    X = X.sort_values(["sid", "ts"], kind="stable")
    return X[E.dedup(X.sid.to_numpy().astype(np.int64), X.ts.to_numpy().astype(np.int64), 30)]


def summ(X):
    if len(X) == 0:
        return {"n": 0}
    rc = X.race_e2.dropna()
    vb = (X.r1_e2 - X.bt2_r1).dropna()
    return {"n": int(len(X)), "days": int(X.day.nunique()), "syms": int(X.sid.nunique()), "hit": float(X.hit_e2.mean()),
            "r1": float(X.r1_e2.mean()), "r1n": float(X.r1n.mean()), "vs_bt": float(vb.mean()) if len(vb) else None,
            "race": float(rc.mean()) if len(rc) else None}


def boot_mean(v, day, rng):
    v, day = np.asarray(v, float), np.asarray(day)
    ok = ~np.isnan(v)
    v, day = v[ok], day[ok]
    days = np.unique(day)
    pos = np.searchsorted(days, day)
    s = np.bincount(pos, weights=v, minlength=len(days))
    n = np.bincount(pos, minlength=len(days)).astype(float)
    d = np.array([(s * w).sum() / max((n * w).sum(), 1) for w in
                  (np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)) for _ in range(B_BOOT))])
    return float(v.mean()), float((1 + np.sum(d <= 0)) / (B_BOOT + 1))


def paired(A_, B_, rng):
    days = np.union1d(A_.day.unique(), B_.day.unique())

    def agg(X):
        p = np.searchsorted(days, X.day.to_numpy())
        return np.bincount(p, weights=X.r1n.to_numpy(), minlength=len(days)), np.bincount(p, minlength=len(days)).astype(float)
    sa, na = agg(A_)
    sb, nb = agg(B_)
    d = [(sa * w).sum() / max((na * w).sum(), 1) - (sb * w).sum() / max((nb * w).sum(), 1)
         for w in (np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)) for _ in range(B_BOOT))]
    return {"diff": float(sa.sum() / na.sum() - sb.sum() / nb.sum()), "p": float((1 + np.sum(np.array(d) <= 0)) / (B_BOOT + 1))}


HYP = {"Z1": ("A", "① 꼬리 반전 + 160분봉 구역에 꼬리"), "Z2": ("B", "② 장악 양봉 + 이탈 봉 저가가 160분봉 구역")}


def judge():
    cost = json.load(open(os.path.join(RES, "cost_model.json"), encoding="utf-8"))
    A = load_all(cost)
    rng = np.random.default_rng(20261103)
    prim, tabs, sig = {}, {}, []
    for hk, (pat, label) in HYP.items():
        P26 = A[(A.year == "2026") & (A.pat == pat)]
        P25 = A[(A.year == "2025") & (A.pat == pat)]
        X, C = dedup(P26[P26.z_hit]), dedup(P26[~P26.z_hit])
        m1, p1 = boot_mean(X.r1n, X.day, rng)
        m2, p2 = boot_mean((X.r1_e2 - X.bt2_r1), X.day, rng)
        pr = paired(X, C, rng)
        prim[hk] = {"label": label, "y2026": summ(X), "no_zone_2026": summ(C), "p_r1n": p1, "p_bt": p2, "diff": pr,
                    "p_hyp": max(p1, p2, pr["p"]), "y2025": summ(dedup(P25[P25.z_hit])),
                    "no_zone_2025": summ(dedup(P25[~P25.z_hit])),
                    "by_period": {per: summ(dedup(A[(A.period == per) & (A.pat == pat) & A.z_hit])) for per in PERIODS}}
        sig.append(X.assign(hyp=hk)[["hyp", "period", "sym", "sid", "ts", "day", "t", "r1_e2", "r1n", "hit_e2"]])
        t = {}
        for nm, mk in (("가장 가까운 구역(Z1)", lambda D: D.z_hit & D.z_near),
                       ("화면에 보이는 지지 3개 안(Z1~Z3, 보조 추가)", lambda D: D.z_hit & D.z_rank.between(1, 3)),
                       ("시험 2회 이상 구역", lambda D: D.z_hit & (D.z_str >= 2)),
                       ("구역 아래를 0.5~2% 쓸고 회복", lambda D: D.z_sweep), ("전일 정규장 저가", lambda D: D.pdl_hit)):
            t[nm] = {"2026": summ(dedup(P26[mk(P26)])), "2025": summ(dedup(P25[mk(P25)]))}
        if pat == "A":
            star = lambda D: D.z_hit & (D.vwap_gap < -0.03)
            t["①★ + 구역"] = {"2026": summ(dedup(P26[star(P26)])), "2025": summ(dedup(P25[star(P25)]))}
        X2 = X.assign(month=X.day // 100)
        t["달별"] = {str(m): summ(g) for m, g in X2.groupby("month")}
        X2["tier"] = X2.cap_b.map(lambda c: "<50억$" if c < 5 else ("50억~200억$" if c < 20 else ("200억~1000억$" if c < 100 else "1000억$+")))
        t["시총"] = {k: summ(g) for k, g in X2.groupby("tier")}
        t["구역 비율(겹침 빼기 전)"] = {per: float(A[(A.period == per) & (A.pat == pat)].z_hit.mean()) for per in PERIODS}
        tabs[hk] = t
    order = sorted(prim, key=lambda k: prim[k]["p_hyp"])
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, prim[k]["p_hyp"] * (len(order) - i)))
        prim[k]["p_holm"] = run
        prim[k]["pass_stat"] = bool(run < 0.05)
        prim[k]["aux_pos"] = bool(prim[k]["y2025"].get("r1n", -1) > 0)
    json.dump(prim, open(os.path.join(RES, "z4_primary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    json.dump(tabs, open(os.path.join(RES, "z4_tables.json"), "w"), ensure_ascii=False, indent=1, default=float)
    pd.concat(sig).to_csv(os.path.join(RES, "z4_signals.csv"), index=False)
    for k, v in prim.items():
        print(k, v["label"], v["y2026"], "구역 없는 같은 신호", v["no_zone_2026"].get("r1n"), "차이", v["diff"],
              "p", round(v["p_r1n"], 3), round(v["p_bt"], 3), "Holm", round(v["p_holm"], 3))


def e1():
    from fetch_quotes import last_quote
    S = pd.read_csv(os.path.join(RES, "z4_signals.csv"))
    head, pace = keys(), Pace(3.0)

    def one(r):
        t = pd.Timestamp(int(r[1]) * 60, unit="s", tz="UTC")
        ai, bi = last_quote(r[0], t + pd.Timedelta(seconds=70), head, pace)
        ao, bo = last_quote(r[0], t + pd.Timedelta(minutes=31), head, pace)
        return ai, bi, ao, bo

    with ThreadPoolExecutor(6) as ex:
        q = list(ex.map(one, S[["sym", "ts"]].itertuples(index=False, name=None)))
    S[["ask_in", "bid_in", "ask_out", "bid_out"]] = np.array(q, dtype=float)
    bars, r = {}, []
    for x in S.itertuples():
        src = PERIODS[x.period]
        if (x.sym, src) not in bars:
            bars[(x.sym, src)] = E.prep(E.load(x.sym, src))
        a = bars[(x.sym, src)]
        t = int(x.t)
        if not (x.ask_in > 0):
            r.append(np.nan)
            continue
        z = t + 1
        while z + 1 < len(a["ts"]) and a["ts"][z + 1] <= a["ts"][t] + 30 and a["day"][z + 1] == a["day"][t]:
            z += 1
        tg = x.ask_in * 1.01
        hit = a["c"][t + 1] >= tg or (z >= t + 2 and np.max(a["h"][t + 2:z + 1]) >= tg)
        r.append(0.01 if hit else (x.bid_out if x.bid_out > 0 else a["c"][z]) / x.ask_in - 1)
    S["r1_e1"] = r
    S.to_csv(os.path.join(RES, "z4_signals_e1.csv"), index=False)
    P = json.load(open(os.path.join(RES, "z4_primary.json"), encoding="utf-8"))
    rng = np.random.default_rng(1103)
    for k, g in S.groupby("hyp"):
        m, p = boot_mean(g.r1_e1, g.day, rng)
        P[k]["e1"] = {"n": int(g.r1_e1.notna().sum()), "r1_e1": m, "p": p, "hit_e1": float((g.r1_e1 == 0.01).mean())}
        P[k]["accept"] = bool(P[k]["pass_stat"] and m > 0 and P[k]["aux_pos"])
        print(k, P[k]["e1"], "채택", P[k]["accept"])
    json.dump(P, open(os.path.join(RES, "z4_primary.json"), "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    {"fetch": fetch, "flags": flags, "judge": judge, "e1": e1}[sys.argv[1]]()
