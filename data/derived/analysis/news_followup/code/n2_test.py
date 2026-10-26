"""N2 평가: 금리 단서 R1·R2·M × type 2 × 결과 3(y1·y10·y20) = 18칸, 주식 사건(2026)과 SOXX 긴 이력 각각(BRIEF.md N2).

py code/n2_test.py → results/n2_cells.csv · n2_terciles.csv · n2_family.json · n2_verdict.json · n2_tables.txt

평가는 swing_trend BRIEF2·c2single_test.py와 같다(Blocks·cuts_of·groups·mdiff를 그대로 가져온다).
- 3분위 경계: 고르기 구간 사건(같은 type)으로만 정한다. 위 = f > 2/3 경계, 아래 = f ≤ 1/3 경계. M(0/1)은 1 − 0.
- 차이 = 위 − 아래 평균 결과(로그). 단순 %p 판: ys = t × (exp(t × y) − 1).
- 통과: s = sign(고르기 차이), T = s × 확인 차이 > 0 이고 T > 귀무 s × 확인 차이 95분위.
- 주 귀무(날짜 묶음 원형 이동): 같은 구간·type 사건 날짜 U(D개)를 k칸 민다. k = 1 … D−1 전 범위(type마다).
  주식은 k마다 같은 종목 맞추기·무작위 추출 R_DRAW회. SOXX는 날짜마다 사건 1개라 k마다 1회.
- 보조 귀무: (가) 주식 k 15 … D−15(앞 단계 범위), (나) 주식 날짜 블록 순열 N_PERM회, (다) SOXX 금리 시계열 원형 이동
  (^TNX 일봉 전체를 k거래일 밀어 R1·R2·M을 다시 계산, 가격·사건·결과는 그대로).
- 계열 귀무: 귀무 반복마다 18칸 통과 수를 같은 문턱으로 센다. 두 자료 동시 통과(같은 방향) 수는 두 귀무 반복을 짝지어 센다.
- 민감도: 금리 하루 늦춤(R1L·R2·ML), 잘린 사건 제외(확인 구간 끝에서 반대 확정이 없는 사건).

구간: 주식 고르기 P1(2026-01~05) → 확인 P2(2026-06~10), 참고 Y25·Y24·Y23.
      SOXX 고르기 L1(2003~2013) → 확인 L2(2014~2023), 표시 L3(2024~2026-10-23).
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / "swing_trend" / "code"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from c2single_test import Blocks, cuts_of, groups, mdiff  # noqa: E402

RES = HERE / "results"
OUTS = ["y1", "y10", "y20"]
TYPES = (1, -1)
R_DRAW, N_PERM, KMIN_OLD = 20, 200, 15
SEED = 20261026
BASE = ["R1", "R2", "M"]
LAG = ["R1L", "R2", "ML"]


def simple(sub, y):
    t = sub.type.values
    return t * (np.exp(t * y) - 1)


def stats(sub, feats, cuts, m=None, simple_y=False):
    out = np.full((len(feats), len(OUTS)), np.nan)
    for i, f in enumerate(feats):
        x = sub[f].values.astype(float)
        if m is not None:
            x = x[m]
        top, bot = groups(x, cuts[f])
        for j, o in enumerate(OUTS):
            y = sub[o].values.astype(float)
            if simple_y:
                y = simple(sub, y)
            out[i, j] = mdiff(y, top, bot)
    return out


def event_null(sub, feats, cuts, ks, draws, rng, perm=0):
    """날짜 묶음 귀무. 반환 (반복, 단서, 결과)."""
    b = Blocks(sub)
    reps = [(k, r) for k in ks for r in range(draws)] if not perm else [(None, r) for r in range(perm)]
    arr = np.full((len(reps), len(feats), len(OUTS)), np.nan)
    for ri, (k, r) in enumerate(reps):
        target = (np.arange(b.D) + k) % b.D if k is not None else rng.permutation(b.D)
        arr[ri] = stats(sub, feats, cuts, b.mapping(target, rng))
    return arr


def tnx_shift_null(xe_sub, tnx, sx_fall, ks, feats, cuts):
    """SOXX: 금리 시계열을 k거래일 원형 이동해 단서를 다시 계산."""
    c = xe_sub.conf_idx.values
    p = c - xe_sub.S1.values
    pp = xe_sub.pp_idx.values
    arr = np.full((len(ks), len(feats), len(OUTS)), np.nan)
    for ri, k in enumerate(ks):
        tk = np.roll(tnx, k)
        s = xe_sub.copy()
        r1 = tk[c] - tk[c - 20]
        r1l = tk[c - 1] - tk[c - 21]
        m = (sx_fall[c] & ((tk[c] - tk[c - 5]) < 0)).astype(float)
        ml = (sx_fall[c] & ((tk[c - 1] - tk[c - 6]) < 0)).astype(float)
        s["R1"], s["R1L"], s["M"], s["ML"], s["R2"] = r1, r1l, m, ml, tk[p] - tk[pp]
        arr[ri] = stats(s, feats, cuts)
    return arr


def judge(act_sel, act_conf, null):
    """act_*: (F, O). null: (R, F, O). 반환 sign, T, q95, p, pass, null_pass(R, F, O)."""
    s = np.nan_to_num(np.sign(act_sel))      # 고르기 구간에서 한쪽 무리가 비면 방향 없음(0) → 판정 불가
    T = s * act_conf
    Tn = s[None] * null
    q95 = np.nanquantile(Tn, 0.95, axis=0)
    p = np.nanmean(Tn >= T[None], axis=0)
    q95[s == 0] = np.nan
    p[(s == 0) | np.isnan(T)] = np.nan
    ok = (s != 0) & (T > 0) & (T > q95)
    npass = (s[None] != 0) & (Tn > 0) & (Tn > q95[None])
    return dict(sign=s, T=T, q95=q95, p=p, ok=ok, npass=npass)


def episodes(dates, cal_idx, gap=5):
    """날짜들을 세션 간격 gap 이하로 이어 붙인 국면 수."""
    if len(dates) == 0:
        return 0
    ii = np.sort(np.unique([cal_idx[d] for d in dates]))
    return int(1 + (np.diff(ii) > gap).sum())


def run_set(ev, sel, conf, refs, feats, rng, kind, extra_null=None):
    """한 자료(주식/SOXX)·한 단서 묶음 평가."""
    ev = ev.copy()
    subs = {(per, t): ev[(ev.period == per) & (ev.type == t)].reset_index(drop=True) for per in [sel, conf] + refs for t in TYPES}
    cuts = {t: {f: cuts_of(subs[(sel, t)][f].values.astype(float)) for f in feats} for t in TYPES}
    act = {(per, t): stats(subs[(per, t)], feats, cuts[t]) for (per, t) in subs}
    act_s = {(per, t): stats(subs[(per, t)], feats, cuts[t], simple_y=True) for (per, t) in subs}
    nulls = {}
    for t in TYPES:
        sub = subs[(conf, t)]
        D = sub.date.nunique()
        draws = R_DRAW if kind == "stock" else 1
        nulls[("full", t)] = event_null(sub, feats, cuts[t], range(1, D), draws, rng)
        if kind == "stock":
            nulls[("k15", t)] = event_null(sub, feats, cuts[t], range(KMIN_OLD, D - KMIN_OLD + 1), draws, rng)
            nulls[("perm", t)] = event_null(sub, feats, cuts[t], None, 1, rng, perm=N_PERM)
            subr = subs[("Y25", t)]
            nulls[("Y25full", t)] = event_null(subr, feats, cuts[t], range(1, subr.date.nunique()), 5, rng)
        if extra_null is not None:
            nulls[("tnxshift", t)] = extra_null(sub, cuts[t], feats)
    J = {}
    for nk in sorted({k for k, _ in nulls}):
        cper = "Y25" if nk == "Y25full" else conf
        J[nk] = {t: judge(act[(sel, t)], act[(cper, t)], nulls[(nk, t)]) for t in TYPES}
    # 칸 표
    rows = []
    for t in TYPES:
        sub2 = subs[(conf, t)]
        for i, f in enumerate(feats):
            top, bot = groups(sub2[f].values.astype(float), cuts[t][f])
            for j, o in enumerate(OUTS):
                okk = ~np.isnan(sub2[o].values.astype(float))
                unc = ~sub2.censored.values
                y = sub2[o].values.astype(float)
                row = dict(set=kind, feats="lag" if feats == LAG else "base", type=t, feature=f, outcome=o,
                           cut_lo=cuts[t][f][0], cut_hi=cuts[t][f][1], binary=cuts[t][f][2],
                           sel_n_top=int(groups(subs[(sel, t)][f].values.astype(float), cuts[t][f])[0].sum()),
                           sel_n_bot=int(groups(subs[(sel, t)][f].values.astype(float), cuts[t][f])[1].sum()),
                           conf_n_top=int((top & okk).sum()), conf_n_bot=int((bot & okk).sum()),
                           conf_dates_top=int(sub2.date[top & okk].nunique()), conf_dates_bot=int(sub2.date[bot & okk].nunique()),
                           conf_diff_uncensored=mdiff(np.where(unc, y, np.nan), top, bot))
                for per in [sel, conf] + refs:
                    row[f"{per}_diff"] = act[(per, t)][i, j]
                    row[f"{per}_diff_simple"] = act_s[(per, t)][i, j]
                ti = TYPES.index(t)
                for nk, jj in J.items():
                    g = jj[t]
                    row[f"q95_{nk}"] = g["q95"][i, j]
                    row[f"p_{nk}"] = g["p"][i, j]
                    row[f"pass_{nk}"] = bool(g["ok"][i, j])
                row["testable"] = bool(np.nan_to_num(np.sign(act[(sel, t)][i, j])) != 0 and not np.isnan(act[(conf, t)][i, j]))
                for per in refs:
                    s = np.sign(act[(sel, t)][i, j])
                    row[f"{per}_same_sign"] = bool(s != 0 and np.sign(act[(per, t)][i, j]) == s)
                rows.append(row)
    cells = pd.DataFrame(rows)
    # 계열 귀무(18칸 통과 수)
    fam = {}
    for nk, jj in J.items():
        n1, n2 = len(jj[1]["npass"]), len(jj[-1]["npass"])
        R = max(n1, n2)
        cnt = np.array([jj[1]["npass"][r % n1].sum() + jj[-1]["npass"][r % n2].sum() for r in range(R)])
        a = int(jj[1]["ok"].sum() + jj[-1]["ok"].sum())
        fam[nk] = dict(actual=a, cells=2 * len(feats) * len(OUTS), reps=R, null_mean=float(cnt.mean()),
                       null_q95=float(np.quantile(cnt, 0.95)), null_max=int(cnt.max()), p=float(np.mean(cnt >= a)))
    # 3분위 평균(표시)
    trows = []
    for (per, t), sub in subs.items():
        for f in feats:
            x = sub[f].values.astype(float)
            a, b, binary = cuts[t][f]
            lab = np.where(x == 1, "1", "0") if binary else np.where(x <= a, "아래", np.where(x > b, "위", "가운데"))
            for gname in (["0", "1"] if binary else ["아래", "가운데", "위"]):
                s_ = sub[lab == gname]
                trows.append(dict(set=kind, feats="lag" if feats == LAG else "base", period=per, type=t, feature=f, group=gname,
                                  n=len(s_), dates=s_.date.nunique(),
                                  **{f"{o}_simple": float(np.nanmean(simple(s_, s_[o].values))) if len(s_) else np.nan for o in OUTS},
                                  v_rate=float((s_.y1 < 0).mean()) if len(s_) else np.nan))
    terc = pd.DataFrame(trows)
    return dict(cells=cells, fam=fam, terc=terc, J=J, cuts=cuts, subs=subs)


def joint(stock, soxx, nk_stock="full", nk_soxx="full"):
    """두 자료 모두 통과 + 같은 방향(고르기 부호 같음)."""
    out = np.zeros((2, len(BASE), len(OUTS)), bool)
    nulls = []
    for ti, t in enumerate(TYPES):
        a, b = stock["J"][nk_stock][t], soxx["J"][nk_soxx][t]
        same = (a["sign"] == b["sign"]) & (a["sign"] != 0)
        out[ti] = a["ok"] & b["ok"] & same
        nulls.append((a["npass"], b["npass"], same))
    R = max(max(len(x[0]), len(x[1])) for x in nulls)
    cnt = np.zeros(R)
    for r in range(R):
        for na, nb, same in nulls:
            cnt[r] += (na[r % len(na)] & nb[r % len(nb)] & same).sum()
    same_dir = int(sum(x[2].sum() for x in nulls))
    a = int(out.sum())
    return out, dict(actual=a, cells=18, same_sign_selection_cells=same_dir, reps=R, null_mean=float(cnt.mean()),
                     null_q95=float(np.quantile(cnt, 0.95)), null_max=int(cnt.max()), p=float(np.mean(cnt >= a)))


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.{d}f}"


def main():
    rng = np.random.default_rng(SEED)
    se = pd.read_parquet(RES / "n2_events_stock.parquet")
    xe = pd.read_parquet(RES / "n2_events_soxx.parquet")
    for d in (se, xe):
        d["date"] = pd.to_datetime(d.date).dt.date
    # SOXX 금리 시계열 이동 귀무 준비
    raw = pd.read_parquet(HERE / "cache" / "n2_prices_raw.parquet")
    adj = pd.read_parquet(HERE / "cache" / "n2_prices_adj.parquet")
    for x in (raw, adj):
        x["date"] = pd.to_datetime(x.date)
    cal = pd.DatetimeIndex(adj[adj.sym == "SOXX"].sort_values("date").date)
    tnx = raw[raw.sym == "^TNX"].set_index("date").C.reindex(cal).ffill().values
    sxr = raw[raw.sym == "SOXX"].set_index("date").C.reindex(cal)
    sx_fall = ((sxr / sxr.shift(5) - 1) < 0).values
    pidx = {d: i for i, d in enumerate(cal.date)}
    xe["pp_idx"] = [pidx[d] for d in xe.prev_pivot_date]
    assert all(pidx[d] == c - s for d, c, s in zip(xe.pivot_date, xe.conf_idx, xe.S1))
    n = len(cal)
    ks_ts = list(range(126, n - 126, 5))

    def ts_null(sub, cuts_t, feats):
        return tnx_shift_null(sub, tnx, sx_fall, ks_ts, feats, cuts_t)

    res = {}
    for feats, tag in ((BASE, "base"), (LAG, "lag")):
        res[("stock", tag)] = run_set(se, "P1", "P2", ["Y25", "Y24", "Y23"], feats, rng, "stock")
        res[("soxx", tag)] = run_set(xe, "L1", "L2", ["L3"], feats, rng, "soxx", extra_null=ts_null)
    cells = pd.concat([r["cells"] for r in res.values()], ignore_index=True)
    terc = pd.concat([r["terc"] for r in res.values()], ignore_index=True)
    cells.to_csv(RES / "n2_cells.csv", index=False, encoding="utf-8-sig")
    terc.to_csv(RES / "n2_terciles.csv", index=False, encoding="utf-8-sig")
    fam = {f"{a}_{b}": r["fam"] for (a, b), r in res.items()}
    jt = {}
    for tag in ("base", "lag"):
        for nks, nkx in (("full", "full"), ("full", "tnxshift"), ("k15", "full"), ("perm", "full")):
            ok, js = joint(res[("stock", tag)], res[("soxx", tag)], nks, nkx)
            js["passing_cells"] = [f"{'매수' if TYPES[ti] == 1 else '매도'} {BASE[i] if tag == 'base' else LAG[i]} {OUTS[j]}"
                                   for ti, i, j in zip(*np.nonzero(ok))]
            jt[f"{tag}_stock-{nks}_soxx-{nkx}"] = js
    fam["joint"] = jt
    (RES / "n2_family.json").write_text(json.dumps(fam, ensure_ascii=False, indent=1), encoding="utf-8")

    # 사건·날짜·국면 수(확인 구간 위·아래)
    scal = sorted(set(pd.to_datetime(raw[raw.sym == "SPY"].date).dt.date))
    sidx = {d: i for i, d in enumerate(scal)}
    epi = []
    for (kind, tag), r in res.items():
        if tag != "base":
            continue
        conf = "P2" if kind == "stock" else "L2"
        for t in TYPES:
            sub = r["subs"][(conf, t)]
            for f in BASE:
                top, bot = groups(sub[f].values.astype(float), r["cuts"][t][f])
                epi.append(dict(set=kind, type=t, feature=f, top_events=int(top.sum()), top_dates=int(sub.date[top].nunique()),
                                top_episodes=episodes(sub.date[top].values, sidx), bot_events=int(bot.sum()),
                                bot_dates=int(sub.date[bot].nunique()), bot_episodes=episodes(sub.date[bot].values, sidx)))
    epi = pd.DataFrame(epi)
    epi.to_csv(RES / "n2_episodes.csv", index=False, encoding="utf-8-sig")

    # 판정
    b = cells[cells.feats == "base"]
    st = b[b.set == "stock"].dropna(axis=1, how="all").reset_index(drop=True)
    sx = b[b.set == "soxx"].dropna(axis=1, how="all").reset_index(drop=True)
    key = ["type", "feature", "outcome"]
    mg = st.merge(sx, on=key, suffixes=("_st", "_sx"))
    mg["same_dir_conf"] = np.sign(mg.P2_diff) == np.sign(mg.L2_diff)
    mg["N2_pass"] = mg.pass_full_st & mg.pass_full_sx & (np.sign(mg.P1_diff) == np.sign(mg.L1_diff))
    lag = cells[(cells.feats == "lag")]
    verdict = dict(
        cells=18,
        stock_P2_pass_full=int(st.pass_full.sum()), stock_family=fam["stock_base"]["full"],
        soxx_L2_pass_full=int(sx.pass_full.sum()), soxx_family=fam["soxx_base"]["full"],
        soxx_L2_pass_tnxshift=int(sx.pass_tnxshift.sum()), soxx_family_tnxshift=fam["soxx_base"]["tnxshift"],
        same_direction_P2_vs_L2=int(mg.same_dir_conf.sum()),
        same_direction_P1_vs_L1=int((np.sign(mg.P1_diff) == np.sign(mg.L1_diff)).sum()),
        testable_stock=int(st.testable.sum()), testable_soxx=int(sx.testable.sum()), testable_both=int((mg.testable_st & mg.testable_sx).sum()),
        N2_pass=int(mg.N2_pass.sum()), N2_pass_cells=mg[mg.N2_pass][key].to_dict("records"),
        joint_family=jt["base_stock-full_soxx-full"],
        lag_stock_pass=int(lag[lag.set == "stock"].pass_full.sum()), lag_soxx_pass=int(lag[lag.set == "soxx"].pass_full.sum()),
        lag_joint=jt["lag_stock-full_soxx-full"]["actual"],
        stock_Y25_pass_cells=st[st.pass_Y25full][key + ["P1_diff_simple", "Y25_diff_simple", "p_Y25full"]].to_dict("records"),
        lag_pass_cells=lag[lag.pass_full][["set"] + key + ["P2_diff_simple", "Y25_diff_simple", "conf_diff_uncensored"]].to_dict("records"),
        stock_pass_cells=st[st.pass_full][key + ["P1_diff_simple", "P2_diff_simple", "p_full", "Y25_same_sign"]].to_dict("records"),
        soxx_pass_cells=sx[sx.pass_full][key + ["L1_diff_simple", "L2_diff_simple", "p_full", "p_tnxshift", "L3_same_sign"]].to_dict("records"),
    )
    verdict["N2"] = "통과" if verdict["N2_pass"] > 0 and verdict["joint_family"]["p"] < 0.05 else "실패"
    (RES / "n2_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    mg.to_csv(RES / "n2_joint_cells.csv", index=False, encoding="utf-8-sig")

    # 표
    L = []
    for (kind, tag), r in res.items():
        sel, conf = ("P1", "P2") if kind == "stock" else ("L1", "L2")
        refs = ["Y25", "Y24", "Y23"] if kind == "stock" else ["L3"]
        c = r["cells"]
        ev = r["subs"]
        L.append(f"=== {kind} · {tag} ({'금리 하루 늦춤' if tag == 'lag' else '확정일 종가'}) ===")
        L.append("사건(개/날짜): " + ", ".join(f"{p}_{t} {len(s)}/{s.date.nunique()}" for (p, t), s in ev.items()))
        for nk, f in r["fam"].items():
            L.append(f"  계열[{nk}] 통과 {f['actual']}/{f['cells']} | 귀무 평균 {f['null_mean']:.2f} · 95분위 {f['null_q95']:.0f} · 최대 {f['null_max']} · p {f['p']:.3f} (반복 {f['reps']})")
        for t in TYPES:
            L.append(f"-- {'매수 확정' if t == 1 else '매도 확정'} · 위 − 아래, 단순 %p ({sel}·{conf}·{'·'.join(refs)}) · {conf} 로그 차이 / 귀무 95분위(full) / p · 통과 full·k15·perm·tnx --")
            for _, x in c[c.type == t].iterrows():
                marks = "".join(("✓" if x.get(f"pass_{nk}", False) else "·") for nk in ("full", "k15", "perm", "tnxshift") if f"pass_{nk}" in x)
                L.append(f"  {x.feature:3} {x.outcome:3} | {pct(x[f'{sel}_diff_simple']):>6} {pct(x[f'{conf}_diff_simple']):>6} "
                         + " ".join(f"{pct(x[f'{p}_diff_simple']):>6}" for p in refs)
                         + f" | {pct(x[f'{conf}_diff']):>6} {pct(x.q95_full):>6} p {x.p_full:.2f} | {marks} | 날짜 위/아래 {x.conf_dates_top}/{x.conf_dates_bot} | 잘림 제외 {pct(x.conf_diff_uncensored)}")
        L.append("")
    L.append("-- 3분위 평균 단순 % (y1 · y20) [사건/날짜], base --")
    tb = terc[terc.feats == "base"]
    for kind, pers in (("stock", ["P1", "P2", "Y25"]), ("soxx", ["L1", "L2", "L3"])):
        for t in TYPES:
            for f in BASE:
                parts = []
                for per in pers:
                    q = tb[(tb.set == kind) & (tb.type == t) & (tb.feature == f) & (tb.period == per)]
                    parts.append(f"{per}: " + " · ".join(f"{x.group} {pct(x.y1_simple)}/{pct(x.y20_simple)} [{x.n}/{x.dates}]" for _, x in q.iterrows()))
                L.append(f"  {kind} {t:+d} {f:3} | " + " | ".join(parts))
    L.append("")
    L.append("-- 확인 구간 위·아래 3분위 국면 수(세션 5일 넘게 떨어지면 새 국면) --")
    L.append(epi.to_string(index=False))
    L.append("")
    L.append("-- 동시 통과(주식 P2 · SOXX L2, 같은 방향) --")
    for k, v in jt.items():
        L.append(f"  {k}: {v['actual']}/18 (고르기 방향 같은 칸 {v['same_sign_selection_cells']}) | 귀무 평균 {v['null_mean']:.3f} · 95분위 {v['null_q95']:.0f} · 최대 {v['null_max']} · p {v['p']:.3f} | {', '.join(v['passing_cells']) or '없음'}")
    L.append(f"판정 N2: {verdict['N2']} (동시 통과 {verdict['N2_pass']}/18, P2·L2 같은 방향 {verdict['same_direction_P2_vs_L2']}/18, P1·L1 같은 방향 {verdict['same_direction_P1_vs_L1']}/18)")
    txt = "\n".join(L)
    (RES / "n2_tables.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
