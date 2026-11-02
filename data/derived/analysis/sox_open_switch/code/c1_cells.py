"""BRIEF4 C1 조건 표 전수 — 한 겹·두 겹 모든 칸을 2026 전체에서 재고 Romano-Wolf 단계 하강으로 가른다.

정본: ../BRIEF4.md 'C1 조건 표 전수'·'원리'. 사건 표: ../results/ev4_events.parquet(ev4_build.py).
- 칸 = 면(2) × 문턱(4) × [조건 하나의 수준 하나 | 서로 다른 조건 둘의 수준 짝]. 수준은 표의 *_lv 열(결측 = 그 조건 칸에서 빠짐).
- 통계량: 2026 사건의 r_w90 평균 ÷ 날 묶음 표준오차(CR1: G/(G−1) · Σ_d (s_d − m·n_d)² / N², G = 사건 있는 날 수).
  [강제 변경, 결과를 본 뒤] r_w90은 +1% 도달이 정확히 1.0인 점 질량이라 거의 모두 도달한 칸은 자기 표준오차가 0에 가깝다.
  그대로(literal) 하면 귀무 최대 t가 2,000회 중 1,968회 +inf라 어떤 칸도 통과할 수 없는 퇴화 검정이 된다.
  주 판(floor)은 se = max(자기 CR1 se, σ_g/√N)(σ_g = 같은 면·문턱 2026 매매당 표준편차)로 관측·부트스트랩 모두 같은 식을 쓴다
  (표준오차를 키우기만 하므로 보수적). literal·pooled(σ_g/√N만) 판을 함께 낸다.
- 자격: 2026 사건 ≥ 25, 날 ≥ 20.
- Romano-Wolf: 2026 거래일 207일을 복원 추출 2,000회(seed 20261102). 칸마다 t* = (m* − m̂) / se*(귀무로 옮긴 스튜던트화).
  단측(위쪽) 최대 t. 칸을 관측 t 내림차순으로 놓고 뒤에서부터 덩어리로 돌며 반복마다 '남은 칸 집합의 최대'(접미 최대)만 이어 간다
  → 단계 하강 p̃_k = (1 + #{b: max_{l≥k} t*_l,b ≥ t_k}) / (B + 1), 보정 p = 앞 순위까지의 누적 최대.
  t* 계산 불가(N* = 0 또는 G* < 2, 0/0)는 −inf, se* = 0이고 m* > m̂이면 +inf(보수적).
- BH-FDR q 0.10(탐색 단서): 칸별 단측 p = 1 − T_{G−1}(t).
- 보정 p ≤ 0.05 칸에는 BRIEF4 '세 갈래 모두 통과하면' 추가 조건(2026 1~6월·7~10월 > 0, 가장 좋은 달 빼도 > 0, 비용 2배 > 0)과
  2023·2024·2025 같은 칸 값, r_w90_s2·r_std 값을 붙인다. 넷 다 넘으면 survivors.
손익 단위 %(1.0 = 1%). 실행: py c1_cells.py → ../results/c1_cells.csv, ../results/c1.json
"""
import json, os, time

import numpy as np
import pandas as pd
from scipy import stats

os.environ["SOS_UNLOCK"] = "frozen"        # BRIEF4: 2023~2026 전체 사용(이 스크립트는 lib 엔진을 부르지 않는다)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
SEED, B = 20261102, 2000
MIN_N, MIN_G = 25, 20
ALPHA, Q_BH = 0.05, 0.10
CHUNK = 800

NAMES = {1: "신호 시각", 2: "오름 속도(L 봉→신호 봉 분)", 3: "오른 종목 시가 대비", 4: "오른 종목 전날 종가 대비",
         5: "갭(오른 종목 쪽 부호)", 6: "첫 5분 방향", 7: "VWAP 위 거리", 8: "1분 RSI14", 9: "RSI 약세 다이버전스",
         10: "신호 봉 윗꼬리 비율", 11: "신호 봉 거래량 ÷ 직전 20봉", 12: "오름 구간 거래량 ÷ L 앞 20봉",
         13: "15분 안 신고가 갱신 횟수", 14: "신호 봉 종가 = 장중 최고 종가", 15: "전날 고가 대비", 16: "프리장 고가 대비",
         17: "QQQ 같은 기준 오름", 18: "NVDA 같은 기준 오름", 19: "SOXX 오름 ÷ (R 오름/3)", 20: "전날 SOXL 수익",
         21: "전날 SOXL 고저폭 ÷ 60일 중앙값", 22: "전날 장초 90분 방향 효율", 23: "5거래일 SOXL 수익",
         24: "프리장 08:00→09:29 SOXL", 25: "요일", 26: "SOXL 전날 종가 ÷ 20일 평균",
         27: "5일 장초 90분 고저폭 ÷ 60일 중앙값", 28: "같은 날 같은 면 순번"}
SIDE = {"user": "사용자 방향(SOXL 오름 → SOXS 매수)", "mirror": "거울(SOXS 오름 → SOXL 매수)"}


def trading_days_2026():
    f = pd.read_parquet(os.path.join(ROOT, "cache", "alp1m_long", "SOXL.parquet"), columns=["Close"])
    m = f.index.hour * 60 + f.index.minute
    days = sorted({d for d, mm in zip(f.index.date, m) if 570 <= mm < 960})   # lib.Data와 같은 날 정의
    return [pd.Timestamp(d) for d in days if d.year == 2026]


def cell_stats(n, s, w=None):
    """n, s: (날, 칸) 사건 수·r 합. w: (반복, 날) 가중(없으면 1). 반환 N, m, se, t, G (반복 또는 1, 칸)."""
    if w is None:
        w = np.ones((1, n.shape[0]))
    N = w @ n
    S = w @ s
    A = w @ (s * s)
    Bm = w @ (s * n)
    C = w @ (n * n)
    G = w @ (n > 0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = S / N
        ss = A - 2 * m * Bm + m * m * C
        ss = np.where(ss < 0, 0.0, ss)                     # 반올림 음수 → 0
        se = np.sqrt(G / (G - 1) * ss) / N
    return N, m, se, G


def main():
    t0 = time.time()
    E = pd.read_parquet(os.path.join(RES, "ev4_events.parquet"))
    DKEY = np.array(pd.to_datetime(E["day"]).dt.date.values)
    meta = json.load(open(os.path.join(RES, "ev4_events_meta.json"), encoding="utf-8"))
    conds = [(c["no"], c["col"] + "_lv", c["levels"]) for c in meta["conditions"]]
    for no, col, lv in conds:                              # 수준 목록이 메타와 같은지
        assert list(E[col].cat.categories) == lv, (col, list(E[col].cat.categories), lv)

    # 한 겹 특징(조건, 수준) 75개
    feats = [(no, col, l) for no, col, lv in conds for l in lv]
    F = np.column_stack([(E[col] == l).fillna(False).values.astype(bool) for _, col, l in feats])
    fc = np.array([no for no, _, _ in feats])
    nf = len(feats)
    pairs = [(a, b) for a in range(nf) for b in range(a + 1, nf) if fc[a] != fc[b]]
    P1 = np.array([a for a, _ in pairs]); P2 = np.array([b for _, b in pairs])
    groups = [(s, x) for s in ["user", "mirror"] for x in [0.025, 0.03, 0.035, 0.04]]
    per_group = nf + len(pairs)

    # 칸 목록
    rows = []
    for gi, (s, x) in enumerate(groups):
        for a in range(nf):
            rows.append((gi, s, x, a, -1))
        for a, b in pairs:
            rows.append((gi, s, x, a, b))
    cells = pd.DataFrame(rows, columns=["g", "side", "x", "f1", "f2"])
    K = len(cells)

    def label(f):
        no, col, l = feats[f]
        return f"c{no:02d}", l

    cells["c1"] = [label(f)[0] for f in cells.f1]
    cells["lv1"] = [label(f)[1] for f in cells.f1]
    cells["c2"] = [label(f)[0] if f >= 0 else "" for f in cells.f2]
    cells["lv2"] = [label(f)[1] if f >= 0 else "" for f in cells.f2]
    cells["id"] = [f"{s}|x{x * 100:g}|{a}={la}" + (f"&{b}={lb}" if b else "")
                   for s, x, a, la, b, lb in zip(cells.side, cells.x, cells.c1, cells.lv1, cells.c2, cells.lv2)]

    def membership(ev_idx_by_group):
        """그룹별 사건 색인 → 칸 소속 행렬 목록(그룹 순서, (사건, 칸 per_group))."""
        out = []
        for gi in range(len(groups)):
            ix = ev_idx_by_group[gi]
            Fg = F[ix]
            M = np.concatenate([Fg, Fg[:, P1] & Fg[:, P2]], axis=1)
            out.append((ix, M))
        return out

    def day_aggs(sub, day_list):
        """사건 부분집합(sub: E 행 위치 배열)의 (날, 칸) n·s·hit 합."""
        dpos = {pd.Timestamp(d).date(): k for k, d in enumerate(day_list)}
        nd = len(day_list)
        n = np.zeros((nd, K)); s = np.zeros((nd, K)); h = np.zeros((nd, K))
        r = E["r_w90"].values; hit = E["hit"].values.astype(float)
        by_g = []
        for (sd, x) in groups:
            gmask = (E["side"].values[sub] == sd) & np.isclose(E["x"].values[sub], x)
            by_g.append(sub[gmask])
        for gi, (ix, M) in enumerate(membership(by_g)):
            if len(ix) == 0:
                continue
            dk = np.array([dpos[d] for d in DKEY[ix]])
            Dm = np.zeros((nd, len(ix))); Dm[dk, np.arange(len(ix))] = 1.0
            Mf = M.astype(float)
            sl = slice(gi * per_group, (gi + 1) * per_group)
            n[:, sl] = Dm @ Mf
            s[:, sl] = Dm @ (Mf * r[ix][:, None])
            h[:, sl] = Dm @ (Mf * hit[ix][:, None])
        return n, s, h

    # ---------- 2026 관측 ----------
    days26 = trading_days_2026()
    assert len(days26) == 207, len(days26)
    sub26 = np.where(E["year"].values == 2026)[0]
    n26, s26, h26 = day_aggs(sub26, days26)
    N, m, se_own, G = (a[0] for a in cell_stats(n26, s26))
    se_own = np.where(se_own < 1e-10, 0.0, se_own)          # 모두 +1%(도달)인 칸의 반올림 잔차 → 0
    # 그룹(면 × 문턱) 2026 매매당 표준편차 → 칸별 바닥 se = σ_g / √N (FLOOR, 아래 '강제 변경' 참고)
    sig_g = []
    for (sd_, x) in groups:
        e = E[(E["side"] == sd_) & np.isclose(E["x"], x) & (E["year"] == 2026)]
        sig_g.append(float(e["r_w90"].std(ddof=1)))
    sig = np.repeat(np.array(sig_g), per_group)

    def se_mode(se_, N_, mode):
        with np.errstate(divide="ignore", invalid="ignore"):
            pool = sig / np.sqrt(N_)
        if mode == "literal":
            return se_
        if mode == "floor":
            return np.maximum(se_, pool)
        return pool                                          # pooled

    elig = (N >= MIN_N) & (G >= MIN_G)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lit = m / se_own
        se = se_mode(se_own, N, "floor")
        t = m / se
        t_pool = m / se_mode(se_own, N, "pooled")
    cells["n"] = N.astype(int); cells["days"] = G.astype(int)
    cells["mean"] = m; cells["se_own"] = se_own; cells["se"] = se; cells["t"] = t
    cells["t_literal"] = t_lit; cells["t_pooled"] = t_pool; cells["sigma_group"] = sig
    cells["hit_rate"] = h26.sum(0) / np.where(N > 0, N, np.nan)
    cells["sum_per_day"] = s26.sum(0) / len(days26)      # 칸 사건 r 합 ÷ 207일(단순 합, 설명용)
    cells["eligible"] = elig
    cells["floor_binds"] = elig & (se_own < sig / np.sqrt(np.where(N > 0, N, np.nan)))
    nonfinite_obs = int((elig & ~np.isfinite(t_lit)).sum())
    print(f"cells {K}, eligible {int(elig.sum())}, literal non-finite t among eligible {nonfinite_obs}, "
          f"floor binds {int(cells.floor_binds.sum())}, {time.time() - t0:.1f}s", flush=True)

    # 단측 t 분포 p와 BH(주 판 = floor t)
    p_t = np.full(K, np.nan)
    p_t[elig] = stats.t.sf(t[elig], df=G[elig] - 1)
    cells["p_t1"] = p_t
    ei = np.where(elig)[0]
    order = np.argsort(p_t[ei], kind="mergesort")
    mm_ = len(ei)
    thr = Q_BH * np.arange(1, mm_ + 1) / mm_
    passed = p_t[ei][order] <= thr
    kmax = (np.where(passed)[0].max() + 1) if passed.any() else 0
    bh = np.zeros(K, bool); bh[ei[order[:kmax]]] = True
    cells["bh_q10"] = bh

    # ---------- Romano-Wolf 단계 하강 ----------
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(days26), size=(B, len(days26)))
    W = np.stack([np.bincount(r_, minlength=len(days26)) for r_ in draws]).astype(float)

    def rw(mode, t_obs):
        eo = ei[np.argsort(-t_obs[ei], kind="mergesort")]      # 자격 칸, 관측 t 내림차순
        t_sorted = t_obs[eo]
        p_tilde = np.zeros(len(eo)); p_marg = np.zeros(len(eo))
        carry = np.full(B, -np.inf)
        n_inf = 0; n_und = 0
        for st in reversed(range(0, len(eo), CHUNK)):
            sl = slice(st, min(st + CHUNK, len(eo)))
            cidx = eo[sl]
            Nb, mb, seb, Gb = cell_stats(n26[:, cidx], s26[:, cidx], W)
            seb = np.where(seb < 1e-10, 0.0, seb)
            if mode != "literal":
                with np.errstate(divide="ignore", invalid="ignore"):
                    pool = sig[cidx][None, :] / np.sqrt(Nb)
                seb = np.maximum(seb, pool) if mode == "floor" else pool
            with np.errstate(divide="ignore", invalid="ignore"):
                tb = (mb - m[cidx][None, :]) / seb
            bad = (Nb <= 0) | (Gb < 2) | np.isnan(tb)
            tb = np.where(bad, -np.inf, tb)
            n_inf += int(np.isposinf(tb).sum()); n_und += int(bad.sum())
            rc = np.maximum.accumulate(tb[:, ::-1], axis=1)[:, ::-1]
            suf = np.maximum(rc, carry[:, None])
            p_tilde[sl] = (1 + (suf >= t_sorted[sl][None, :]).sum(0)) / (B + 1)
            p_marg[sl] = (1 + (tb >= t_sorted[sl][None, :]).sum(0)) / (B + 1)
            carry = np.maximum(carry, tb.max(1))
        p_adj = np.maximum.accumulate(p_tilde)
        fin = np.isfinite(carry)
        meth = "linear" if fin.all() else "higher"
        q = {f"q{int(round(p * 100))}": float(np.quantile(carry, p, method=meth)) for p in [0.5, 0.9, 0.95, 0.99]}
        info = {"null_max_t_step1": q, "quantile_method": meth, "null_max_plus_inf_reps": int((~fin).sum()),
                "null_cells_x_reps_plus_inf": n_inf, "null_cells_x_reps_undefined": n_und,
                "max_obs_t": float(t_sorted[0]), "n_adj_le_0.05": int((p_adj <= ALPHA).sum()),
                "min_adj_p": float(p_adj.min())}
        print(mode, json.dumps(info, ensure_ascii=False), f"{time.time() - t0:.1f}s", flush=True)
        return eo, p_adj, p_marg, info

    rw_info = {}
    for mode, t_obs in [("floor", t), ("literal", t_lit), ("pooled", t_pool)]:
        eo, p_adj, p_marg, info = rw(mode, t_obs)
        rw_info[mode] = info
        suf_ = "" if mode == "floor" else f"_{mode}"
        cells["p_rw_adj" + suf_] = np.nan
        cells.loc[eo, "p_rw_adj" + suf_] = p_adj
        if mode == "floor":
            cells["p_boot_marg"] = np.nan; cells["rank"] = np.nan
            cells.loc[eo, "p_boot_marg"] = p_marg
            cells.loc[eo, "rank"] = np.arange(1, len(eo) + 1)
    q = rw_info["floor"]["null_max_t_step1"]

    # ---------- (사후, 설명용) 그룹 안 사건 결과 섞기 ----------
    # 거의 모두 +1%인 칸은 날 복원 추출이 그 칸의 우연 변동을 못 만든다(표본에 손실이 없으면 다시 뽑아도 없다).
    # 그래서 '조건이 결과와 무관'(그룹 안에서 사건 결과를 섞어도 같다)이라는 다른 귀무로, 전 자격 칸 최대가 우연히 나올 확률을 본다.
    # 통계량 두 개: 평균 z = (m_c − m_g)·√N_c/σ_g, 도달 z = (hit_c − p_g·N_c)/√(N_c·p_g·(1−p_g)). 사건 단위 섞기(날 묶음 무시 → 실제보다 관대).
    rngp = np.random.default_rng(SEED + 1)
    NP = 2000
    perm_max_mz = np.full(NP, -np.inf); perm_max_hz = np.full(NP, -np.inf)
    obs_mz = np.full(K, np.nan); obs_hz = np.full(K, np.nan)
    for gi, (sd_, x) in enumerate(groups):
        ix = sub26[(E["side"].values[sub26] == sd_) & np.isclose(E["x"].values[sub26], x)]
        Fg = F[ix]
        M = np.concatenate([Fg, Fg[:, P1] & Fg[:, P2]], axis=1).astype(float)
        sl = slice(gi * per_group, (gi + 1) * per_group)
        el = elig[sl]
        if not el.any():
            continue
        Me = M[:, el]
        Nc = Me.sum(0)
        rv = E["r_w90"].values[ix]; hv = E["hit"].values[ix].astype(float)
        mg, sg, pg = rv.mean(), rv.std(ddof=1), hv.mean()
        oz = ((Me.T @ rv) / Nc - mg) * np.sqrt(Nc) / sg
        ohz = ((Me.T @ hv) - pg * Nc) / np.sqrt(Nc * pg * (1 - pg))
        tmp = np.full(per_group, np.nan); tmp[el] = oz; obs_mz[sl] = tmp
        tmp = np.full(per_group, np.nan); tmp[el] = ohz; obs_hz[sl] = tmp
        PI = np.argsort(rngp.random((NP, len(ix))), axis=1)
        Rp = rv[PI].T; Hp = hv[PI].T                              # (사건, NP)
        zp = ((Me.T @ Rp) / Nc[:, None] - mg) * np.sqrt(Nc)[:, None] / sg
        hzp = ((Me.T @ Hp) - pg * Nc[:, None]) / np.sqrt(Nc * pg * (1 - pg))[:, None]
        perm_max_mz = np.maximum(perm_max_mz, zp.max(0)); perm_max_hz = np.maximum(perm_max_hz, hzp.max(0))
    cells["perm_mean_z"] = obs_mz; cells["perm_hit_z"] = obs_hz
    cells["perm_p_mean_z"] = (1 + (perm_max_mz[None, :] >= obs_mz[:, None]).sum(1)) / (NP + 1)
    cells["perm_p_hit_z"] = (1 + (perm_max_hz[None, :] >= obs_hz[:, None]).sum(1)) / (NP + 1)
    cells.loc[~elig, ["perm_p_mean_z", "perm_p_hit_z"]] = np.nan
    perm_info = {"perms": NP, "seed": SEED + 1, "unit": "그룹(면 × 문턱) 안 2026 사건 결과(r_w90·hit) 섞기, 사건 단위",
                 "null_max_mean_z": {f"q{q_}": float(np.quantile(perm_max_mz, q_ / 100)) for q_ in (50, 95)},
                 "null_max_hit_z": {f"q{q_}": float(np.quantile(perm_max_hz, q_ / 100)) for q_ in (50, 95)},
                 "obs_max_mean_z": float(np.nanmax(obs_mz)), "obs_max_hit_z": float(np.nanmax(obs_hz)),
                 "n_p_le_0.05_mean_z": int((cells["perm_p_mean_z"] <= 0.05).sum()),
                 "n_p_le_0.05_hit_z": int((cells["perm_p_hit_z"] <= 0.05).sum()),
                 "top_hit_z": [{"id": r.id, "n": int(r.n), "hit_rate": float(r.hit_rate), "mean": float(r.mean),
                                "hit_z": float(r.perm_hit_z), "p_max_hit_z": float(r.perm_p_hit_z),
                                "mean_z": float(r.perm_mean_z), "p_max_mean_z": float(r.perm_p_mean_z)}
                               for r in cells[elig].sort_values("perm_hit_z", ascending=False).head(5).itertuples(index=False)],
                 "note": "BRIEF4 밖 사후 점검(판정에 안 씀). 귀무가 'r_w90 평균 ≤ 0'이 아니라 '조건과 결과 무관'이다. 날 묶음을 무시해 p가 실제보다 작게(관대) 나온다."}
    print("perm", json.dumps({k: v for k, v in perm_info.items() if k != "top_hit_z"}, ensure_ascii=False), flush=True)

    # ---------- 2023~2025 같은 칸 ----------
    for y in [2023, 2024, 2025]:
        suby = np.where(E["year"].values == y)[0]
        dy = sorted({pd.Timestamp(d) for d in E["day"].values[suby]})
        ny, sy, _ = day_aggs(suby, dy)
        Ny, my, sey, Gy = (a[0] for a in cell_stats(ny, sy))
        with np.errstate(divide="ignore", invalid="ignore"):
            cells[f"n_{y}"] = Ny.astype(int); cells[f"mean_{y}"] = my; cells[f"t_{y}"] = my / sey

    # ---------- 칸 세부(후보·best) ----------
    def cell_mask(row, year=None):
        gm = (E["side"].values == row.side) & np.isclose(E["x"].values, row.x)
        mk = gm & F[:, row.f1]
        if row.f2 >= 0:
            mk &= F[:, row.f2]
        if year is not None:
            mk &= E["year"].values == year
        return mk

    def clus(sub):
        """사건 표 부분집합의 매매당 평균·날 묶음 t."""
        if len(sub) == 0:
            return {"n": 0, "days": 0, "mean": None, "t": None}
        g = sub.groupby("day")["v"].agg(["sum", "count"])
        Nn, Ss = g["count"].sum(), g["sum"].sum(); mm = Ss / Nn; Gg = len(g)
        ss = ((g["sum"] - mm * g["count"]) ** 2).sum()
        se_ = np.sqrt(Gg / (Gg - 1) * ss) / Nn if Gg > 1 else np.nan
        return {"n": int(Nn), "days": int(Gg), "mean": float(mm), "t": float(mm / se_) if se_ and se_ > 0 else None}

    def detail(row):
        e26 = E[cell_mask(row, 2026)]
        pick = lambda col: e26[["day"]].assign(v=e26[col].values).dropna(subset=["v"])
        main_ = clus(pick("r_w90"))
        h1 = e26[e26["month"] <= "2026-06"]; h2 = e26[e26["month"] >= "2026-07"]
        h1s = clus(h1[["day"]].assign(v=h1["r_w90"].values)); h2s = clus(h2[["day"]].assign(v=h2["r_w90"].values))
        msum = e26.groupby("month", observed=True)["r_w90"].agg(["sum", "count"])
        best_m = msum["sum"].idxmax()
        rest = e26[e26["month"] != best_m]
        rest_s = clus(rest[["day"]].assign(v=rest["r_w90"].values))
        c2 = clus(pick("r_w90_c2")); s2 = clus(pick("r_w90_s2")); sd = clus(pick("r_std"))
        rep = {}
        for y in [2023, 2024, 2025]:
            ey = E[cell_mask(row, y)]
            rep[str(y)] = clus(ey[["day"]].assign(v=ey["r_w90"].values))
            rep[str(y)]["same_sign"] = bool(rep[str(y)]["mean"] is not None and rep[str(y)]["mean"] > 0)
        checks = {"h1_gt0": bool(h1s["mean"] is not None and h1s["mean"] > 0),
                  "h2_gt0": bool(h2s["mean"] is not None and h2s["mean"] > 0),
                  "drop_best_month_gt0": bool(rest_s["mean"] is not None and rest_s["mean"] > 0),
                  "cost2x_gt0": bool(c2["mean"] is not None and c2["mean"] > 0)}
        r_ = f"c{int(row.c1[1:]):02d} {NAMES[int(row.c1[1:])]} {row.lv1}"
        if row.c2:
            r_ += f" & c{int(row.c2[1:]):02d} {NAMES[int(row.c2[1:])]} {row.lv2}"
        rule = f"{SIDE[row.side]}, 문턱 {row.x * 100:g}%, {r_}"
        return {
            "id": row.id, "rule": rule, "adj_p": float(row.p_rw_adj),
            "stats_2026": {**main_, "t_own_se": main_["t"], "t": float(row.t), "se_used(floor)": float(row.se),
                           "t_literal": float(row.t_literal), "t_pooled": float(row.t_pooled),
                           "adj_p_literal": float(row.p_rw_adj_literal), "adj_p_pooled": float(row.p_rw_adj_pooled),
                           "floor_binds": bool(row.floor_binds), "hit_rate": float(row.hit_rate),
                           "sum_per_day_207": float(row.sum_per_day), "rank_t": int(row.rank),
                           "p_boot_marg": float(row.p_boot_marg), "p_t1": float(row.p_t1), "bh_q10": bool(row.bh_q10)},
            "robustness": {"h1_2026_01_06": h1s, "h2_2026_07_10": h2s,
                           "best_month": {"month": str(best_m), "sum": float(msum.loc[best_m, "sum"]),
                                          "n": int(msum.loc[best_m, "count"]),
                                          "share_of_total": float(msum.loc[best_m, "sum"] / e26["r_w90"].sum()) if e26["r_w90"].sum() != 0 else None},
                           "drop_best_month": rest_s, "cost2x_r_w90_c2": c2, "stop2_r_w90_s2": s2,
                           "std_r_std(결정≤09:45만)": sd, "checks": checks, "all_checks_pass": all(checks.values())},
            "replication": rep,
        }

    cand = cells[cells["p_rw_adj"] <= ALPHA].sort_values(["p_rw_adj", "t"], ascending=[True, False])
    cand_d = [detail(r) for r in cand.itertuples(index=False)] if len(cand) else []
    survivors = [d for d in cand_d if d["robustness"]["all_checks_pass"]]
    best = cells[cells["eligible"]].sort_values(["p_rw_adj", "t"], ascending=[True, False]).head(10)
    best_d = [detail(r) for r in best.itertuples(index=False)]
    bh_list = cells[cells["bh_q10"]].sort_values("p_t1")

    # 그룹 기준선(조건 없음, 설명용)
    base = {}
    for gi, (sd_, x) in enumerate(groups):
        e = E[(E["side"] == sd_) & np.isclose(E["x"], x) & (E["year"] == 2026)]
        base[f"{sd_}|x{x * 100:g}"] = clus(e[["day"]].assign(v=e["r_w90"].values))

    # ---------- 저장 ----------
    out_cols = ["id", "side", "x", "c1", "lv1", "c2", "lv2", "n", "days", "mean", "se_own", "se", "t", "hit_rate", "sum_per_day",
                "eligible", "floor_binds", "rank", "p_t1", "p_boot_marg", "p_rw_adj", "bh_q10",
                "t_literal", "p_rw_adj_literal", "t_pooled", "p_rw_adj_pooled",
                "perm_mean_z", "perm_p_mean_z", "perm_hit_z", "perm_p_hit_z",
                "n_2023", "mean_2023", "t_2023", "n_2024", "mean_2024", "t_2024", "n_2025", "mean_2025", "t_2025"]
    cells[out_cols].to_csv(os.path.join(RES, "c1_cells.csv"), index=False, float_format="%.6g", encoding="utf-8-sig")
    per_group_elig = {f"{s}|x{x * 100:g}": int(cells[(cells.g == gi) & cells.eligible].shape[0]) for gi, (s, x) in enumerate(groups)}
    res = {
        "brief": "BRIEF4.md 'C1 조건 표 전수'",
        "data": "results/ev4_events.parquet(5,194건), 2026 거래일 207일(2026-01-02~10-28)",
        "scope": {
            "conditions": 28, "single_levels": nf, "pairs_per_group": len(pairs), "cells_per_group": per_group,
            "groups": 8, "cells_total": int(K), "eligible": int(elig.sum()), "eligible_by_group": per_group_elig,
            "eligibility": f"2026 사건 ≥ {MIN_N}, 날 ≥ {MIN_G}",
            "bootstrap": {"B": B, "seed": SEED, "unit": "2026 거래일 207일 복원 추출", "chunk": CHUNK},
            "null_max_t_step1": q, "rw_by_se_mode": rw_info,
            "literal_nonfinite_obs_t_eligible": nonfinite_obs, "floor_binds_eligible": int(cells.floor_binds.sum()),
            "sigma_group_2026": {f"{s_}|x{x_ * 100:g}": v for (s_, x_), v in zip(groups, sig_g)},
            "n_rw_adj_le_0.05": int(len(cand)), "n_bh_q10": int(bh.sum()),
            "n_survivors": len(survivors),
        },
        "method": {
            "stat": "칸 r_w90 매매당 평균 ÷ 날 묶음 CR1 표준오차, 단측(위쪽)",
            "rw": "t* = (m* − m̂)/se*, 관측 t 내림차순 접미 최대 = 단계 하강 남은 칸 집합 최대, p̃ = (1+#)/(B+1), 보정 p = 누적 최대",
            "bh": "칸별 단측 p = 1 − T_{G−1}(t), BH q 0.10(탐색 단서, 판정 아님)",
            "extra_checks": "2026 1~6월·7~10월 매매당 평균 > 0, 이익 합 최대 달을 빼도 > 0, r_w90_c2(반 호가 2배) > 0",
            "best_month": "칸 안 r_w90 합이 가장 큰 달",
            "forced_change": "BRIEF4의 칸 자기 날 묶음 se 스튜던트화(literal)는 r_w90의 +1.0 점 질량 때문에 퇴화(귀무 최대 t +inf 1,968/2,000회, 관측 최대 t 3.3e8 = 28건 모두 도달 칸). 주 판은 se 바닥 σ_g/√N(floor), 비교로 literal·pooled. 결과를 본 뒤 정한 변경이며 보수적 방향이다.",
            "one_sided": "위쪽 단측(최대 t). 칸 사건 집합은 사건 표의 부분집합(조건 맞는 사건만 실제로 사면 한 번에 하나 규칙의 이어짐이 조금 달라진다)",
            "sum_per_day": "칸 사건 r_w90 합 ÷ 207일(복리·겹침 무시 단순 합, 설명용)",
        },
        "baseline_2026_no_condition": base,
        "candidates_rw_le_0.05": cand_d,
        "survivors": survivors,
        "best_top10_by_adj_p": best_d,
        "posthoc_permutation_check": perm_info,
        "bh_q10_cells": [{"id": r.id, "n": int(r.n), "days": int(r.days), "mean": float(r.mean), "t": float(r.t),
                          "p_t1": float(r.p_t1), "p_rw_adj": float(r.p_rw_adj)} for r in bh_list.head(60).itertuples(index=False)],
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(res, open(os.path.join(RES, "c1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res["scope"], ensure_ascii=False, indent=1))
    print("candidates", len(cand_d), "survivors", len(survivors))
    for d in best_d:
        s_ = d["stats_2026"]
        print(f"{d['id']:60s} n {s_['n']:3d} d {s_['days']:3d} m {s_['mean']:+.3f} t {s_['t']:.2f} adj {d['adj_p']:.3f} "
              f"marg {s_['p_boot_marg']:.4f} chk {d['robustness']['checks']}")


if __name__ == "__main__":
    main()
