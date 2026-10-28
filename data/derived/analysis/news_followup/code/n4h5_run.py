"""N4-H5 — SOXX 실시간 반전 확인 ±5거래일 안 PB 노트(R·D·S·T)와 그 반전의 지속(y1·y20). BRIEF.md §N4 H5.

py code/n4h5_run.py   → results/n4h5_*.csv · n4h5_verdict.json · n4h5_run.log(표준출력)

입력(읽기만)
- research/pb_flow_census_20261026/census.csv (원 노트 252그룹; census_unverified.csv 는 쓰지 않는다)
- research/pb_flow_census_20261026/coverage.csv (주별 포착)
- cache/n3_raw.parquet (SOXX 일봉 2001~2026-10-23, auto_adjust=False — N3 와 같은 자료)
- ../swing_trend/results/events_ZZA3.parquet (주식 133종목 사건, 2023-06~2026-10-22)
- swing_trend/code/swing.py 의 atr·zigzag, code/n3_analyze.py 의 zz_runs·classify(N3 정의 그대로)

시점 규칙(protocol timing, note_date 는 쓰지 않는다)
- a  = 행동 세션(그 세션 시가에 쓸 수 있다). 날짜만 있는 행(exact 가 아님): 그 날짜 다음 정규장. exact: 09:30 ET 전이면 그날, 아니면 다음 정규장.
- kb = '종가까지 알려진' 세션. 날짜만 있는 행: a 와 같다(그날 장 마감 뒤일 수 있어 보수적). exact: 16:00 ET 전이면 그날, 아니면 다음 정규장.
- 확인 c(종가에 확정, c+1 시가 행동)마다 창(세션 수)
    any  = kb ∈ [c−5, c+5]  (등록 H5 그대로, 사후 정보 포함)
    pre  = kb ∈ [c−5, c]    ('확인일 종가까지 공개된 것' = 사전, 과제 정의)
    post = kb ∈ [c+1, c+5]  (확인 뒤 = 사후)
    pre_act = a ∈ [c−4, c+1] (민감도·사후: 확인 매매 시가(c+1)에 쓸 수 있으면 사전)
- 같은 family 5거래일 안 연속 노트: first = 행동 세션 a 기준 직전 채택 신호와 5세션 미만이면 버린다(N1 최소 간격 5와 같음), all = 전부.

결과(swing_trend BRIEF2·events.py 와 같은 정의, 확정 다음 날 시가 체결)
- y1 = type × log(반대 확정 다음 날 시가 / c+1 시가). 반대 확정이 없거나 체결이 자료 밖이면 마지막 종가(10/23)로 자른다(censored).
- y10·y20 = type × log(O[c+11 또는 c+21] / O[c+1]).
- 'V자' = y1 < 0(그 반전 매매가 손실로 끝남).

판정(BRIEF N4: 날짜 묶음 부트스트랩 95% 구간 + 원형 이동 귀무)
- 차이 = 평균 y(노트 있음) − 평균 y(노트 없음). 통과 = 차이 > 0, 부트스트랩 하한 > 0, 귀무 95분위 초과.
- 부트스트랩: 20세션 달력 블록을 복원 추출(같은 블록 사건은 함께), B_SOXX·B_STOCK 회, 같은 (det·type·구간) 칸은 같은 재표본을 쓴다.
- SOXX 귀무(주): 노트 세션(kb·a)을 2023-01-03~2026-10-23 세션 범위에서 원형으로 k칸 민다. k = 1…S−1 전부.
  보조: 같은 type 사건 사이 노트 표지 원형 회전 k = 1…n−1 전부.
- 주식 귀무(과제 지시): 같은 구간·type 사건의 고유 날짜 D개에 붙은 표지를 원형으로 k칸 돌린다. k = 1…D−1 전부.
- 기간 3(BRIEF '가설 5 × 기간 3'의 해석): 전체 2023~2026, 2026(1/2~10/23), 2026 6~10월(P2). 연도별·P1 은 설명용.

등록 밖(사후) 표시: pre_act 창, 대안 계열(alt)·제목뿐 행 제외(nohead)·범위(scope) 민감도, 포착 주 한정 비교, 주식 사건 확장 전체,
'확인 지연' 분석(노트 시가 행동 대 확정 대기, N3 classify 그대로) 전체.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent.parent
ROOT = HERE.parents[3]
CENSUS = ROOT / "research" / "pb_flow_census_20261026"
SWINGR = HERE.parent / "swing_trend" / "results"
RES = HERE / "results"
RES.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / "cache" / "numba"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / "swing_trend" / "code"))
sys.path.insert(0, str(HERE / "code"))
from n3_analyze import classify, zz_runs  # noqa: E402

SEED = 20261026
B_SOXX, B_STOCK = 5000, 3000
BLOCK = 20
TIE = 1e-9          # 이산 귀무의 동점은 통과로 치지 않는다
MIN_N = 3           # (사후) SOXX 칸 한쪽 사건 3개 미만이면 판정 불가
MIN_DATES = 10      # (사후) 주식 칸 한쪽 날짜 10일 미만이면 판정 불가(swing BRIEF2 '날짜 10일 이상'과 같음)
FAMS = ("R", "D", "S", "T")
VARS = ("all", "first", "alt", "nohead", "scope")
WINS = {"any": ("kb", -5, 5), "pre": ("kb", -5, 0), "post": ("kb", 1, 5), "pre_act": ("a", -4, 1)}
OUTS = ("y1", "y10", "y20")
START, END = pd.Timestamp("2023-01-03"), pd.Timestamp("2026-10-23")
REG_PERIODS = ("ALL", "Y2026", "P2")
SOXX_PERIODS = ("ALL", "Y2023", "Y2024", "Y2025", "Y2026", "P1", "P2")
STOCK_PERIODS = ("ALL", "Y23", "Y24", "Y25", "Y2026", "P1", "P2")

# 대안 계열(사후 민감도): coding_notes 의 'family=X'(다른 부호자)·'X 대안' 문구와 reliability.md 재부호화 불일치.
ALT = {"G0001": "R", "G0012": "T", "G0061": "S", "G0066": "N", "G0086": "N", "G0132": "D", "G0150": "T",
       "G0164": "N", "G0167": "S", "G0183": "N", "G0187": "D", "G0228": "T", "G0232": "N", "G0246": "S",
       "G0249": "N", "G0250": "N", "G0252": "N",
       "G0131": "R", "G0211": "R", "G0043": "T", "G0032": "S"}
SCOPE_OUT = re.compile(r"금융|은행|에너지|산업재|헬스|유틸|원자재|경기민감|소비재|중국|아시아|일본|한국|홍콩|유럽|UK|EM|신흥|"
                       r"단일|개별 종목|최다 공매도|샤오미|방산|CTA")


# ---------------------------------------------------------------- 노트
def note_table(dates: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.read_csv(CENSUS / "census.csv")
    rows = []
    for _, r in d.iterrows():
        pa = r.published_at
        kb = a = np.nan
        if isinstance(pa, str) and pa:
            if r.time_precision == "exact" and len(pa) > 10:
                ts = pd.Timestamp(pa).tz_convert("America/New_York")
                day = pd.Timestamp(ts.date())
                j = int(dates.searchsorted(day))
                sess = j < len(dates) and dates[j] == day
                mins = ts.hour * 60 + ts.minute + ts.second / 60
                a = j if (sess and mins < 9 * 60 + 30) else (j + 1 if sess else j)
                kb = j if (sess and mins < 16 * 60) else (j + 1 if sess else j)
            else:
                j = int(dates.searchsorted(pd.Timestamp(pa[:10]), side="right"))
                a = kb = j
        scope = str(r.scope)
        rows.append(dict(group_id=r.group_id, bank=r.bank, family=r.family, alt_family=ALT.get(r.group_id, r.family),
                         published_at=pa, time_precision=r.time_precision, verification=r.verification,
                         uncertain_dup=r.uncertain_dup if isinstance(r.uncertain_dup, str) else "",
                         scope=scope, broad_scope=not bool(SCOPE_OUT.search(scope)),
                         a=a, kb=kb))
    nt = pd.DataFrame(rows)
    ok = nt.a.notna() & (nt.a < len(dates))
    nt["a_date"] = pd.NaT
    nt["kb_date"] = pd.NaT
    nt.loc[ok, "a_date"] = dates[nt.a[ok].astype(int)]
    ok2 = nt.kb.notna() & (nt.kb < len(dates))
    nt.loc[ok2, "kb_date"] = dates[nt.kb[ok2].astype(int)]
    return nt


def decluster(a: np.ndarray) -> np.ndarray:
    """행동 세션 순으로 직전 채택 신호와 5세션 미만이면 버린다."""
    o = np.argsort(a, kind="stable")
    keep = np.zeros(len(a), bool)
    last = -10 ** 9
    for i in o:
        if a[i] - last >= 5:
            keep[i] = True
            last = a[i]
    return keep


def note_sets(nt: pd.DataFrame) -> dict:
    """(family, variant) → DataFrame(그 묶음의 노트, a·kb 정수)."""
    base = nt[nt.a.notna()].copy()
    base["a"] = base.a.astype(int)
    base["kb"] = base.kb.astype(int)
    out = {}
    for f in FAMS:
        g = base[base.family == f]
        out[(f, "all")] = g
        out[(f, "first")] = g[decluster(g.a.to_numpy())]
        out[(f, "alt")] = base[base.alt_family == f]
        out[(f, "nohead")] = g[g.verification != "headline_summary_only"]
        out[(f, "scope")] = g[g.broad_scope]
    return out


def counts(ev_idx: np.ndarray, note_idx: np.ndarray, lo: int, hi: int) -> np.ndarray:
    s = np.sort(note_idx)
    return np.searchsorted(s, ev_idx + hi, "right") - np.searchsorted(s, ev_idx + lo, "left")


def shifted_labels(ev_idx: np.ndarray, note_idx: np.ndarray, lo: int, hi: int, i0: int, S: int) -> np.ndarray:
    """세션 원형 이동 귀무: (S−1, n_ev) 표지. 범위 밖 노트는 뺀다."""
    x = note_idx[(note_idx >= i0) & (note_idx < i0 + S)] - i0
    out = np.zeros((S - 1, len(ev_idx)), bool)
    for k in range(1, S):
        out[k - 1] = counts(ev_idx, i0 + (x + k) % S, lo, hi) > 0
    return out


# ---------------------------------------------------------------- 통계
def diff_stats(y: np.ndarray, lab: np.ndarray) -> dict:
    ok = ~np.isnan(y)
    y, lab = y[ok], lab[ok]
    n1, n0 = int(lab.sum()), int((~lab).sum())
    m1 = float(y[lab].mean()) if n1 else np.nan
    m0 = float(y[~lab].mean()) if n0 else np.nan
    v1 = float((y[lab] < 0).mean()) if n1 else np.nan
    v0 = float((y[~lab] < 0).mean()) if n0 else np.nan
    return dict(n1=n1, n0=n0, m1=m1, m0=m0, diff=m1 - m0, v1=v1, v0=v0)


def boot_ci(y, lab, blk, W, ublk):
    """W: (B, nb) 블록 복원 추출 횟수. 반환 (lo, hi, NaN 비율)."""
    ok = ~np.isnan(y)
    y, lab, blk = y[ok], lab[ok], blk[ok]
    pos = np.searchsorted(ublk, blk)
    nb = len(ublk)
    s1 = np.bincount(pos, weights=y * lab, minlength=nb)
    c1 = np.bincount(pos, weights=lab.astype(float), minlength=nb)
    s0 = np.bincount(pos, weights=y * ~lab, minlength=nb)
    c0 = np.bincount(pos, weights=(~lab).astype(float), minlength=nb)
    M = W @ np.stack([s1, c1, s0, c0], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = M[:, 0] / M[:, 1] - M[:, 2] / M[:, 3]
    good = ~np.isnan(d)
    if good.sum() < 100:
        return np.nan, np.nan, float(1 - good.mean())
    lo, hi = np.quantile(d[good], [0.025, 0.975])
    return float(lo), float(hi), float(1 - good.mean())


def null_diffs(y: np.ndarray, labs: np.ndarray) -> np.ndarray:
    """labs: (K, n) 귀무 표지 → (K,) 차이."""
    ok = ~np.isnan(y)
    y, L = y[ok], labs[:, ok].astype(float)
    c1 = L.sum(1)
    c0 = len(y) - c1
    s1 = L @ y
    s0 = y.sum() - s1
    with np.errstate(invalid="ignore", divide="ignore"):
        return s1 / c1 - s0 / c0


def mde(y: np.ndarray, n1: int, n0: int) -> float:
    y = y[~np.isnan(y)]
    if n1 < 1 or n0 < 1 or len(y) < 3:
        return np.nan
    return float((1.645 + 0.842) * y.std(ddof=1) * np.sqrt(1 / n1 + 1 / n0))


def v_auc(y1: np.ndarray, lab: np.ndarray) -> float:
    """표지가 '이어짐'(y1 ≥ 0)을 가리키는 정도. 0.5 + (P(표지|이어짐) − P(표지|V))/2."""
    ok = ~np.isnan(y1)
    v, l = y1[ok] < 0, lab[ok]
    if v.all() or (~v).all():
        return np.nan
    return float(0.5 + (l[~v].mean() - l[v].mean()) / 2)


# ---------------------------------------------------------------- SOXX 사건
def soxx_events(z: dict, det: str, i0: int) -> pd.DataFrame:
    st, pi, pc, pt = (z[det][k] for k in ("st", "pi", "pc", "pt"))
    O, C = z["O"], z["C"]
    n = len(C)
    dates = z["dates"]
    rows = []
    for e, (p, c, t) in enumerate(zip(pi, pc, pt)):
        if c < i0 or c + 1 >= n:
            continue
        c2 = int(pc[e + 1]) if e + 1 < len(pc) else None
        cens = c2 is None or c2 + 1 >= n
        ex = C[-1] if cens else O[c2 + 1]
        r = dict(det=det, type=int(t), conf_idx=int(c), conf_date=dates[c], pivot_date=dates[p], S1=int(c - p),
                 leg_move=float(t * np.log(C[c] / C[p])), r20=float(np.log(C[c] / C[c - 20])),
                 exit_date=dates[c2] if c2 is not None else pd.NaT, censored=bool(cens),
                 y1=float(t * np.log(ex / O[c + 1])))
        for h in (10, 20):
            r[f"y{h}"] = float(t * np.log(O[c + h + 1] / O[c + 1])) if c + h + 1 < n else np.nan
        rows.append(r)
    ev = pd.DataFrame(rows)
    ev["year"] = pd.DatetimeIndex(ev.conf_date).year
    return ev


def period_mask(dts: pd.Series, per: str) -> np.ndarray:
    d = pd.DatetimeIndex(dts)
    if per == "ALL":
        return np.ones(len(d), bool)
    if per.startswith("Y20"):
        return d.year == int(per[1:])
    if per == "P1":
        return (d >= "2026-01-02") & (d <= "2026-05-29")
    if per == "P2":
        return (d >= "2026-06-01") & (d <= "2026-10-23")
    raise ValueError(per)


def run_soxx(z, nsets, i0, S, rng, cov):
    ev_all, cells, nullstore = [], [], {}
    for det in ("ZZA3", "ZZP8"):
        ev = soxx_events(z, det, i0)
        ev["block"] = (ev.conf_idx - i0) // BLOCK
        cidx = ev.conf_idx.to_numpy()
        # 포착(사후 민감도): 사전 창 [c−5, c] 의 모든 주에 GS 공개 노트가 1개 이상
        ev["gs_cov_pre"] = [cov_ok(z["dates"], c, cov) for c in cidx]
        labels, nulls = {}, {}
        for (f, v), g in nsets.items():
            for w, (col, lo, hi) in WINS.items():
                idx = g[col].to_numpy()
                lab = counts(cidx, idx, lo, hi) > 0
                labels[(f, v, w)] = lab
                ev[f"{f}_{v}_{w}"] = lab
                nulls[(f, v, w)] = shifted_labels(cidx, idx, lo, hi, i0, S)
                if v == "all":
                    ids = g.group_id.to_numpy()
                    ev[f"{f}_{w}_ids"] = [";".join(ids[(idx >= c + lo) & (idx <= c + hi)]) for c in cidx]
        for t in (1, -1):
            for per in SOXX_PERIODS:
                m = (ev.type.to_numpy() == t) & period_mask(ev.conf_date, per)
                if m.sum() < 2:
                    continue
                sub = ev[m]
                ublk = np.unique(sub.block)
                W = rng.multinomial(len(ublk), np.full(len(ublk), 1 / len(ublk)), size=B_SOXX).astype(float)
                for key, lab_full in labels.items():
                    f, v, w = key
                    lab = lab_full[m]
                    NL = nulls[key][:, m]
                    # 보조 귀무: 같은 type·구간 사건끼리 표지 회전
                    nn = len(lab)
                    rot = np.stack([np.roll(lab, k) for k in range(1, nn)]) if nn > 1 else np.zeros((0, nn), bool)
                    for o in OUTS:
                        y = sub[o].to_numpy(float)
                        st_ = diff_stats(y, lab)
                        lo_, hi_, nanb = boot_ci(y, lab, sub.block.to_numpy(), W, ublk)
                        nd = null_diffs(y, NL)
                        q95 = float(np.nanquantile(nd, 0.95)) if np.isfinite(nd).any() else np.nan
                        p = float(np.mean(nd[np.isfinite(nd)] >= st_["diff"])) if np.isfinite(st_["diff"]) and np.isfinite(nd).any() else np.nan
                        p_low = float(np.mean(nd[np.isfinite(nd)] <= st_["diff"])) if np.isfinite(st_["diff"]) and np.isfinite(nd).any() else np.nan
                        rd = null_diffs(y, rot) if len(rot) else np.array([np.nan])
                        p_rot = float(np.mean(rd[np.isfinite(rd)] >= st_["diff"])) if np.isfinite(st_["diff"]) and np.isfinite(rd).any() else np.nan
                        ok = (np.isfinite(st_["diff"]) and st_["diff"] > 0 and np.isfinite(lo_) and lo_ > 0
                              and np.isfinite(q95) and st_["diff"] > q95 + TIE and p < 0.05)
                        cells.append(dict(det=det, type=t, period=per, family=f, variant=v, window=w, outcome=o,
                                          **st_, boot_lo=lo_, boot_hi=hi_, boot_nan=nanb, null_q95=q95, p_shift=p,
                                          p_low=p_low, null_valid=int(np.isfinite(nd).sum()), p_rot=p_rot, n_rot=int(np.isfinite(rd).sum()),
                                          judgeable=bool(st_["n1"] >= MIN_N and st_["n0"] >= MIN_N),
                                          mde80=mde(y, st_["n1"], st_["n0"]),
                                          auc_cont=v_auc(sub.y1.to_numpy(float), lab) if o == "y1" else np.nan,
                                          passed=bool(ok)))
                        if per in REG_PERIODS and f == "R" and v in ("all", "first") and w in ("any", "pre") and o in ("y1", "y20"):
                            nullstore[(det, t, per, v, w, o)] = (nd, st_["diff"])
        ev_all.append(ev)
    return pd.concat(ev_all, ignore_index=True), pd.DataFrame(cells), nullstore


def cov_ok(dates, c, cov):
    lo = dates[max(c - 5, 0)]
    hi = dates[c]
    wk = pd.date_range(lo - pd.Timedelta(days=lo.weekday()), hi, freq="W-MON")
    return bool(all(cov.get(w, 0) > 0 for w in wk))


def family_null(nullstore, cells, det, v, w):
    """등록 칸(R·type 2·y1/y20·기간 3) 통과 수를 같은 k 에서 센 계열 귀무. 부트스트랩 하한 조건은 귀무에서 생략(근사)."""
    keys = [k for k in nullstore if k[0] == det and k[3] == v and k[4] == w]
    if not keys:
        return {}
    K = min(len(nullstore[k][0]) for k in keys)
    cnt = np.zeros(K, int)
    for k in keys:
        nd = nullstore[k][0][:K]
        q95 = np.nanquantile(nd, 0.95)
        x = np.nan_to_num(nd, nan=-9)
        pk = np.array([np.mean(x >= v) for v in x])
        cnt += (x > 0) & (x > q95 + TIE) & (pk < 0.05)
    sub = cells[(cells.det == det) & (cells.family == "R") & (cells.variant == v) & (cells.window == w)
                & cells.period.isin(REG_PERIODS) & cells.outcome.isin(["y1", "y20"])]
    act = int(sub.passed.sum())
    act_shift = int(((sub["diff"] > 0) & (sub["diff"] > sub.null_q95 + TIE) & (sub.p_shift < 0.05)).sum())
    return dict(cells=len(keys), passed_full=act, passed_shift_only=act_shift, null_mean=float(cnt.mean()),
                null_q95=float(np.quantile(cnt, 0.95)), p=float(np.mean(cnt >= act_shift)), reps=int(K))


# ---------------------------------------------------------------- 주식 사건
def run_stock(dates, nsets, i0, rng):
    ev = pd.read_parquet(SWINGR / "events_ZZA3.parquet")
    ev["date"] = pd.to_datetime(ev.date)
    sidx = dates.searchsorted(ev.date)
    assert (dates[sidx] == ev.date).all(), "주식 사건 날짜가 SOXX 세션에 없다"
    ev["sidx"] = sidx
    ev["block"] = (ev.sidx - i0) // BLOCK
    cells = []
    for t in (1, -1):
        for per in STOCK_PERIODS:
            if per == "ALL":
                m = ev.type == t
            elif per == "Y2026":
                m = (ev.type == t) & ev.period.isin(["P1", "P2"])
            else:
                m = (ev.type == t) & (ev.period == per)
            sub = ev[m].reset_index(drop=True)
            U = np.unique(sub.sidx.to_numpy())
            D = len(U)
            di = np.searchsorted(U, sub.sidx.to_numpy())
            circ = (np.arange(D)[None, :] + np.arange(1, D)[:, None]) % D        # (D−1, D)
            ublk = np.unique(sub.block)
            W = rng.multinomial(len(ublk), np.full(len(ublk), 1 / len(ublk)), size=B_STOCK).astype(float)
            Y = {o: sub[o].to_numpy(float) for o in OUTS}
            # 날짜별 합
            per_date = {}
            for o in OUTS:
                ok = ~np.isnan(Y[o])
                per_date[o] = (np.bincount(di[ok], weights=Y[o][ok], minlength=D), np.bincount(di[ok], minlength=D).astype(float))
            for (f, v), g in nsets.items():
                for w, (col, lo, hi) in WINS.items():
                    LU = counts(U, g[col].to_numpy(), lo, hi) > 0                  # 날짜 표지
                    lab = LU[di]
                    LR = LU[circ].astype(float)                                     # (D−1, D)
                    for o in OUTS:
                        y = Y[o]
                        st_ = diff_stats(y, lab)
                        lo_, hi_, nanb = boot_ci(y, lab, sub.block.to_numpy(), W, ublk)
                        sy, cn = per_date[o]
                        s1, c1 = LR @ sy, LR @ cn
                        with np.errstate(invalid="ignore", divide="ignore"):
                            nd = s1 / c1 - (sy.sum() - s1) / (cn.sum() - c1)
                        fin = np.isfinite(nd)
                        q95 = float(np.quantile(nd[fin], 0.95)) if fin.any() else np.nan
                        p = float(np.mean(nd[fin] >= st_["diff"])) if fin.any() and np.isfinite(st_["diff"]) else np.nan
                        p_low = float(np.mean(nd[fin] <= st_["diff"])) if fin.any() and np.isfinite(st_["diff"]) else np.nan
                        ok = (np.isfinite(st_["diff"]) and st_["diff"] > 0 and np.isfinite(lo_) and lo_ > 0
                              and np.isfinite(q95) and st_["diff"] > q95 + TIE and p < 0.05)
                        cells.append(dict(type=t, period=per, family=f, variant=v, window=w, outcome=o, **st_,
                                          dates_with=int(LU.sum()), dates=D, boot_lo=lo_, boot_hi=hi_, boot_nan=nanb,
                                          null_q95=q95, p_rot=p, p_low=p_low, null_valid=int(fin.sum()),
                                          judgeable=bool(np.unique(di[lab & ~np.isnan(y)]).size >= MIN_DATES
                                                         and np.unique(di[~lab & ~np.isnan(y)]).size >= MIN_DATES),
                                          auc_cont=v_auc(Y["y1"], lab) if o == "y1" else np.nan, passed=bool(ok)))
    return pd.DataFrame(cells), ev


# ---------------------------------------------------------------- 확인 지연(사후)
def lead_tables(z, nsets, i0, S):
    """노트 시가 행동 대 확정 기다리기(N3 classify). 같은 상태 임의 세션(기준선)과 세션 원형 이동 귀무."""
    rows, base_rows, per_note = [], [], []
    for det in ("ZZA3", "ZZP8"):
        look = {}
        for want, fam_code in ((1, "R"), (-1, "T")):
            cat = np.empty(S, object)
            gain = np.full(S, np.nan)
            lead = np.full(S, np.nan)
            lag = np.full(S, np.nan)
            for j in range(S):
                r = classify(z, det, i0 + j, fam_code)
                cat[j], gain[j], lead[j], lag[j] = r["cat"], r["gain"], r["lead"], r["lag_share"]
            look[want] = (cat, gain, lead, lag)
            isl = np.array([str(c).startswith("선행") for c in cat])
            sweet = np.isin(cat, ["선행·극점 뒤", "선행·극점 당일"])
            yrs = pd.DatetimeIndex(z["dates"][i0:i0 + S]).year
            for per in ("ALL", "Y2026"):
                pm = np.ones(S, bool) if per == "ALL" else (yrs == 2026)
                gg = isl & pm & np.isfinite(gain)
                base_rows.append(dict(det=det, want=want, period=per, n_lead_days=int(gg.sum()),
                                      gain_mean=float(gain[gg].mean()), gain_pos=float((gain[gg] > 0).mean()),
                                      sweet_share=float(sweet[isl & pm].mean()), lead_med=float(np.median(lead[gg]))))
        for (f, v), g in nsets.items():
            if v not in ("all", "first"):
                continue
            want = -1 if f == "T" else 1
            cat, gain, lead, lag = look[want]
            isl = np.array([str(c).startswith("선행") for c in cat])
            sweet = np.isin(cat, ["선행·극점 뒤", "선행·극점 당일"])
            inr = (g.a.to_numpy() >= i0) & (g.a.to_numpy() < i0 + S)
            a = g.a.to_numpy()[inr] - i0
            for j, gid in zip(a, g.group_id.to_numpy()[inr]):
                if v == "all":
                    per_note.append(dict(det=det, family=f, group_id=gid, a_date=z["dates"][i0 + j], cat=cat[j],
                                         lead=lead[j], gain=gain[j], lag_share=lag[j]))
            yrs = pd.DatetimeIndex(z["dates"][i0:i0 + S]).year

            def metric(aa):
                m = isl[aa] & np.isfinite(gain[aa])
                return (gain[aa][m].mean() if m.any() else np.nan, int(m.sum()), int(sweet[aa].sum()), int(isl[aa].sum()))

            for per in ("ALL", "Y2026"):
                aa = a if per == "ALL" else a[yrs[a] == 2026]
                if not len(aa):
                    continue
                gm, nl, nsw, nld = metric(aa)
                nulls = np.array([metric((aa + k) % S)[0] for k in range(1, S)])
                nsw_null = np.array([metric((aa + k) % S)[2] for k in range(1, S)])
                base = [b for b in base_rows if b["det"] == det and b["want"] == want and b["period"] == per][0]
                cats = pd.Series(cat[aa]).value_counts().to_dict()
                rows.append(dict(det=det, family=f, variant=v, period=per, n_notes=len(aa), n_lead=nld, n_lead_gain=nl,
                                 n_sweet=nsw, sweet_null_mean=float(nsw_null.mean()), p_sweet=float(np.mean(nsw_null >= nsw)),
                                 gain_mean=float(gm), base_gain_mean=base["gain_mean"], excess=float(gm - base["gain_mean"]),
                                 null_q95=float(np.nanquantile(nulls, 0.95)), p_shift=float(np.nanmean(nulls >= gm)),
                                 lead_med=float(np.nanmedian(lead[aa][isl[aa]])) if nld else np.nan,
                                 lag_share_med_sweet=float(np.nanmedian(lag[aa][sweet[aa]])) if nsw else np.nan,
                                 **{f"cat_{k}": int(vv) for k, vv in cats.items()}))
    return pd.DataFrame(rows), pd.DataFrame(base_rows), pd.DataFrame(per_note)


def confirm_side(z, ev, nsets):
    """매수(매도) 확정마다: 극점 뒤 ~ 확정 매매 시가(c+1) 사이에 쓸 수 있었던 첫 R(T) 노트가 있었는가, 앞당긴 세션·가격 이득."""
    rows = []
    O = z["O"]
    for _, r in ev.iterrows():
        f = "R" if r.type == 1 else "T"
        for ff in ([f, "D", "S"] if r.type == 1 else [f]):
            a = nsets[(ff, "all")].a.to_numpy()
            p = int(z["dates"].searchsorted(r.pivot_date))
            c = int(r.conf_idx)
            m = (a >= p) & (a <= c)             # 극점 당일 시가 ~ 확정 세션 시가(확정 매매 c+1 시가보다 앞)
            if m.any():
                ae = int(a[m].min())
                gain = (O[c + 1] / O[ae] - 1) if r.type == 1 else (O[ae] / O[c + 1] - 1)
                rows.append(dict(det=r.det, type=r.type, conf_date=r.conf_date, family=ff, first_a=z["dates"][ae],
                                 sessions_saved=c + 1 - ae, leg_delay=c + 1 - p, gain_vs_wait=gain, y1=r.y1))
            else:
                rows.append(dict(det=r.det, type=r.type, conf_date=r.conf_date, family=ff, first_a=pd.NaT,
                                 sessions_saved=0, leg_delay=c + 1 - p, gain_vs_wait=np.nan, y1=r.y1))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    raw = pd.read_parquet(HERE / "cache" / "n3_raw.parquet")
    z = zz_runs(raw, "SOXX")
    dates = z["dates"]
    i0 = int(dates.searchsorted(START))
    i1 = int(dates.searchsorted(END, side="right")) - 1
    S = i1 - i0 + 1
    assert dates[i0] == START and dates[i1] == END

    nt = note_table(dates)
    nt.to_csv(RES / "n4h5_notes.csv", index=False, encoding="utf-8-sig")
    nsets = note_sets(nt)
    cov_df = pd.read_csv(CENSUS / "coverage.csv")
    cov = cov_df[cov_df.bank == "GS"].set_index(pd.to_datetime(cov_df[cov_df.bank == "GS"].week_start)).groups_published.to_dict()

    ev, cells, nullstore = run_soxx(z, nsets, i0, S, rng, cov)
    ev.to_csv(RES / "n4h5_soxx_events.csv", index=False, encoding="utf-8-sig")
    cells.to_csv(RES / "n4h5_soxx_cells.csv", index=False, encoding="utf-8-sig")

    # 포착 주 한정(사후): R 사전 표지가 없는 확인 중 그 주에 GS 공개 노트가 있었던 것만 '없음'으로
    cov_rows = []
    for det in ("ZZA3", "ZZP8"):
        for t in (1, -1):
            e = ev[(ev.det == det) & (ev.type == t)]
            lab = e["R_all_pre"].to_numpy()
            covd = e.gs_cov_pre.to_numpy()
            for o in ("y1", "y20"):
                y = e[o].to_numpy(float)
                keep = lab | covd
                st_ = diff_stats(y[keep], lab[keep])
                cov_rows.append(dict(det=det, type=t, outcome=o, gs_cov_share=float(covd.mean()),
                                     cov_share_noR=float(covd[~lab].mean()) if (~lab).any() else np.nan, **st_))
    cov_t = pd.DataFrame(cov_rows)
    cov_t.to_csv(RES / "n4h5_soxx_coverage.csv", index=False, encoding="utf-8-sig")

    fam = {f"{det}_{v}_{w}": family_null(nullstore, cells, det, v, w)
           for det in ("ZZA3", "ZZP8") for v in ("all", "first") for w in ("any", "pre")}

    st_cells, sev = run_stock(dates, nsets, i0, rng)
    st_cells.to_csv(RES / "n4h5_stock_cells.csv", index=False, encoding="utf-8-sig")

    lead, base, per_note = lead_tables(z, nsets, i0, S)
    lead.to_csv(RES / "n4h5_lead.csv", index=False, encoding="utf-8-sig")
    base.to_csv(RES / "n4h5_lead_baseline.csv", index=False, encoding="utf-8-sig")
    per_note.to_csv(RES / "n4h5_lead_notes.csv", index=False, encoding="utf-8-sig")
    cs = confirm_side(z, ev, nsets)
    cs.to_csv(RES / "n4h5_confirm_side.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- 요약
    reg = cells[(cells.family == "R") & cells.variant.isin(["all", "first"]) & cells.window.isin(["any", "pre"])
                & cells.period.isin(REG_PERIODS) & cells.outcome.isin(["y1", "y20"])]
    reg_st = st_cells[(st_cells.family == "R") & st_cells.variant.isin(["all", "first"]) & st_cells.window.isin(["any", "pre"])
                      & st_cells.period.isin(REG_PERIODS)]
    npre = {}
    for det in ("ZZA3", "ZZP8"):
        for t in (1, -1):
            for per in REG_PERIODS:
                m = (ev.det == det) & (ev.type == t) & period_mask(ev.conf_date, per)
                npre[f"{det}_{t}_{per}"] = dict(n=int(m.sum()), R_pre=int(ev[m].R_all_pre.sum()), R_any=int(ev[m].R_all_any.sum()),
                                                R_post=int(ev[m].R_all_post.sum()))
    notes_summary = dict(
        groups=len(nt), with_published_at=int(nt.a.notna().sum()),
        by_family={f: int(((nt.family == f) & nt.a.notna()).sum()) for f in FAMS},
        by_family_first={f: int(len(nsets[(f, "first")])) for f in FAMS},
        exact=int((nt.time_precision == "exact").sum()),
        R_in_range=int(((nsets[("R", "all")].kb >= i0) & (nsets[("R", "all")].kb <= i1)).sum()),
    )
    # 기저율: 임의 세션에서 R 사전 창 [c−5, c] 가 켜질 비율
    allidx = np.arange(i0, i1 + 1)
    base_rate = {f: float((counts(allidx, nsets[(f, "all")].kb.to_numpy(), -5, 0) > 0).mean()) for f in FAMS}
    base_rate_any = {f: float((counts(allidx, nsets[(f, "all")].kb.to_numpy(), -5, 5) > 0).mean()) for f in FAMS}
    verdict = dict(
        sessions=dict(start=str(START.date()), end=str(END.date()), S=S),
        notes=notes_summary, base_rate_pre=base_rate, base_rate_any=base_rate_any,
        soxx_events={det: dict(ev[ev.det == det].groupby("type").size().rename(str).to_dict()) for det in ("ZZA3", "ZZP8")},
        soxx_counts=npre,
        soxx_registered_cells=int(len(reg)), soxx_registered_pass=int(reg.passed.sum()),
        soxx_registered_pass_rows=reg[reg.passed][["det", "type", "period", "variant", "window", "outcome", "diff", "boot_lo", "null_q95"]].to_dict("records"),
        soxx_family_null=fam,
        soxx_cells_total=int(len(cells)), soxx_pass_total=int(cells.passed.sum()),
        stock_events=dict(n=int(len(sev)), dates=int(sev.date.nunique())),
        stock_reg_cells=int(len(reg_st)), stock_reg_pass=int(reg_st.passed.sum()),
        stock_reg_pass_rows=reg_st[reg_st.passed][["type", "period", "variant", "window", "outcome", "diff", "boot_lo", "null_q95", "p_rot"]].to_dict("records"),
        stock_cells_total=int(len(st_cells)), stock_pass_total=int(st_cells.passed.sum()),
        boot=dict(B_soxx=B_SOXX, B_stock=B_STOCK, block_sessions=BLOCK, seed=SEED),
        nulls=dict(soxx_shift=f"k=1..{S - 1}", stock="날짜 표지 원형 회전 k=1..D−1"),
    )
    (RES / "n4h5_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    print(json.dumps(verdict, ensure_ascii=False, indent=1, default=str))
    cols = ["det", "type", "period", "family", "variant", "window", "outcome", "n1", "n0", "m1", "m0", "diff", "boot_lo", "boot_hi",
            "null_q95", "p_shift", "p_low", "p_rot", "mde80", "v1", "v0", "auc_cont", "judgeable", "passed"]
    show = lambda q: print(q[cols].round(4).to_string(index=False))
    C_ = cells
    print("\n== SOXX 등록 칸(R, all, any·pre, 기간 3, y1·y20) ==")
    show(C_[(C_.family == "R") & (C_.variant == "all") & C_.window.isin(["any", "pre"]) & C_.period.isin(REG_PERIODS) & C_.outcome.isin(["y1", "y20"])])
    print("\n== SOXX R first(첫 신호만) 전체 기간 ==")
    show(C_[(C_.family == "R") & (C_.variant == "first") & (C_.period == "ALL") & C_.outcome.isin(["y1", "y20"]) & C_.window.isin(["any", "pre"])])
    print("\n== SOXX R post·pre_act·민감도(전체 기간, y1·y20) ==")
    show(C_[(C_.family == "R") & (C_.period == "ALL") & C_.outcome.isin(["y1", "y20"])
            & ((C_.window.isin(["post", "pre_act"]) & (C_.variant == "all")) | (C_.variant.isin(["alt", "nohead", "scope"]) & (C_.window == "pre")))])
    print("\n== SOXX R pre 연도별(all, y1) ==")
    show(C_[(C_.family == "R") & (C_.variant == "all") & (C_.window == "pre") & (C_.outcome == "y1")])
    print("\n== SOXX D·S·T pre·post(all, 전체·2026, y1·y20) ==")
    show(C_[(C_.family != "R") & (C_.variant == "all") & C_.window.isin(["pre", "post"]) & C_.period.isin(["ALL", "Y2026"]) & C_.outcome.isin(["y1", "y20"])])
    print("\n== 포착 주 한정(사후) ==")
    print(cov_t.round(4).to_string(index=False))
    scol = ["type", "period", "family", "variant", "window", "outcome", "n1", "n0", "dates_with", "dates", "m1", "m0", "diff",
            "boot_lo", "boot_hi", "null_q95", "p_rot", "p_low", "v1", "v0", "auc_cont", "judgeable", "passed"]
    T_ = st_cells
    print("\n== 주식 R(all, any·pre·post, 기간 전부) ==")
    print(T_[(T_.family == "R") & (T_.variant == "all") & T_.window.isin(["any", "pre", "post"])][scol].round(4).to_string(index=False))
    print("\n== 주식 R first·민감도(pre, 기간 3) ==")
    print(T_[(T_.family == "R") & (T_.variant != "all") & (T_.window == "pre") & T_.period.isin(REG_PERIODS)][scol].round(4).to_string(index=False))
    print("\n== 주식 D·S·T pre(all, 기간 3) ==")
    print(T_[(T_.family != "R") & (T_.variant == "all") & (T_.window == "pre") & T_.period.isin(REG_PERIODS)][scol].round(4).to_string(index=False))
    print("\n== 확인 지연(사후) ==")
    print(lead.round(4).to_string(index=False))
    print(base.round(4).to_string(index=False))
    print("\n== 확정 쪽: 극점 당일~확정 세션 사이 첫 노트(시가 행동) ==")
    g = cs.groupby(["det", "type", "family"]).agg(n=("conf_date", "size"), with_note=("first_a", lambda s: int(s.notna().sum())),
                                                  saved_med=("sessions_saved", lambda s: float(s[s > 0].median()) if (s > 0).any() else np.nan),
                                                  delay_med=("leg_delay", "median"), gain_mean=("gain_vs_wait", "mean"),
                                                  gain_pos=("gain_vs_wait", lambda s: int((s > 0).sum())))
    print(g.round(4).to_string())
    print("\n== SOXX 사건별 표지 ==")
    ecols = ["det", "type", "conf_date", "pivot_date", "S1", "leg_move", "r20", "exit_date", "censored", "y1", "y20", "R_all_pre", "R_all_post",
             "R_pre_ids", "R_post_ids", "D_all_pre", "S_all_pre", "T_all_pre", "gs_cov_pre"]
    e2 = ev[ecols].copy()
    for c_ in ("conf_date", "pivot_date", "exit_date"):
        e2[c_] = pd.to_datetime(e2[c_]).dt.date
    print(e2.round(4).to_string(index=False))
    print("\n== R 사전 대 없음: 확정 시점 이미 움직인 크기(설명용) ==")
    print(ev.groupby(["det", "type", "R_all_pre"])[["S1", "leg_move", "r20", "y1", "y20"]].median().round(4).to_string())


if __name__ == "__main__":
    main()
