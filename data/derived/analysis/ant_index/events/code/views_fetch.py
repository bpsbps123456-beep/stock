"""부록 13 조회·추천 특징 1단계: 줄(수집 한 번)마다 수치 필드와 '수집 시각' 추정. 가격을 읽지 않는다.

입력(읽기만):
  data/derived/desk/ant/posts/*.jsonl  — snapshot_1023/snapshot_manifest.json의 바이트만(SHA-256 확인)
  data/derived/desk/ant/collect.log     — '시각 · 번호' 줄(거꾸로 받기 진행)
  snapshot_1023/results/units.parquet   — 시각만 댓글의 날짜(로더가 matched/row로 확정한 것만 쓴다)
  (부록 14 갱신: 스냅샷·산출 폴더만 snap_paths.SNAP으로 바꿈. 10/22 진단판(final 스냅샷)은 events/results/views_*와 views_leak.md)
  desk/ant.py의 규칙(코드 상수): 5초 간격, 40건 묶음, 25묶음마다 로그, 3시간마다 따라잡기, 등록 24시간 뒤 다시 받기

수집 시각 추정(줄마다 T̂, 하한 T_lo, 상한 T_hi):
  거꾸로 받은 줄(backward): 로그 점(시각, 그때 low 번호)을 실행별로 잇고 번호로 직선 보간한다.
    실행 경계 = 로그의 '저장' 수가 다시 작아진 곳. 실행 첫 1,000번호(첫 로그 전)는 순수 받기 속도로 거꾸로 늘리고,
    실행 마지막 로그 뒤 번호는 같은 속도로 앞으로 늘린다. 상·하한은 앞뒤 로그 시각.
  실시간 줄(forward = 실행 1 시작 번호 위의 첫 줄, recheck = 둘째 줄):
    하한 LB = max(글 시각, 줄 안 댓글 시각 최대). 시각만 댓글은 그 줄 수집 날짜의 댓글이다(원천 규칙).
    상한 UB = 다음 줄에만 있는 댓글(작성자·시각이 첫 줄에 없음) 중 가장 이른 시각.
    순서 제약: 같은 파일 안 줄 순서 = 수집 순서, forward는 번호 순, recheck는 (등록+24h) 순으로 수집 시각이 줄지 않는다.
    → LB는 앞으로 누적 최대, UB는 뒤로 누적 최소. 자기 줄 누적 하한보다 이른 UB는 숨김·차단 해제 댓글로 보고 버린다.
    forward 줄을 번호 순으로 구간 교집합(탐욕)으로 묶어 '따라잡기 사건'을 만들고, recheck 줄은
    max(등록+24h, LB) 뒤 첫 사건에 놓는다.
산출: snapshot_1023/results/views_rows.parquet(줄 표), views_fetch_meta.json(로그·실행·검증 수치)
실행: py views_fetch.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import snap_paths as SP

HERE = Path(__file__).resolve().parent
EV = HERE.parent
ANT = EV.parent
SNAP = SP.SNAP                   # 부록 14: 사건 연구 스냅샷(2026-10-23 07:59, final/과 같은 꼴)
OUT = SP.SNAP_RES
ROOT = ANT.parents[3]
ANT_RAW = ROOT / "data" / "derived" / "desk" / "ant"
POSTS = ANT_RAW / "posts"
LOG = ANT_RAW / "collect.log"

FULL_RX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
TOD_RX = re.compile(r"^\d{2}:\d{2}:\d{2}$")
LOG_RX = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+번호 ([\d,]+) \((\S+)\)\s+저장 ([\d,]+)\s+없음 ([\d,]+)\s+실패 ([\d,]+)")
LOG_SPAN = 1000          # desk/ant.py: CHUNK 40 × 25묶음마다 로그
SETTLE = pd.Timedelta(hours=24)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- 1. 줄 읽기

def read_rows(manifest: dict, units_path: Path = SP.SNAP_RES / "units.parquet") -> tuple[pd.DataFrame, list, dict]:
    """스냅샷 바이트만 읽어 줄 표(수치 필드)와 줄별 댓글 목록 [(작성자, 'HH:MM:SS', 시각|None, 출처)]을 만든다."""
    u = pd.read_parquet(units_path, columns=["no", "utype", "a", "text", "dt_kst", "dt_conf"])
    u = u[u["utype"] == "comment"]
    look: dict = defaultdict(list)
    for no, a, tx, dt, cf in zip(u["no"].to_numpy(), u["a"].to_numpy(), u["text"].to_numpy(),
                                 u["dt_kst"].to_numpy(), u["dt_conf"].to_numpy()):
        ts = pd.Timestamp(dt)
        look[(int(no), a, tx, ts.strftime("%H:%M:%S"))].append((ts, cf))
    del u
    recs, cmts = [], []
    stat = defaultdict(int)
    for name, spec in sorted(manifest["files"].items()):
        with open(POSTS / name, "rb") as fh:
            data = fh.read(spec["bytes"])
        if sha(data) != spec["sha256"]:
            raise ValueError(f"{name}: 스냅샷 SHA-256 불일치")
        for i, line in enumerate(data.split(b"\n")):
            if not line.strip():
                continue
            r = json.loads(line.decode("utf-8"))
            cs = []
            for c in r["comments"]:
                d = c["dt"]
                if FULL_RX.match(d):
                    cs.append((c["a"], d[11:], pd.Timestamp(d), "full"))
                elif TOD_RX.match(d):
                    good = [x for x in look.get((r["no"], c["a"], c["text"], d), ()) if x[1] in ("matched", "row")]
                    if good:
                        cs.append((c["a"], d, min(x[0] for x in good), "tod_ok"))
                    else:
                        cs.append((c["a"], d, None, "tod_unk"))
                else:
                    stat["comment_dt_unknown"] += 1
            for c in cs:
                stat["c_" + c[3]] += 1
            recs.append({"file": name[:10], "line": i, "no": int(r["no"]), "dt": r["dt"],
                         "views": int(r.get("views") or 0), "up": int(r.get("up") or 0),
                         "down": int(r.get("down") or 0), "ncomment": int(r.get("ncomment") or 0),
                         "nsaved": len(r["comments"]),
                         "nroot": sum(1 for c in r["comments"] if int(c.get("depth") or 0) == 0)})
            cmts.append(cs)
    df = pd.DataFrame(recs)
    df["dt"] = pd.to_datetime(df["dt"])
    df["ridx"] = np.arange(len(df))
    return df, cmts, dict(stat)


def row_bounds(df: pd.DataFrame, cmts: list) -> pd.DataFrame:
    """줄마다 k(그 글의 몇 번째 줄), 증거 하한 LB0, 다음 줄 새 댓글 상한 UB0, 수집 날짜(시각만 댓글로 확정되면)."""
    df = df.sort_values(["file", "line"]).reset_index(drop=True)
    df["k"] = df.groupby("no").cumcount()
    df["n_rows"] = df.groupby("no")["no"].transform("size")
    nxt = {}
    for no, g in df.groupby("no"):
        rr = g["ridx"].tolist()
        for a, b in zip(rr, rr[1:]):
            nxt[a] = b
    lb, ub, dr, n_new, ub_src_ok = [], [], [], [], []
    for ridx, P in zip(df["ridx"].to_numpy(), df["dt"]):
        cs = cmts[ridx]
        full = [c[2] for c in cs if c[3] == "full"]
        tods = [c[1] for c in cs if c[3] != "full"]
        okd = {c[2].normalize() for c in cs if c[3] == "tod_ok"}
        lo = P.normalize()
        if tods and min(tods) < P.strftime("%H:%M:%S"):
            lo += pd.Timedelta(days=1)
        if full:
            lo = max(lo, max(full).normalize() + pd.Timedelta(days=1))
        d_r = max(okd) if okd else None
        dd = d_r if d_r is not None else lo   # 날짜를 못 정한 시각만 댓글은 가능한 가장 이른 날짜(하한이 되도록)
        # 날짜가 붙은 댓글은 수집 날짜보다 앞선 날의 댓글이다(원천 규칙) → 수집 시각 ≥ lo(그 다음 날 0시)
        cand = [P, lo] + full + ([dd + pd.Timedelta(max(tods))] if tods else [])
        lb.append(max(cand))
        dr.append(d_r)
        u_ = pd.NaT
        nn = 0
        if ridx in nxt:
            prev = {(c[0], c[1]) for c in cs}
            new = [c[2] for c in cmts[nxt[ridx]] if (c[0], c[1]) not in prev and c[2] is not None]
            nn = len(new)
            if new:
                u_ = min(new)
        ub.append(u_)
        n_new.append(nn)
    df["LB0"] = pd.to_datetime(pd.Series(lb))
    df["UB0_raw"] = pd.to_datetime(pd.Series(ub))
    df["coll_date_cmt"] = pd.to_datetime(pd.Series(dr))
    df["n_new_next"] = n_new
    return df


# ---------------------------------------------------------------- 2. 로그 → 거꾸로 받기 시각

def parse_log(path: Path = LOG) -> pd.DataFrame:
    recs = []
    for line in path.read_text("utf-8").splitlines():
        m = LOG_RX.match(line)
        if m:
            recs.append({"t": pd.Timestamp(m.group(1)), "low": int(m.group(2).replace(",", "")),
                         "min_date": m.group(3), "saved": int(m.group(4).replace(",", "")),
                         "missing": int(m.group(5).replace(",", "")), "fail": int(m.group(6).replace(",", ""))})
    lg = pd.DataFrame(recs)
    lg["run"] = (lg["saved"].diff() < 0).cumsum()
    return lg


def log_model(lg: pd.DataFrame) -> dict:
    """실행별 점과 순수 받기 속도(초/번호). 순수 속도 = 로그 간격(1,000번호) 길이의 하위 25%."""
    gaps = []
    for _, g in lg.groupby("run"):
        dt = g["t"].diff().dt.total_seconds().dropna() / LOG_SPAN
        gaps += dt.tolist()
    gaps = np.array(gaps)
    rate = float(np.quantile(gaps, 0.25))
    runs = []
    for r, g in lg.groupby("run"):
        g = g.sort_values("t")
        runs.append({"run": int(r), "L0": int(g["low"].iloc[0]) + LOG_SPAN, "first_t": g["t"].iloc[0],
                     "last_t": g["t"].iloc[-1], "first_low": int(g["low"].iloc[0]), "last_low": int(g["low"].iloc[-1]),
                     "t": g["t"].tolist(), "low": g["low"].tolist()})
    for rr in runs:
        rr["start_t_est"] = rr["first_t"] - pd.Timedelta(seconds=LOG_SPAN * rate)
    for i, rr in enumerate(runs):
        rr["end_low"] = runs[i + 1]["L0"] if i + 1 < len(runs) else -1   # 다음 실행이 이어 받은 번호(그 위까지가 이 실행)
        rr["next_start_t"] = runs[i + 1]["start_t_est"] if i + 1 < len(runs) else None
        rr["prev_last_t"] = runs[i - 1]["last_t"] if i > 0 else None
    return {"rate_s_per_no": rate, "gap_quantiles_s_per_no": {q: float(np.quantile(gaps, q)) for q in (0, .25, .5, .75, 1)},
            "runs": runs}


def backward_time(no: np.ndarray, model: dict, S: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """번호 → (T̂, 하한, 상한, 실행). 로그 점은 그 묶음 끝 시각(분 단위 내림)이라 +30초를 점 시각으로 둔다."""
    rate = pd.Timedelta(seconds=model["rate_s_per_no"])
    half = pd.Timedelta(seconds=30)
    T = np.full(len(no), np.datetime64("NaT"), dtype="datetime64[s]")
    lo, hi = T.copy(), T.copy()
    run = np.full(len(no), -1)
    for rr in model["runs"]:
        sel = (no < rr["L0"]) & (no >= rr["end_low"]) if rr["end_low"] >= 0 else (no < rr["L0"])
        if not sel.any():
            continue
        idx = np.flatnonzero(sel)
        lows = np.array(rr["low"][::-1], dtype=float)            # 오름차순 번호
        ts = np.array([t + half for t in rr["t"][::-1]], dtype="datetime64[s]").astype("int64").astype(float)
        n = no[idx].astype(float)
        est = np.interp(n, lows, ts)                               # 안쪽: 직선 보간(번호가 클수록 이른 시각)
        first_t = (rr["t"][0] + half).value / 1e9
        last_t = (rr["t"][-1] + half).value / 1e9
        a = n > rr["first_low"]
        est[a] = first_t - (n[a] - rr["first_low"]) * rate.total_seconds()
        b = n < rr["last_low"]
        est[b] = last_t + (rr["last_low"] - n[b]) * rate.total_seconds()
        T[idx] = est.astype("int64").astype("datetime64[s]")
        # 상·하한: 감싸는 로그 시각. lows는 오름차순, tl은 같은 순서의 로그 시각(번호가 클수록 이른 시각).
        # j = lows[j-1] < n <= lows[j]: n은 lows[j] 로그 '뒤'(하한 tl[j]), lows[j-1] 로그 '전'(상한 tl[j-1] + 1분)에 받았다.
        tl = np.array(rr["t"][::-1], dtype="datetime64[s]")
        j = np.searchsorted(lows, n, side="left")
        nat = np.datetime64("NaT")
        lo_t = np.where(j < len(lows), tl[np.clip(j, 0, len(tl) - 1)], nat)
        hi_t = np.where(j >= 1, tl[np.clip(j - 1, 0, len(tl) - 1)] + np.timedelta64(60, "s"), nat)
        eq_last = n == lows[0]                                     # 마지막 로그 번호 그 자체
        hi_t = np.where(eq_last, tl[0] + np.timedelta64(60, "s"), hi_t)
        # 첫 로그 위(실행 시작 구간): 하한 = 직전 실행 마지막 로그(없으면 추정 시작 − 1시간)
        start_lo = rr["prev_last_t"] if rr["prev_last_t"] is not None else rr["start_t_est"] - pd.Timedelta(hours=1)
        lo_t = np.where(a, np.datetime64(start_lo.to_datetime64(), "s"), lo_t)
        # 마지막 로그 아래(실행 끝 구간): 상한 = 다음 실행 추정 시작(없으면 S)
        end_hi = rr["next_start_t"] if rr["next_start_t"] is not None else S
        hi_t = np.where(b, np.datetime64(end_hi.to_datetime64(), "s"), hi_t)
        lo[idx], hi[idx] = lo_t, hi_t
        run[idx] = rr["run"]
    return T, lo, hi, run


# ---------------------------------------------------------------- 3. 실시간 줄: 순서 제약 + 따라잡기 사건

def propagate(L: np.ndarray, U: np.ndarray, chains: list[np.ndarray], iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """사슬마다 수집 시각이 줄지 않는다: L은 앞으로 누적 최대, U는 뒤로 누적 최소. 고정점까지 반복."""
    L, U = L.copy(), U.copy()
    for _ in range(iters):
        L0, U0 = L.copy(), U.copy()
        for ch in chains:
            L[ch] = np.maximum.accumulate(L[ch])
            U[ch] = np.minimum.accumulate(U[ch][::-1])[::-1]
        if (L0 == L).all() and (U0 == U).all():
            break
    return L, U


def stab_events(L: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
    """번호 순 forward 구간을 앞에서부터 교집합이 빌 때까지 한 사건으로 묶는다(최소 사건 수 탐욕). 반환: 사건 번호, [(lo, hi)]."""
    ev = np.zeros(len(L), dtype=int)
    out = []
    lo, hi, k = L[0], U[0], 0
    for i in range(1, len(L)):
        nlo, nhi = max(lo, L[i]), min(hi, U[i])
        if nlo <= nhi:
            lo, hi = nlo, nhi
        else:
            out.append((lo, hi))
            k += 1
            lo, hi = L[i], U[i]
        ev[i] = k
    out.append((lo, hi))
    return ev, out


def schedule(anchors: np.ndarray, wide: list[tuple], active: list[tuple], period_s: float, max_gap_s: float,
             S64: np.datetime64) -> tuple[np.ndarray, np.ndarray]:
    """따라잡기 시각 목록 = 증거로 잡힌 사건(anchors) + 빈 곳을 주기로 채운 가상 사건.
    active: 수집기가 돌던 구간 [(시작, 끝)]. 구간 안에서 이웃 사건 간격이 max_gap_s를 넘으면 period_s 간격으로 채우고,
    마지막 사건 뒤로는 끝(또는 S)까지 채운다. 넓은 증거 사건(wide: (lo, hi))과 겹치는 가상 사건은 그 범위로 자른다."""
    pts = sorted(set(anchors.tolist()))
    virt = []
    per = np.timedelta64(int(period_s), "s")
    for st, en in active:
        en = min(en, S64)
        inside = [np.datetime64(t, "s") for t in pts if st <= np.datetime64(t, "s") <= en]
        seq = inside if inside else [st]
        cur = seq[0]
        for nxt in seq[1:] + [None]:
            limit = nxt if nxt is not None else en
            gap = (limit - cur) / np.timedelta64(1, "s")
            if nxt is None:
                t = cur + per
                while t < en:
                    virt.append(t)
                    t = t + per
            elif gap > max_gap_s:
                m = max(int(round(gap / period_s)) - 1, 1)
                step = (limit - cur) / (m + 1)
                for i in range(1, m + 1):
                    virt.append(cur + step * i)
            if nxt is not None:
                cur = nxt
    vv = [np.datetime64(t, "s") for t in virt]
    anc = [np.datetime64(t, "s") for t in pts]
    # 넓은 증거 사건 = 그 범위 안에 따라잡기가 한 번 있었다. 이미 안에 점이 있으면 두고, 없으면 가장 가까운 가상 점 하나만 끌어온다
    for lo, hi in wide:
        if any(lo <= t <= hi for t in anc + vv) or not vv:
            continue
        d = [abs((min(max(t, lo), hi) - t) / np.timedelta64(1, "s")) for t in vv]
        i = int(np.argmin(d))
        if d[i] <= period_s / 2:
            vv[i] = min(max(vv[i], lo), hi)
    times = np.array(anc + vv, dtype="datetime64[s]")
    is_virtual = np.array([False] * len(pts) + [True] * len(vv))
    o = np.argsort(times, kind="stable")
    return times[o], is_virtual[o]


ANCHOR_MAX_W = 60 * 60       # 증거 사건 폭이 이 안이면 따라잡기 시각으로 믿는다(초)
PERIOD_DEFAULT = 3.2 * 3600  # 증거 사건이 모자랄 때만 쓰는 기본 간격(3시간 + 받기 시간)
MAX_GAP = 4.2 * 3600         # 이 간격을 넘으면 사이에 놓친 따라잡기가 있다고 본다


def realtime_times(df: pd.DataFrame, fwd0: int, S: pd.Timestamp, model: dict):
    S64 = np.datetime64(S.to_datetime64(), "s")
    one = np.timedelta64(1, "s")
    cls = np.where(df["k"] >= 1, "recheck", np.where(df["no"] >= fwd0, "forward", "backward"))
    df["cls"] = cls
    df["due"] = df["dt"] + SETTLE
    isb = (df["cls"] == "backward").to_numpy()
    isr = (df["cls"] == "recheck").to_numpy()
    isf = (df["cls"] == "forward").to_numpy()
    rt = df.index[~isb].to_numpy()
    L = df["LB0"].to_numpy().astype("datetime64[s]").copy()
    L[isr] = np.maximum(L[isr], df["due"].to_numpy().astype("datetime64[s]")[isr])
    U_raw = df["UB0_raw"].to_numpy().astype("datetime64[s]")
    # 시각만 댓글로 정해진 수집 날짜: 수집 시각은 그날 안
    cd = df["coll_date_cmt"].to_numpy().astype("datetime64[s]")
    hasd = ~np.isnat(cd)
    L = np.where(hasd, np.maximum(L, cd), L)
    Ud = np.where(hasd, cd + np.timedelta64(86400, "s") - one, S64)
    chains = [g.sort_values("line").index.to_numpy() for _, g in df.loc[rt].groupby("file")]
    fwi = df.index[isf].to_numpy()
    fwi = fwi[np.argsort(df.loc[fwi, "no"].to_numpy(), kind="stable")]
    chains.append(fwi)
    chains.append(df[isr].sort_values(["due", "no"]).index.to_numpy())
    # 1) 하한만 전파 → 자기 누적 하한 이하의 상한(숨김·차단 해제·시각을 고친 댓글)은 버린다
    Lp, _ = propagate(L, Ud.copy(), chains)
    ub_ok = ~np.isnat(U_raw) & (U_raw > Lp)
    dropped = int((~np.isnat(U_raw) & ~ub_ok & ~isb).sum())
    U = np.minimum(np.where(ub_ok, U_raw, S64), Ud)
    # 거꾸로 받은 줄이 섞인 파일(실행 1 시작의 10/12~13): 그 줄의 로그 상·하한도 같은 파일 사슬에 넣는다
    L = np.where(isb, np.maximum(df["T_lo_b"].to_numpy().astype("datetime64[s]"), L), L)
    U = np.where(isb, np.minimum(df["T_hi_b"].to_numpy().astype("datetime64[s]"), U), U)
    allch = list(chains)
    for _, g in df.groupby("file"):
        if (g["cls"] != "backward").any() and (g["cls"] == "backward").any():
            allch.append(g.sort_values("line").index.to_numpy())
    Lp, Up = propagate(L, U, allch)
    infeasible = int(((Lp > Up) & ~isb).sum())
    # 2) 증거 사건: forward 구간을 번호 순으로 교집합 묶음
    evid, evs = stab_events(Lp[fwi], Up[fwi])
    ev_lo = np.array([e[0] for e in evs], dtype="datetime64[s]")
    ev_hi = np.array([e[1] for e in evs], dtype="datetime64[s]")
    ev_w = (ev_hi - ev_lo) / one
    ev_mid = ev_lo + (ev_hi - ev_lo) // 2
    anchor = ev_w <= ANCHOR_MAX_W
    # 수집기가 돌던 구간(로그 실행): 시작 추정 − 30분 ~ 마지막 로그 + 남은 묶음
    rate = model["rate_s_per_no"]
    active = []
    for r in model["runs"]:
        st = np.datetime64((r["start_t_est"] - pd.Timedelta(minutes=30)).to_datetime64(), "s")
        en = r["last_t"] + pd.Timedelta(seconds=(r["last_low"] - r["end_low"]) * rate + 60) if r["end_low"] >= 0 else S
        active.append((st, np.datetime64(en.to_datetime64(), "s")))
    wide = [(lo_, hi_) for lo_, hi_, a in zip(ev_lo, ev_hi, anchor) if not a]
    am0 = np.sort(ev_mid[anchor])
    g0 = np.diff(am0) / np.timedelta64(1, "s")
    g0 = g0[(g0 >= 2.5 * 3600) & (g0 <= 4.0 * 3600)]
    period = float(np.median(g0)) if len(g0) >= 10 else PERIOD_DEFAULT   # 따라잡기 간격 = 증거 사건 간격(2.5~4시간) 중앙값
    sched, virt = schedule(ev_mid[anchor], wide, active, period, MAX_GAP, S64)
    # 3) 배정: forward = 누적 하한 뒤 첫 따라잡기, recheck = max(등록+24h, 하한) 뒤 첫 따라잡기. 누적 상한으로 자른다
    T = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[s]")
    method = np.full(len(df), "", dtype=object)
    on_virtual = np.zeros(len(df), dtype=bool)
    for sel, name in ((isf, "forward"), (isr, "recheck")):
        idx = np.flatnonzero(sel)
        need = Lp[idx]
        j = np.searchsorted(sched, need, side="left")
        ok = j < len(sched)
        jj = np.clip(j, 0, len(sched) - 1)
        t = np.where(ok, sched[jj], need)
        T[idx] = np.minimum(np.maximum(t, need), Up[idx])
        on_virtual[idx] = ok & virt[jj]
        method[idx] = np.where(~ok, f"{name}_lb", np.where(virt[jj], f"{name}_virtual", f"{name}_anchor"))
    df["T_rt"], df["T_rt_lo"], df["T_rt_hi"], df["method_rt"] = T, Lp, Up, method
    ev_tab = pd.DataFrame({"t": sched, "virtual": virt})
    ev_tab["n_forward"] = [int((T[isf] == t).sum()) for t in sched]
    ev_tab["n_recheck"] = [int((T[isr] == t).sum()) for t in sched]
    ev_tab["gap_h"] = pd.Series(sched).diff().dt.total_seconds().to_numpy() / 3600
    grp = pd.DataFrame({"lo": ev_lo, "hi": ev_hi, "w_min": ev_w / 60, "anchor": anchor,
                        "n_rows": np.bincount(evid, minlength=len(evs))})
    am = np.sort(ev_mid[anchor])
    gaps_h = np.diff(am) / np.timedelta64(1, "h")
    info = {"ub_dropped_conflict": dropped, "ub_used": int((ub_ok & ~isb).sum()), "infeasible_rows": infeasible,
            "n_forward": int(isf.sum()), "n_recheck": int(isr.sum()), "evidence_groups": int(len(evs)),
            "anchor_groups": int(anchor.sum()),
            "wide_groups": [{"lo": str(a_), "hi": str(b_), "rows": int(n_)}
                            for a_, b_, n_, an in zip(ev_lo, ev_hi, grp["n_rows"], anchor) if not an],
            "schedule_events": int(len(sched)), "virtual_events": int(virt.sum()),
            "rows_on_virtual": {"forward": int((on_virtual & isf).sum()), "recheck": int((on_virtual & isr).sum())},
            "anchor_gap_h_quantiles": {str(q): round(float(np.quantile(gaps_h, q)), 2) for q in (.1, .25, .5, .75, .9)},
            "period_h": round(period / 3600, 4), "max_gap_h": MAX_GAP / 3600, "anchor_max_w_min": ANCHOR_MAX_W / 60}
    return ev_tab, info, grp


def main() -> int:
    man = json.loads((SNAP / "snapshot_manifest.json").read_text("utf-8"))
    S = pd.Timestamp(man["S"])
    df, cmts, cstat = read_rows(man)
    df = row_bounds(df, cmts)
    lg = parse_log()
    model = log_model(lg)
    fwd0 = model["runs"][0]["L0"]
    Tb, lob, hib, runb = backward_time(df["no"].to_numpy(), model, S)
    df["T_b"], df["T_lo_b"], df["T_hi_b"], df["run"] = Tb, lob, hib, runb
    ev_tab, rinfo, grp = realtime_times(df, fwd0, S, model)
    isb = df["cls"] == "backward"
    df["T"] = np.where(isb, df["T_b"], df["T_rt"])
    df["T_lo"] = np.where(isb, df["T_lo_b"], df["T_rt_lo"])
    df["T_hi"] = np.where(isb, df["T_hi_b"], df["T_rt_hi"])
    df["method"] = np.where(isb, np.where(df["no"] > df["run"].map({r["run"]: r["first_low"] for r in model["runs"]}),
                                          "log_extrap_start", "log_interp"), df["method_rt"])
    for c in ("T", "T_lo", "T_hi"):
        df[c] = pd.to_datetime(df[c])
    # 사건 표에 번호 범위(지운 번호도 받는 데 5초가 든다)를 붙인다
    fwd = df[df["cls"] == "forward"]
    ev_tab["fwd_no_span"] = [int(g["no"].max() - g["no"].min() + 1) if len(g) else 0
                             for t in ev_tab["t"] for g in [fwd[fwd["T"] == t]]]
    ev_tab["pred_minutes"] = (ev_tab["fwd_no_span"] + ev_tab["n_recheck"]) * model["rate_s_per_no"] / 60
    # 검증 A: 거꾸로 받기 로그 간격의 '남는 시간'(간격 − 1,000번호 × 순수 속도) = 그 사이 따라잡기에 쓴 시간.
    # 따라잡기 사건이 든 간격만 남는 시간이 있어야 하고, 남는 시간은 그 사건들의 받은 건수 × 5초와 비슷해야 한다
    iv = []
    for rr in model["runs"]:
        for t0, t1 in zip(rr["t"][:-1], rr["t"][1:]):
            ex = (t1 - t0).total_seconds() / 60 - LOG_SPAN * model["rate_s_per_no"] / 60
            inside = ev_tab[(ev_tab["t"] > t0) & (ev_tab["t"] <= t1)]
            iv.append({"t0": t0, "t1": t1, "excess_min": ex, "n_events": int(len(inside)),
                       "n_virtual": int(inside["virtual"].sum()), "pred_min": float(inside["pred_minutes"].sum())})
    iv = pd.DataFrame(iv)
    anc_iv = iv[(iv["n_events"] > 0) & (iv["n_virtual"] == 0)]
    v_log = {"intervals": int(len(iv)),
             "excess_ge5_with_event": int(((iv["excess_min"] >= 5) & (iv["n_events"] > 0)).sum()),
             "excess_ge5_without_event": int(((iv["excess_min"] >= 5) & (iv["n_events"] == 0)).sum()),
             "excess_lt2_with_event": int(((iv["excess_min"] < 2) & (iv["n_events"] > 0)).sum()),
             "excess_lt2_with_event_pred_min_median": float(iv.loc[(iv["excess_min"] < 2) & (iv["n_events"] > 0), "pred_min"].median())
             if ((iv["excess_min"] < 2) & (iv["n_events"] > 0)).any() else None,
             "anchor_intervals": int(len(anc_iv)),
             "anchor_excess_vs_pred_spearman": float(anc_iv[["excess_min", "pred_min"]].corr("spearman").iloc[0, 1]) if len(anc_iv) > 5 else None,
             "anchor_excess_minus_pred_median_min": float((anc_iv["excess_min"] - anc_iv["pred_min"]).median()) if len(anc_iv) else None,
             "anchor_excess_minus_pred_mad_min": float((anc_iv["excess_min"] - anc_iv["pred_min"]).abs().median()) if len(anc_iv) else None}
    iv.to_csv(OUT / "views_fetch_log_intervals.csv", index=False, encoding="utf-8-sig")
    # 검증: 증거 하한(댓글)보다 이른 추정, 시각만 댓글로 정해진 수집 날짜와 추정 날짜 불일치
    v = {"log_excess": v_log}
    young_b = (df["cls"] == "backward") & ((df["T"] - df["dt"]).dt.total_seconds() < 3 * 86400) & (df["LB0"] > df["dt"])
    gap_b = (df.loc[young_b, "T"] - df.loc[young_b, "LB0"]).dt.total_seconds() / 60
    v["young_backward_T_minus_last_comment_min"] = {"n": int(young_b.sum()),
                                                    **{str(q): round(float(gap_b.quantile(q)), 1) for q in (.05, .25, .5)}}
    lbv = (df["T"] < df["LB0"])
    v["T_before_comment_lb"] = {c: int(lbv[df["cls"] == c].sum()) for c in ("backward", "forward", "recheck")}
    v["T_before_comment_lb_minutes_max"] = {c: float(((df["LB0"] - df["T"]).dt.total_seconds()[lbv & (df["cls"] == c)] / 60).max())
                                           if (lbv & (df["cls"] == c)).any() else 0.0 for c in ("backward", "forward", "recheck")}
    has = df["coll_date_cmt"].notna()
    for c in ("backward", "forward", "recheck"):
        m = has & (df["cls"] == c)
        v[f"coll_date_check_{c}"] = {"rows_with_date": int(m.sum()),
                                     "same_date": int((df.loc[m, "T"].dt.normalize() == df.loc[m, "coll_date_cmt"]).sum())}
    v["T_after_S"] = int((df["T"] > S).sum())
    width = (df["T_hi"] - df["T_lo"]).dt.total_seconds() / 60
    v["bound_width_min"] = {c: {q: round(float(width[df["cls"] == c].quantile(q)), 1) for q in (.1, .25, .5, .75, .9, .99)}
                            for c in ("backward", "forward", "recheck")}
    keep = ["ridx", "file", "line", "no", "k", "n_rows", "cls", "dt", "views", "up", "down", "ncomment", "nsaved",
            "nroot", "LB0", "UB0_raw", "n_new_next", "coll_date_cmt", "run", "T", "T_lo", "T_hi", "method"]
    out = df[keep].copy()
    out["age_h"] = (out["T"] - out["dt"]).dt.total_seconds() / 3600
    out["age_lo_h"] = (out["T_lo"] - out["dt"]).dt.total_seconds() / 3600
    out["age_hi_h"] = (out["T_hi"] - out["dt"]).dt.total_seconds() / 3600
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "views_rows.parquet", index=False)
    ev_tab.to_parquet(OUT / "views_fetch_events.parquet", index=False)
    grp.to_parquet(OUT / "views_fetch_groups.parquet", index=False)
    meta = {"snapshot": {"S": man["S"], "files": len(man["files"]), "rows": int(len(df)), "posts": int(df["no"].nunique())},
            "comment_occurrences": cstat,
            "log": {"lines": int(len(lg)), "runs": [{k: (str(v_) if isinstance(v_, pd.Timestamp) else v_)
                                                    for k, v_ in r.items() if k not in ("t", "low")} for r in model["runs"]],
                    "rate_s_per_no": model["rate_s_per_no"], "gap_quantiles_s_per_no": model["gap_quantiles_s_per_no"],
                    "forward_from_no": fwd0},
            "realtime": rinfo, "checks": v,
            "class_counts": out["cls"].value_counts().to_dict(),
            "method_counts": out["method"].value_counts().to_dict()}
    (OUT / "views_fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
