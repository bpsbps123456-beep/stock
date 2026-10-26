"""N3 — Codex 기사 31건 다시 보기(설명용, 통과 판정 없음). BRIEF.md §N3.

py code/n3_analyze.py
입력(읽기만): research/news_turning_points_20261025/{prices.csv, events.json, event_outcomes.csv, metrics.json}
            cache/n3_raw.parquet(code/n3_fetch.py, 지그재그용 긴 이력)
            swing_trend/code/swing.py 의 atr·zigzag(그대로 가져다 쓴다)
출력: results/n3_*.csv, results/n3_summary.json

1) 수익 대조: 주 통계 포함(keep) 기사마다
   A = Codex 원래 대조(같은 해 + 직전 20일 고점 대비 낙폭 구간 0~5·5~10·10%+)
   B = A + 직전 5일 SOXX 수익(진입 전날 종가까지) 3분위(그 해 기준)
   5·10·20일(다음 장 시가 → h번째 장 종가) 평균·상승 비율을 대조 칸 평균과 비교.
   기사 날짜 묶음 부트스트랩: 같은 묶음 안에서 진입일 간격 < h 거래일이면(평가 창이 겹치면) 한 묶음.
2) 앞뒤 관계: SOXX 3×ATR20(ZZA3)·8%(ZZP8) 실시간 지그재그 확정일(swing.zigzag)과 기사 진입일.
   확정 c일 종가에 알고 c+1일 시가에 행동 / 기사는 진입일 시가에 행동 — 같은 행동 시점끼리 비교한다.
   극점(사후 저점·고점)은 설명용이다.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent.parent
ROOT = HERE.parents[3]
CODEX = ROOT / "research" / "news_turning_points_20261025"
SWING = HERE.parent / "swing_trend" / "code"
RES = HERE / "results"
RES.mkdir(exist_ok=True)

# 다른 연구 폴더에 캐시를 쓰지 않는다.
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / "cache" / "numba"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(SWING))
from swing import atr, zigzag  # noqa: E402

HS = (5, 10, 20)
B = 20000
RNG = np.random.default_rng(20261026)
BOTTOM = ("D", "S", "R")


# ---------------------------------------------------------------- 1. 일별 표(Codex analyze.py 와 같은 정의)
def daily_table() -> pd.DataFrame:
    import exchange_calendars as xcals

    cal = xcals.get_calendar("XNYS", start="2023-12-01", end="2026-11-30")
    sessions = cal.sessions_in_range("2023-12-01", "2026-10-23").tz_localize(None)
    px = pd.read_csv(CODEX / "prices.csv", header=[0, 1], index_col=0, parse_dates=True)
    op = px["open"].reindex(sessions)
    cl = px["close"].reindex(sessions)
    d = pd.DataFrame(index=sessions)
    d["i"] = np.arange(len(sessions))
    d["year"] = sessions.year
    d["dd20"] = (cl.SOXX / cl.SOXX.rolling(20, min_periods=20).max() - 1).shift(1)
    d["bucket"] = pd.cut(-d.dd20, [-np.inf, .05, .10, np.inf], right=False, labels=[0, 1, 2]).astype("Int64")
    d["r5prev"] = cl.SOXX.pct_change(5, fill_method=None).shift(1)   # 진입 전날 종가까지
    for h in HS:
        d[f"r{h}"] = cl.SOXX.shift(-(h - 1)) / op.SOXX - 1
    d["O"] = op.SOXX
    base = d[(d.year >= 2024) & (d.year <= 2026)].copy()
    # 3분위: 그 해 거래일 전체의 r5prev 분포(대조 정의용, 사후 경계)
    base["t5"] = pd.Series(pd.NA, index=base.index, dtype="Int64")
    cuts = {}
    for y, g in base.groupby("year"):
        q1, q2 = np.quantile(g.r5prev.dropna(), [1 / 3, 2 / 3])
        cuts[int(y)] = (float(q1), float(q2))
        base.loc[g.index, "t5"] = np.where(g.r5prev <= q1, 0, np.where(g.r5prev <= q2, 1, 2))
    return base, cuts


def load_events(base: pd.DataFrame) -> pd.DataFrame:
    ev = pd.read_csv(CODEX / "event_outcomes.csv", encoding="utf-8-sig")
    ev["entry"] = pd.to_datetime(ev.entry)
    # Codex 값 재현 확인
    b = base.reindex(ev.entry)
    for h in HS:
        a, c = ev[f"SOXX_r{h}"].to_numpy(float), b[f"r{h}"].to_numpy(float)
        ok = (np.isnan(a) & np.isnan(c)) | (np.abs(a - c) < 1e-12)
        assert ok.all(), f"r{h} 재현 불일치"
    assert (ev.bucket.to_numpy() == b.bucket.to_numpy()).all()
    ev["t5"] = b.t5.to_numpy()
    ev["r5prev"] = b.r5prev.to_numpy()
    return ev


def control_cells(ev: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ev.iterrows():
        out = dict(id=r.id)
        for h in HS:
            col = f"r{h}"
            yy = base[(base.year == r.year) & base[col].notna()]
            a = yy[yy.bucket == r.bucket]
            bb = a[a.t5 == r.t5]
            for nm, cell in (("A", a), ("B", bb)):
                out[f"{nm}_n{h}"] = len(cell)
                out[f"{nm}_mean{h}"] = cell[col].mean() if len(cell) else np.nan
                out[f"{nm}_up{h}"] = (cell[col] > 0).mean() if len(cell) else np.nan
                out[f"{nm}_dn{h}"] = (cell[col] < 0).mean() if len(cell) else np.nan
            # B10: B 칸이 10일 미만이면 A 로 대신(민감도)
            small = out[f"B_n{h}"] < 10
            for st_ in ("n", "mean", "up", "dn"):
                out[f"B10_{st_}{h}"] = out[f"A_{st_}{h}"] if small else out[f"B_{st_}{h}"]
            out[f"Y_mean{h}"] = yy[col].mean()
            out[f"Y_up{h}"] = (yy[col] > 0).mean()
        rows.append(out)
    return pd.DataFrame(rows)


def clusters(idx: np.ndarray, h: int) -> np.ndarray:
    """진입일 순으로 앞 기사와 간격 < h 거래일이면 같은 묶음(평가 창이 겹친다)."""
    o = np.argsort(idx)
    lab = np.empty(len(idx), int)
    k = 0
    for j, t in enumerate(o):
        if j > 0 and idx[t] - idx[o[j - 1]] >= h:
            k += 1
        lab[t] = k
    return lab


def boot(vals: np.ndarray, lab: np.ndarray) -> tuple[float, float]:
    u = np.unique(lab)
    if len(u) < 2:
        return (np.nan, np.nan)
    groups = [vals[lab == g] for g in u]
    sums = np.array([g.sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    pick = RNG.integers(0, len(u), size=(B, len(u)))
    m = sums[pick].sum(1) / cnts[pick].sum(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def returns_table(ev: pd.DataFrame, ctl: pd.DataFrame) -> pd.DataFrame:
    x = ev.merge(ctl, on="id")
    x = x[x.keep]
    groups = [(f, x[x.family == f]) for f in ("D", "S", "R", "T")]
    groups.append(("DSR", x[x.family.isin(BOTTOM)]))
    groups.append(("DSR_2026", x[x.family.isin(BOTTOM) & (x.year == 2026)]))
    groups.append(("R_2026", x[(x.family == "R") & (x.year == 2026)]))
    rows = []
    for name, g in groups:
        for h in HS:
            gg = g[g[f"SOXX_r{h}"].notna()]
            if not len(gg):
                continue
            r = gg[f"SOXX_r{h}"].to_numpy(float)
            up = (r > 0).astype(float)
            lab = clusters(gg.i.to_numpy(), h)
            for cn in ("A", "B", "B10"):
                cm = gg[f"{cn}_mean{h}"].to_numpy(float)
                cu = gg[f"{cn}_up{h}"].to_numpy(float)
                lo, hi = boot(r - cm, lab)
                ulo, uhi = boot(up - cu, lab)
                rows.append(dict(group=name, h=h, control=cn, n=len(gg), clusters=len(np.unique(lab)),
                                 mean=r.mean(), up=up.mean(), ctl_mean=cm.mean(), ctl_up=cu.mean(),
                                 ex_mean=(r - cm).mean(), ex_mean_lo=lo, ex_mean_hi=hi,
                                 ex_up=(up - cu).mean(), ex_up_lo=ulo, ex_up_hi=uhi,
                                 ctl_cell_min=int(gg[f"{cn}_n{h}"].min()), ctl_cell_med=float(gg[f"{cn}_n{h}"].median())))
    return pd.DataFrame(rows)


def codex_check(ev: pd.DataFrame, ctl: pd.DataFrame) -> dict:
    """Codex metrics.json 의 matched_hit5·matched_mean5 재현."""
    m = json.loads((CODEX / "metrics.json").read_text(encoding="utf-8"))["families"]
    x = ev.merge(ctl, on="id")
    x = x[x.keep & x.SOXX_r5.notna()]
    out = {}
    for f, g in x.groupby("family"):
        hit = (g.A_up5 if f != "T" else g.A_dn5).mean()   # T 의 Codex 적중은 하락 비율
        out[f] = dict(codex_matched_mean5=m[f]["matched_mean5"], ours_mean5=float(g.A_mean5.mean()),
                      codex_matched_hit5=m[f]["matched_hit5"], ours_hit5=float(hit),
                      codex_mean5=m[f]["mean5"], ours_event_mean5=float(g.SOXX_r5.mean()))
    return out


# ---------------------------------------------------------------- 2. 앞뒤 관계
def zz_runs(raw: pd.DataFrame, sym: str) -> dict:
    g = raw[raw.sym == sym].sort_values("date").reset_index(drop=True)
    O, H, L, C = (g[k].to_numpy(np.float64) for k in ("O", "H", "L", "C"))
    a20 = atr(H, L, C, 20)
    out = {}
    for det, args in (("ZZA3", (0.0, 3.0 * a20)), ("ZZP8", (0.08, np.zeros(0)))):
        st, pi, pc, pt = zigzag(C, *args)
        out[det] = dict(st=st, pi=pi, pc=pc, pt=pt)
    out["dates"] = pd.DatetimeIndex(g.date)
    out["O"], out["H"], out["L"], out["C"] = O, H, L, C
    return out


def classify(z: dict, det: str, ie: int, fam: str) -> dict:
    """ie = 기사 진입일(긴 이력 인덱스). 상태는 ie−1 종가까지."""
    st, pi, pc, pt = (z[det][k] for k in ("st", "pi", "pc", "pt"))
    O, H, L = z["O"], z["H"], z["L"]
    want = -1 if fam == "T" else 1          # 기사가 가리키는 확정 방향
    s_prev = int(st[ie - 1])
    in_target = (s_prev == 1) if want == 1 else (s_prev == 0)
    res = dict(state_prev=s_prev)
    # 마지막 하락 확정(D용) — ie−1 이하
    dn = [k for k in range(len(pc)) if pt[k] == -1 and pc[k] <= ie - 1]
    res["since_last_dn"] = ie - (pc[dn[-1]] + 1) if dn else np.nan
    if in_target:
        # 이미 그 방향 확정 뒤에 왔다(후행)
        k = [k for k in range(len(pc)) if pt[k] == want and pc[k] <= ie - 1][-1]
        act = pc[k] + 1
        res.update(cat="후행" if ie > act else "동시", lead=act - ie, pivot_rel=int(pi[k]) - ie,
                   conf_date=z["dates"][pc[k]].date(), pivot_date=z["dates"][pi[k]].date(),
                   leg_delay=act - int(pi[k]),
                   gain=(O[act] / O[ie] - 1) if want == 1 else (O[ie] / O[act] - 1),
                   lag_share=np.nan, path_ext=np.nan)
        return res
    nxt = [k for k in range(len(pc)) if pt[k] == want and pc[k] >= ie - 1 + 1]
    if not nxt:
        res.update(cat="미확정", lead=np.nan, pivot_rel=np.nan, conf_date=None, pivot_date=None,
                   leg_delay=np.nan, gain=np.nan, lag_share=np.nan, path_ext=np.nan)
        return res
    k = nxt[0]
    act = pc[k] + 1
    p = int(pi[k])
    if act >= len(O):
        res.update(cat="확정 뒤 체결 없음", lead=np.nan, pivot_rel=p - ie, conf_date=z["dates"][pc[k]].date(),
                   pivot_date=z["dates"][p].date(), leg_delay=np.nan, gain=np.nan, lag_share=np.nan, path_ext=np.nan)
        return res
    if p <= ie - 1:
        cat = "선행·극점 뒤"     # 극점은 지났고 확정 전: 늦음을 줄여 준 자리
    elif p == ie:
        cat = "선행·극점 당일"
    else:
        cat = "선행·극점 앞"     # 극점이 아직: 더 빠지거나(저점형) 더 오른다(T)
    gain = (O[act] / O[ie] - 1) if want == 1 else (O[ie] / O[act] - 1)
    # 기사 진입 뒤 확정 전까지 반대쪽 최대 움직임(저점형: 장중 최저, T: 장중 최고)
    seg = slice(ie, act)
    path_ext = (L[seg].min() / O[ie] - 1) if want == 1 else (H[seg].max() / O[ie] - 1)
    res.update(cat=cat, lead=act - ie, pivot_rel=p - ie, conf_date=z["dates"][pc[k]].date(),
               pivot_date=z["dates"][p].date(), leg_delay=act - p, gain=gain,
               lag_share=(act - ie) / (act - p) if act > p and p <= ie else np.nan, path_ext=path_ext)
    return res


def state_baseline(z: dict, det: str, idx: np.ndarray) -> pd.DataFrame:
    """같은 상태의 임의 날(2024~2026)에 '확정 기다리기 대신 그날 시가 행동'의 이득. 기사 비교 기준."""
    rows = []
    for want in (1, -1):
        for ie in idx:
            r = classify(z, det, int(ie), "T" if want == -1 else "R")
            if r["cat"].startswith("선행"):
                rows.append(dict(want=want, year=int(z["dates"][ie].year), cat=r["cat"], lead=r["lead"], gain=r["gain"]))
    return pd.DataFrame(rows)


def main():
    base, cuts = daily_table()
    ev = load_events(base)
    ctl = control_cells(ev, base)
    rt = returns_table(ev, ctl)
    rt.to_csv(RES / "n3_returns.csv", index=False, encoding="utf-8-sig")
    chk = codex_check(ev, ctl)

    evx = ev.merge(ctl, on="id")
    keep_cols = ["id", "date", "family", "title", "entry", "keep", "clustered", "year", "bucket", "t5", "r5prev",
                 "SOXX_r5", "SOXX_r10", "SOXX_r20"] + [c for c in ctl.columns if c != "id"]
    evx[keep_cols].to_csv(RES / "n3_events.csv", index=False, encoding="utf-8-sig")
    t5dist = evx[evx.keep].groupby(["family", "t5"]).size().unstack(fill_value=0)

    raw = pd.read_parquet(HERE / "cache" / "n3_raw.parquet")
    tim_rows, agree = [], {}
    zs = {s: zz_runs(raw, s) for s in ("SOXX", "^SOX")}
    # Codex 세션과 긴 이력 날짜가 같은지
    cs = base.index
    for s, z in zs.items():
        dd = z["dates"]
        sub = dd[(dd >= cs[0]) & (dd <= cs[-1])]
        assert len(sub) == len(cs) and (sub == cs).all(), f"{s} 날짜 불일치"
    # SOXX 긴 이력 종가 = Codex 스냅샷 종가 확인
    px = pd.read_csv(CODEX / "prices.csv", header=[0, 1], index_col=0, parse_dates=True)
    zc = pd.Series(zs["SOXX"]["C"], index=zs["SOXX"]["dates"]).reindex(cs)
    close_maxdiff = float((zc / px["close"].SOXX.reindex(cs) - 1).abs().max())

    for _, r in ev.iterrows():
        for s, z in zs.items():
            ie = int(z["dates"].searchsorted(r.entry))
            assert z["dates"][ie] == r.entry
            for det in ("ZZA3", "ZZP8"):
                c = classify(z, det, ie, r.family)
                tim_rows.append(dict(id=r.id, family=r.family, date=r.date, entry=r.entry.date(), keep=r.keep,
                                     sym=s, det=det, **c))
    tim = pd.DataFrame(tim_rows)
    tim.to_csv(RES / "n3_timing.csv", index=False, encoding="utf-8-sig")
    # SOXX 대 ^SOX 분류 일치
    pv = tim.pivot_table(index=["id", "det"], columns="sym", values="cat", aggfunc="first")
    agree = {det: int((pv.xs(det, level="det")["SOXX"] == pv.xs(det, level="det")["^SOX"]).sum()) for det in ("ZZA3", "ZZP8")}

    # Codex 8% 극점 확정일과 ZZP8 확정일 비교(2024~2026)
    cp = pd.read_csv(HERE.parent / "news_turning_points_20261025" / "pivots_8.csv")
    z = zs["SOXX"]
    zz = pd.DataFrame(dict(date=z["dates"][z["ZZP8"]["pi"]].date.astype(str), conf=z["dates"][z["ZZP8"]["pc"]].date.astype(str),
                           kind=np.where(z["ZZP8"]["pt"] == 1, "bottom", "top")))
    cp = cp[cp.date >= "2024-01-01"]
    zz = zz[zz.date >= "2024-01-01"]
    m8 = cp.merge(zz, on=["date", "kind"], how="outer", indicator=True)
    pivot_match = dict(codex=len(cp), zzp8=len(zz), both=int((m8._merge == "both").sum()),
                       conf_same=int((m8[m8._merge == "both"].confirmed_date == m8[m8._merge == "both"].conf).sum()))

    # 같은 상태 임의 날 기준(SOXX)
    zi = zs["SOXX"]
    idx = np.flatnonzero((zi["dates"] >= "2024-01-02") & (zi["dates"] <= "2026-10-23"))
    bl_rows = []
    for det in ("ZZA3", "ZZP8"):
        bl = state_baseline(zi, det, idx)
        bl["det"] = det
        bl_rows.append(bl)
    bl = pd.concat(bl_rows)
    bl.to_csv(RES / "n3_state_baseline.csv", index=False, encoding="utf-8-sig")

    # 요약: SOXX, keep, 유형별
    T = tim[(tim.sym == "SOXX") & tim.keep]
    summ = []
    for det in ("ZZA3", "ZZP8"):
        for fam in ("D", "S", "R", "T"):
            g = T[(T.det == det) & (T.family == fam)]
            lead = g[g.cat.str.startswith("선행")]
            w = 1 if fam != "T" else -1
            b_ = bl[(bl.det == det) & (bl.want == w)]
            row = dict(det=det, family=fam, n=len(g))
            for c_ in ("선행·극점 앞", "선행·극점 당일", "선행·극점 뒤", "동시", "후행", "미확정"):
                row[c_] = int((g.cat == c_).sum())
            row.update(lead_med=float(lead.lead.median()) if len(lead) else np.nan,
                       lead_gain_mean=float(lead.gain.mean()) if len(lead) else np.nan,
                       lead_gain_pos=int((lead.gain > 0).sum()),
                       lead_gain_n=int(lead.gain.notna().sum()),
                       lag_med=float(-g[g.cat == "후행"].lead.median()) if (g.cat == "후행").any() else np.nan,
                       lag_gain_mean=float(g[g.cat == "후행"].gain.mean()) if (g.cat == "후행").any() else np.nan,
                       base_lead_gain_mean=float(b_.gain.mean()), base_lead_gain_pos=float((b_.gain > 0).mean()),
                       base_n=len(b_), base_lead_med=float(b_.lead.median()),
                       since_dn_med=float(g.since_last_dn.median()))
            summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(RES / "n3_timing_summary.csv", index=False, encoding="utf-8-sig")

    out = dict(codex_reproduction=chk, r5_tercile_cuts=cuts, t5_distribution=t5dist.to_dict(),
               zz_sym_agree=agree, close_maxdiff_vs_codex=close_maxdiff, codex_pivot8_vs_zzp8=pivot_match,
               bootstrap=dict(B=B, seed=20261026, cluster="같은 묶음 안 진입일 간격 < h 거래일"),
               cells=dict(returns="유형 4 + 묶음 3(DSR·DSR_2026·R_2026) × 기간 3 × 대조 2(+B10 민감도) × 통계 2",
                          timing="keep 기사 29 × 지그재그 2(ZZA3·ZZP8) × 지수 2(SOXX·^SOX)"))
    (RES / "n3_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(rt.round(4).to_string())
    print(summ.round(4).to_string())
    print(T[["id", "family", "entry", "det", "state_prev", "cat", "lead", "pivot_rel", "pivot_date", "conf_date",
             "leg_delay", "gain", "path_ext", "lag_share", "since_last_dn"]].round(4).to_string())


if __name__ == "__main__":
    main()
