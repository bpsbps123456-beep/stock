"""바닥 사례 비교(설명용) 1단계: 날짜별 지문 표 results/daily.parquet (2026-06-01~10-17, 139일).

판정이 아니다. 동결 가설(HYPOTHESES*.md)과 무관한 사례 비교용 묘사 표이고, 매수·매도 근거가 아니다.

하루 창 = 그 KST 날짜 08:00 ~ 다음 날 08:00 직전(07:59). 동결 W_KR 창(직전 거래일 08:00 ~ 거래일 08:00, 주말·연휴 창은
  72~96시간) 대신 달력 하루 창을 쓴다. 평일 하루 창은 W_KR 창(P = 그날)과 시각이 같다(검증에서 대조).
사건 날짜(KOSPI 종가로 고정, 결과를 보고 바꾸지 않는다): 7월 저점 7/30, 10월 저점(지금까지) 10/15,
  바닥이 아니었던 급락일 6/23·7/2·7/8·7/13. 기준 고점 6/22(7월 국면)·9/23(10월 국면).

열 정의
- d_no: 하루 글 수 = 경계 번호 증가 B(마감) − B(열림)(loader.number_boundary 그대로). n_saved: 그 번호 범위의 저장 글 수.
- n_posts: 하루 창에 쓴 저장 글(units 글). n_comments_preclose: 그 글들에 창 마감(다음 날 08:00) 전에 달린 댓글.
  comments_per_post = n_comments_preclose / n_posts. n_comments_in_win: 자기 시각이 창 안인 댓글(부모 글 날짜 무관, 보조).
- d_no_z20: 직전 20개 하루 창(달력일, 자기 제외, 20개가 다 완전할 때만)의 평균·표준편차(ddof 1)로 잰 z. _lnz20은 ln 판.
- {MC,LL_US,LS,KX,ANG}_A: 사전 v0.2(lexicon_v02_final.load_v02, 측정 통과 묶음) 작성자 몫 = 창 안 유효 단위(뉴스형 글·삭제
  댓글·정치 잡담 제외, feat_text와 같음)에서 그 묶음을 한 번이라도 쓴 고유 작성자 ÷ 유효 단위 고유 작성자. _auth_n = 분자.
- new30_share: 그날 작성자(정치 전용 단위만 쓴 작성자 제외, feat_activity와 같음) 가운데 [창 열림 − 30일, 창 열림) 이력에
  없던 해시의 몫. 이력 시작이 자료 시작(첫 글 2026-05-28 13:36) 앞이면 결측(→ 6/27까지 결측).
  new21_share: 같은 규칙의 21일판(보조, 6/22~6/24를 채우려는 비등록 판).
- ref_*: 미리보기 예측 라벨(preview/results/pred_windows.parquet, 7/1 이후 W_KR 창). 하루 창 열림 시각이 든 W_KR 창의 값을
  붙인다. 창 정의가 달라 참고용이다(금요일·연휴 앞 날짜는 72~96시간 창 하나를 여러 날이 함께 쓴다, ref_pred_win_hours).
- kospi_*: ^KS11 일봉 종가. 비거래일은 kospi_close 결측, kospi_last_close는 직전 종가. kospi_ret = 직전 거래일 대비.
  kospi_dd_ytd = 2026-01-02 이후 누적 최고 종가 대비, kospi_dd_ref = 고정 기준 고점 대비(6/22~9/22 → 6/22, 9/23~ → 9/23).
- rule_drop5_later_lower: 그날 −5% 이하이고 10/16까지 그보다 낮은 종가가 왔는가(규칙 점검용, 뒤 가격을 쓴 묘사 열).
- collection: '소급'(10/13 0시 전에 열린 창, 마감 뒤 며칠~몇 달 뒤 수집) / '실시간'(feat_activity.RT_START 이후).
- complete: 창 마감 ≤ 로더 t_end(10/18 00:00). 10/17 창은 글이 10/18 00:00까지만 있어 16시간 관측.

입력(읽기만): preview/results/units.parquet·windows_kr·feat_activity·feat_text·pred_windows·prices·loader_summary.json·
  loader_salt_continuity.csv, interim/results/lexicon_v02*.json(interim/code 매처 경유).
산출: results/daily.parquet, results/unit_hits.parquet(단위별 묶음 적중, 다음 단계 재사용), results/daily_meta.json.

실행: py daily.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BC = HERE.parent
ANT = BC.parent
PRE = ANT / "preview" / "results"
OUT = BC / "results"
sys.path.insert(0, str(ANT / "interim" / "code"))
import lexicon_v02 as L  # noqa: E402
from feat_activity import RT_START, politics_only  # noqa: E402
from lexicon_v02_final import load_v02  # noqa: E402
from loader import _ns, number_boundary  # noqa: E402

FIRST, LAST = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-10-17")
HIST_FIRST = pd.Timestamp("2026-05-29")      # 5/28 창은 열림(08:00)이 첫 글(13:36)보다 앞이라 경계 번호가 없다
OPEN_H = 8
Z_N = 20
NEW_DAYS = (30, 21)
BUNDLES = ["MC", "LL_US", "LS", "KX", "ANG"]
EVENTS = {
    "2026-07-30": ("bottom_jul", "7월 저점"),
    "2026-10-15": ("bottom_oct_sofar", "10월 저점(지금까지)"),
    "2026-06-23": ("drop_not_bottom", "바닥 아닌 급락"),
    "2026-07-02": ("drop_not_bottom", "바닥 아닌 급락"),
    "2026-07-08": ("drop_not_bottom", "바닥 아닌 급락"),
    "2026-07-13": ("drop_not_bottom", "바닥 아닌 급락"),
}
EXPECT_CLOSE = {"2026-06-22": 9115, "2026-07-30": 5594, "2026-09-23": 7081, "2026-10-15": 5554}
REF_PEAKS = [(pd.Timestamp("2026-06-22"), pd.Timestamp("2026-09-22")), (pd.Timestamp("2026-09-23"), LAST)]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def day_of(ts: pd.Series) -> pd.Series:
    """시각 → 하루 창 날짜(08:00 경계)."""
    return (ts - pd.Timedelta(hours=OPEN_H)).dt.normalize()


def unit_hits(u: pd.DataFrame) -> pd.DataFrame:
    an = load_v02()
    sel = u[(u["wdate"] >= FIRST) & (u["wdate"] <= LAST)]
    recs = []
    for ridx, ut, ti, tx in zip(sel.index, sel["utype"], sel["title"], sel["text"]):
        title = (ti if isinstance(ti, str) else "") if ut == "post" else None
        body = tx if isinstance(tx, str) else ""
        r = an(title, body, ut)
        st = {h.stratum for h in r.hits if L.STRATUM_SCOPE[h.stratum] in ("all", ut)}
        rec = {"uidx": int(ridx), "excl": r.excluded, "pol": politics_only(ti, tx, ut)}
        for b in BUNDLES:
            rec[f"h_{b}"] = b in st
        recs.append(rec)
    h = pd.DataFrame(recs).set_index("uidx")
    h["wdate"] = sel["wdate"]
    return h


def author_history(u: pd.DataFrame) -> dict[str, np.ndarray]:
    t = u["dt_kst"].to_numpy(dtype="datetime64[s]")
    o = np.argsort(t, kind="stable")
    out: dict[str, list] = defaultdict(list)
    for a, ti in zip(u["a"].to_numpy()[o], t[o]):
        out[a].append(ti)
    return {k: np.asarray(v, dtype="datetime64[s]") for k, v in out.items()}


def main() -> int:
    t0 = datetime.now()
    OUT.mkdir(parents=True, exist_ok=True)
    summ = json.loads((PRE / "loader_summary.json").read_text("utf-8"))
    t_end = pd.Timestamp(summ["snapshot"]["t_end"])
    cols = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "title", "text", "trunc_any"]
    u = pd.read_parquet(PRE / "units.parquet", columns=cols)
    for c in ("dt_kst", "post_dt_kst"):
        u[c] = u[c].astype("datetime64[ns]")
    u["wdate"] = day_of(u["dt_kst"])
    u["pwdate"] = day_of(u["post_dt_kst"])
    posts = u[u["utype"] == "post"]
    comm = u[u["utype"] == "comment"]
    c0 = pd.Timestamp(posts["dt_kst"].min())

    days = pd.date_range(HIST_FIRST, LAST, freq="D")
    D = pd.DataFrame({"date": days})
    D["weekday"] = D["date"].dt.weekday
    D["win_open"] = D["date"] + pd.Timedelta(hours=OPEN_H)
    D["win_close"] = D["win_open"] + pd.Timedelta(days=1)
    D["complete"] = D["win_close"] <= t_end
    D["hours_observed"] = ((np.minimum(D["win_close"], t_end) - D["win_open"]).dt.total_seconds() / 3600).round(2)
    D["collection"] = np.where(D["win_open"] >= pd.Timestamp(RT_START), "실시간", "소급")

    # 글 수(번호 증가)·저장 수
    pdt, pno = _ns(posts["dt_kst"]), posts["no"].to_numpy()
    bo, bc = number_boundary(pdt, pno, _ns(D["win_open"])), number_boundary(pdt, pno, _ns(D["win_close"]))
    sno = np.sort(pno)
    D["d_no"] = bc - bo
    D["n_saved"] = np.searchsorted(sno, np.nan_to_num(bc), "right") - np.searchsorted(sno, np.nan_to_num(bo), "right")
    D.loc[~np.isfinite(D["d_no"]), "n_saved"] = np.nan
    D["missing_rate"] = np.where(D["d_no"] > 0, 1 - D["n_saved"] / D["d_no"], np.nan)
    idx = pd.Index(D["date"])
    D["n_posts"] = posts.groupby("wdate").size().reindex(idx).fillna(0).astype(int).to_numpy()
    close_of = dict(zip(D["date"], D["win_close"]))
    pc = comm[comm["pwdate"].isin(set(D["date"]))]
    pc = pc[pc["dt_kst"] < pc["pwdate"].map(close_of)]
    D["n_comments_preclose"] = pc.groupby("pwdate").size().reindex(idx).fillna(0).astype(int).to_numpy()
    D["comments_per_post"] = D["n_comments_preclose"] / D["n_posts"].replace(0, np.nan)
    D["n_comments_in_win"] = comm.groupby("wdate").size().reindex(idx).fillna(0).astype(int).to_numpy()
    D["trunc_share"] = posts.groupby("wdate")["trunc_any"].mean().reindex(idx).to_numpy()

    # 직전 20일 z
    x = D["d_no"].to_numpy(float)
    ok = D["complete"].to_numpy() & np.isfinite(x)
    lx = np.log(np.where(x > 0, x, np.nan))
    z, lz, base = (np.full(len(D), np.nan) for _ in range(3))
    for k in range(len(D)):
        if k < Z_N or not ok[k - Z_N:k].all():
            continue
        h, lh = x[k - Z_N:k], lx[k - Z_N:k]
        base[k] = h.mean()
        z[k] = (x[k] - h.mean()) / h.std(ddof=1)
        lz[k] = (lx[k] - lh.mean()) / lh.std(ddof=1)
    D["d_no_base20_mean"], D["d_no_z20"], D["d_no_lnz20"] = base, z, lz

    # 사전 묶음 작성자 몫
    t1 = datetime.now()
    H = unit_hits(u)
    match_sec = (datetime.now() - t1).total_seconds()
    H.reset_index().to_parquet(OUT / "unit_hits.parquet", index=False)
    V = H[H["excl"] == ""].join(u[["a"]])
    flags = [f"h_{b}" for b in BUNDLES]
    g = V.groupby(["wdate", "a"])[flags].max()
    D["n_auth_valid"] = g.groupby(level=0).size().reindex(idx).to_numpy()
    ratio, cnt = g.groupby(level=0).mean().reindex(idx), g.groupby(level=0).sum().reindex(idx)
    for b in BUNDLES:
        D[f"{b}_A"] = ratio[f"h_{b}"].to_numpy()
        D[f"{b}_auth_n"] = cnt[f"h_{b}"].to_numpy()

    # 새 작성자 몫
    hist = author_history(u)
    ua = H.join(u[["a"]])
    keep = ua.groupby(["wdate", "a"])["pol"].all()
    keep = keep[~keep]
    auth_by_day: dict[pd.Timestamp, list[str]] = defaultdict(list)
    for (d, a) in keep.index:
        auth_by_day[d].append(a)
    D["n_auth_nonpol"] = [len(auth_by_day.get(d, [])) if FIRST <= d <= LAST else np.nan for d in D["date"]]
    for L_days in NEW_DAYS:
        n_new, share = [], []
        for d, o in zip(D["date"], D["win_open"]):
            lo = o - pd.Timedelta(days=L_days)
            auths = auth_by_day.get(d, [])
            if lo < c0 or not auths:
                n_new.append(np.nan), share.append(np.nan)
                continue
            lo64, o64 = np.datetime64(lo.to_pydatetime(), "s"), np.datetime64(o.to_pydatetime(), "s")
            k = 0
            for a in auths:
                ts = hist[a]
                j = np.searchsorted(ts, lo64, "left")
                if not (j < len(ts) and ts[j] < o64):
                    k += 1
            n_new.append(k), share.append(k / len(auths))
        D[f"n_new{L_days}"], D[f"new{L_days}_share"] = n_new, share
    salt = pd.read_csv(PRE / "loader_salt_continuity.csv", encoding="utf-8-sig")
    D["salt_flag_drop"] = D["date"].dt.strftime("%Y-%m-%d").map(dict(zip(salt["date"], salt["flag_drop"]))) \
        .fillna(False).astype(bool)

    # 미리보기 예측 라벨(참고)
    w = pd.read_parquet(PRE / "windows_kr.parquet")
    w = w[w["variant"] == "main"][["win_id", "open_kst", "close_kst", "length_h"]].copy()
    for c in ("open_kst", "close_kst"):
        w[c] = w[c].astype("datetime64[ns]")
    pw = pd.read_parquet(PRE / "pred_windows.parquet").merge(w[["win_id"]], on="win_id")
    pw = pw.merge(w, on="win_id", suffixes=("", "_w"))
    ref = {c: [] for c in ("ref_pred_win_id", "ref_pred_win_hours", "ref_pred_n_pairs", "ref_pred_n_pred", "ref_P_bar",
                           "ref_P_share", "ref_P_strong1")}
    for d, o in zip(D["date"], D["win_open"]):
        m = pw[(pw["open_kst_w"] <= o) & (o < pw["close_kst_w"])] if d >= pd.Timestamp("2026-07-01") else pw.iloc[0:0]
        if len(m) == 1:
            r = m.iloc[0]
            vals = [r["win_id"], r["length_h_w"], r["n_pairs"], r["n_pred"], r["P_bar"], r["P_share"], r["P_strong1"]]
        else:
            vals = [None] + [np.nan] * 6
        for c, v in zip(ref, vals):
            ref[c].append(v)
    for c, v in ref.items():
        D[c] = v

    # KOSPI
    p = pd.read_parquet(PRE / "prices.parquet")
    k = p[p["ticker"] == "^KS11"].set_index("date")["close"].sort_index()
    k.index = k.index.astype("datetime64[ns]")
    k = k[k.index >= pd.Timestamp("2026-01-02")]
    ret = k.pct_change()
    cmax = k.cummax()
    peak_date = k.groupby(cmax.ne(cmax.shift()).cumsum()).transform(lambda s: s.index[0])
    D["krx_session"] = D["date"].isin(k.index)
    D["kospi_close"] = D["date"].map(k)
    D["kospi_ret"] = D["date"].map(ret)
    last = k.reindex(pd.date_range(k.index[0], LAST, freq="D")).ffill()
    D["kospi_last_close"] = D["date"].map(last)
    cm_d = cmax.reindex(last.index).ffill()
    D["kospi_dd_ytd"] = D["kospi_last_close"] / D["date"].map(cm_d) - 1
    D["kospi_ytd_peak_date"] = D["date"].map(peak_date.reindex(last.index).ffill())
    ref_peak = pd.Series(pd.NaT, index=D.index, dtype="datetime64[ns]")
    for a, b in REF_PEAKS:
        ref_peak[(D["date"] >= a) & (D["date"] <= b)] = a
    D["kospi_ref_peak_date"] = ref_peak
    D["kospi_dd_ref"] = D["kospi_last_close"] / D["kospi_ref_peak_date"].map(k) - 1
    fut_min = k[::-1].cummin()[::-1].shift(-1)          # 다음 거래일부터 10/16까지 최저 종가
    D["rule_drop5_later_lower"] = D["date"].map((ret <= -0.05) & (fut_min < k)).fillna(False).astype(bool)

    ev = D["date"].dt.strftime("%Y-%m-%d").map(EVENTS)
    D["event"] = ev.map(lambda t: t[0] if isinstance(t, tuple) else "")
    D["event_ko"] = ev.map(lambda t: t[1] if isinstance(t, tuple) else "")

    D = D[(D["date"] >= FIRST) & (D["date"] <= LAST)].reset_index(drop=True)
    order = ["date", "weekday", "win_open", "win_close", "complete", "hours_observed", "collection", "event", "event_ko",
             "d_no", "n_saved", "missing_rate", "n_posts", "n_comments_preclose", "comments_per_post",
             "n_comments_in_win", "trunc_share", "d_no_base20_mean", "d_no_z20", "d_no_lnz20", "n_auth_valid"] + \
        [f"{b}_A" for b in BUNDLES] + [f"{b}_auth_n" for b in BUNDLES] + \
        ["n_auth_nonpol", "n_new30", "new30_share", "n_new21", "new21_share", "salt_flag_drop"] + list(ref) + \
        ["krx_session", "kospi_close", "kospi_ret", "kospi_last_close", "kospi_dd_ytd", "kospi_ytd_peak_date",
         "kospi_ref_peak_date", "kospi_dd_ref", "rule_drop5_later_lower"]
    D = D[order]
    D.to_parquet(OUT / "daily.parquet", index=False)

    # ---------------- 검증
    chk: dict = {}
    for d, v in EXPECT_CLOSE.items():
        chk[f"close_{d}"] = {"file": round(float(k[pd.Timestamp(d)]), 2), "expected": v,
                             "ok": round(float(k[pd.Timestamp(d)])) == v}
    for d in ("2026-07-30", "2026-10-15"):
        r = D[D["date"] == pd.Timestamp(d)].iloc[0]
        chk[f"dd_ref_{d}"] = round(float(r["kospi_dd_ref"]), 4)
    chk["drop_days"] = {d: {"ret": round(float(ret[pd.Timestamp(d)]), 4),
                            "later_lower": bool(D.loc[D["date"] == pd.Timestamp(d), "rule_drop5_later_lower"].iloc[0])}
                        for d, t in EVENTS.items() if t[0] == "drop_not_bottom"}
    rule_days = D.loc[D["rule_drop5_later_lower"], "date"].dt.strftime("%Y-%m-%d").tolist()
    chk["rule_drop5_later_lower_all_days"] = rule_days
    chk["rule_days_not_in_fixed_list"] = [d for d in rule_days if d not in EVENTS]
    # 평일 24시간 W_KR 창과 대조(같은 시각 범위 → 같은 값이어야 한다)
    wk = pd.read_parquet(PRE / "windows_kr.parquet")
    wk = wk[(wk["variant"] == "main") & wk["complete"] & (wk["length_h"] == 24)].copy()
    wk["open_kst"] = wk["open_kst"].astype("datetime64[ns]")
    fa = pd.read_parquet(PRE / "feat_activity.parquet")
    fa = fa[fa["market"] == "kr"].set_index("win_id")
    ft = pd.read_parquet(PRE / "feat_text.parquet")
    ft = ft[ft["market"] == "kr"].set_index("win_id")
    m = D.set_index("win_open").join(wk.set_index("open_kst")[["win_id", "d_no", "n_saved_in_range"]], rsuffix="_w",
                                     how="inner")
    m = m.join(fa[["NW30"]], on="win_id").join(ft[[f"{b}_A_all" for b in BUNDLES if b != "KX"]], on="win_id")
    cmp = {"windows": int(len(m)),
           "d_no_max_abs_diff": float((m["d_no"] - m["d_no_w"]).abs().max()),
           "n_saved_max_abs_diff": float((m["n_saved"] - m["n_saved_in_range"]).abs().max())}
    for b in BUNDLES:
        if b == "KX":
            continue
        dd = (m[f"{b}_A"] - m[f"{b}_A_all"]).abs()
        cmp[f"{b}_A_max_abs_diff"] = float(dd.max())
    nw = (m["new30_share"] - m["NW30"]).abs()
    cmp["new30_vs_NW30"] = {"n_both": int(nw.notna().sum()), "max_abs_diff": float(nw.max())}
    chk["vs_frozen_24h_windows"] = cmp
    chk["ref_pred_days"] = int(D["ref_pred_win_id"].notna().sum())
    chk["ref_pred_days_exact_24h"] = int((D["ref_pred_win_hours"] == 24).sum())
    chk["z20_first_valid"] = str(D.loc[D["d_no_z20"].notna(), "date"].min().date())
    chk["new30_first_valid"] = str(D.loc[D["new30_share"].notna(), "date"].min().date())
    chk["new21_first_valid"] = str(D.loc[D["new21_share"].notna(), "date"].min().date())
    chk["incomplete_days"] = D.loc[~D["complete"], "date"].dt.strftime("%Y-%m-%d").tolist()
    chk["unit_hits_rows"] = int(len(H))
    chk["unit_hits_excl"] = H["excl"].value_counts().to_dict()
    meta = {
        "made": datetime.now().replace(microsecond=0).isoformat(),
        "purpose": "바닥 사례 비교(설명용). 동결 가설 판정 아님, 매수·매도 근거 아님",
        "window": "하루 창 = KST 그날 08:00 ~ 다음 날 08:00 직전. 동결 W_KR 대신 달력 하루 창을 썼다",
        "rows": int(len(D)), "first": str(FIRST.date()), "last": str(LAST.date()), "columns": list(D.columns),
        "events": EVENTS, "ref_peaks": [[str(a.date()), str(b.date())] for a, b in REF_PEAKS],
        "z20": "직전 20개 하루 창(달력일, 자기 제외)이 모두 완전할 때만", "new_author": "feat_activity.newcomer_features 규칙을 하루 창에 적용",
        "lexicon_bundles": BUNDLES, "lexicon_status": {b: json.loads((ANT / "interim" / "results" / "lexicon_v02_full.json")
                                                                     .read_text("utf-8"))["status"][b] for b in BUNDLES},
        "t_end": str(t_end), "data_start_c0": str(c0), "rt_start": str(RT_START),
        "checks": chk,
        "inputs_sha256": {n: sha(PRE / n) for n in ("units.parquet", "windows_kr.parquet", "feat_activity.parquet",
                                                     "feat_text.parquet", "pred_windows.parquet", "prices.parquet")},
        "lexicon_sha256": {n: sha(ANT / "interim" / "results" / n) for n in ("lexicon_v02.json", "lexicon_v02_full.json")},
        "outputs_sha256": {n: sha(OUT / n) for n in ("daily.parquet", "unit_hits.parquet")},
        "code_sha256": sha(Path(__file__)),
        "runtime_sec": {"total": round((datetime.now() - t0).total_seconds(), 1), "matcher": round(match_sec, 1)},
    }
    (OUT / "daily_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps({k_: meta[k_] for k_ in ("rows", "checks", "runtime_sec")}, ensure_ascii=False, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
