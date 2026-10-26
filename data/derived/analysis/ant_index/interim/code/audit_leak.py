"""시점 누출 감사: 로더(loader.py)와 특징 3단계(feat_activity·feat_text·feat_topic)를 원 줄부터 끝까지 다시 돌린다.

시험
  A 미래 교란(내용): 마감 T 뒤에 쓴 글·댓글의 제목·본문·작성자를 바꾸고(감정어를 잔뜩 넣은 글과 무작위 글),
    모든 줄의 조회·추천·다른의견(글·댓글)과 다시 받은 줄의 ncomment·제목·본문을 바꾸고,
    T 뒤 새 글과 '다시 받은 줄'(T 뒤 댓글만 든 새 줄)을 더한다. → 마감 ≤ T 창의 모든 특징이 그대로여야 한다.
  B 미래 교란(지우기): A + T 뒤 댓글 20%·T 뒤 글 20%를 모든 줄에서 지운다. 댓글 날짜 추론이 다른 줄에 기대므로
    바뀔 수 있는 것은 '마감 ≤ T 창에서 단위가 빠지는 쪽'뿐이어야 한다(늦게만 틀림).
  D 댓글 날짜: (1) 맞춤을 끈 날짜가 맞춘 날짜(정답)보다 이르지 않은가, (2) 다시 받은 줄을 모두 지운(첫 줄만) 정보 상태의
    날짜가 전체 정보 날짜보다 이르지 않은가, (3) 파일 줄 순서 = 수집 순서(맞춘 날짜가 줄 순서대로 줄지 않는가),
    (4) '수집 당일 = 시각만' 규칙(24시간 규칙이 아님)을 다시 센다.
  N 기준선: rz8·z60·pct250·확장 회귀·직전 n창 평균 함수마다 '자기 값·미래 값을 바꿔도 앞 출력이 그대로', '자기 값을 바꾸면
    직전 자료만으로 만든 기준과 같은 출력'을 무작위 자료로 본다.
  F 수치 필드: 코드에서 views·up·down 사용처를 찾고, units에 그 열이 없는지 본다(A가 값 교란도 함께 본다).

읽기만: posts 스냅샷(loader_manifest.json의 바이트만), state.json(소금 지문·high만, 값 출력 없음), 사전 v0.2.
가격은 읽지 않는다(feat_text의 H24 사건 표지는 prices=None으로 끈다). 쓰기: interim/results/audit_leak_*.json
실행: py audit_leak.py [--quick]   (이 폴더에서)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import tempfile
import time as _time
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import loader as LD  # noqa: E402
import feat_activity as FA  # noqa: E402
import feat_text as FX  # noqa: E402
import feat_topic as FT  # noqa: E402
from lexicon_v02_final import load_v02  # noqa: E402

RESULTS = CODE.parent / "results"
SEED = 20261016
LOADED = ["떡상 가즈아 풀매수 몰빵 익절 수익인증 갓닉스", "한강 폭락 떡락 패닉셀 투매 반토막 물렸 깡통",
          "SOXL 속슬 TQQQ 신용 미수 빚투 영끌 반대매매 마진콜", "삼전 하닉 엔비디아 마이크론 미장 국장 코스피 나스닥",
          "다 팔았 손절했 손절할까 팔까요 존버 공매도 세력 개미털기 설거지", "풀매수 했습니다 샀습니다 탑승했 주린이 부럽 못 탔"]
NUMERIC_POST = ("views", "up", "down")


# ================================================================ 자료
def load_base() -> dict:
    man = json.loads((RESULTS / "loader_manifest.json").read_text("utf-8"))
    rows, man2 = LD.read_snapshot(LD.POSTS_DIR, LD.CUTOFF, man)
    salt = LD.salt_check()           # 값은 출력하지 않는다(지문만 비교)
    if salt.get("fingerprint") != man.get("salt_fingerprint"):
        raise RuntimeError("소금 지문이 스냅샷과 다르다")
    cal = LD.trading_calendars(LD.CAL_START, date.fromisoformat(LD.CUTOFF), RESULTS / "loader_calendar.parquet")
    return {"rows": rows, "man": man2, "cal": cal, "high": salt.get("high")}


def run_loader(rows, base, high=None) -> dict:
    return LD.build(rows, base["man"], base["cal"], {"high": base["high"] if high is None else high})


def _write_inputs(b: dict, tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    b["units"].to_parquet(tmp / "units.parquet", index=False)
    b["windows_kr"].to_parquet(tmp / "windows_kr.parquet", index=False)
    b["windows_us"].to_parquet(tmp / "windows_us.parquet", index=False)
    shutil.copy(RESULTS / "loader_calendar.parquet", tmp / "loader_calendar.parquet")
    (tmp / "loader_summary.json").write_text(json.dumps({"snapshot": {"t_end": str(b["t_end"])}}), "utf-8")


def legacy_trunc(b: dict) -> dict:
    """고치기 전 동작 재현: 잘림 표지를 마지막 줄(trunc_last)로 쓴다(units의 trunc_first 자리에 trunc_last를 넣고
    창 잘림 비율도 trunc_last로 다시 센다). 고친 코드와 같은 하네스로 고치기 전후를 비교하려고 둔다."""
    b = dict(b)
    u = b["units"].copy()
    u["trunc_first"] = u["trunc_last"]
    b["units"] = u
    posts = u[u["utype"] == "post"]
    for wn, market in (("windows_kr", "kr"), ("windows_us", "us")):
        w = b[wn].copy()
        share = pd.Series(np.nan, index=w.index)
        for (mk, v), col in LD.ID_COL.items():
            if mk != market:
                continue
            m = w["variant"] == v
            g = posts[posts[col].notna()].groupby(col)["trunc_last"].mean()
            share[m] = w.loc[m, "win_id"].map(g).astype(float)
        w["trunc_share"] = share
        b[wn] = w
    return b


def run_features(b: dict, tmp: Path) -> dict[str, pd.DataFrame]:
    """세 특징 모듈을 결과 폴더 대신 tmp에서 돌린다(파켓 왕복으로 실제 실행과 같은 자료형)."""
    _write_inputs(b, tmp)
    out = {}
    # feat_activity
    fb = FA.build(results=tmp)
    fk, fu = fb["fk"], fb["fu"]
    fa = pd.concat([fk, fu], ignore_index=True, sort=False)
    out["fa"] = fa.drop(columns=["outage", "spam_wave"]).rename(columns={"outage_b": "outage", "spam_b": "spam_wave"})
    out["fa_wkend"] = fb["wkend"]
    out["fa_h40"] = fb["h40"]
    # feat_text(가격 없음: H24 사건 표지만 빠진다)
    cols = ["no", "cid", "a", "utype", "dt_kst", "title", "text", "dt_conf", "win_kr", "win_u", "pwin_kr", "pwin_u",
            "pre_close_kr", "pre_close_u", "rank_ok_kr", "rank_ok_u", "c120_kr", "c120_u"]
    u = pd.read_parquet(tmp / "units.parquet", columns=cols)
    m = FX.match_units(u)
    wk = pd.read_parquet(tmp / "windows_kr.parquet")
    wu = pd.read_parquet(tmp / "windows_us.parquet")
    out["ft"] = pd.concat([FX.build_market("kr", u, m, wk, None), FX.build_market("us", u, m, wu, None)],
                          ignore_index=True, sort=False)
    # feat_topic
    FT.F_T_END = pd.Timestamp(b["t_end"]).to_pydatetime()
    units = pd.read_parquet(tmp / "units.parquet")
    hits = FT.unit_hits(units, load_v02())
    F = FT.Frame(units, hits)
    calp = pd.read_parquet(tmp / "loader_calendar.parquet")
    cal = {"krx": [d.date() for d in calp.loc[calp["krx"], "date"]],
           "nyse": [d.date() for d in calp.loc[calp["nyse"], "date"]]}
    wk_all = wk[wk["variant"] == "main"].sort_values("close_kst").reset_index(drop=True)
    wu_all = wu[wu["variant"] == "main"].sort_values("close_kst").reset_index(drop=True)
    wkc, wuc = wk_all[wk_all["complete"]].reset_index(drop=True), wu_all[wu_all["complete"]].reset_index(drop=True)
    kr, kr_st = FT.build_kr_daily(F, wkc, cal)
    kr["table"] = "KR_D"
    us, us_st = FT.build_us_daily(F, wuc, cal)
    us["table"] = "US_D"
    kw, kw_st = FT.build_weekly(F, wk_all, "kr", cal)
    uw, uw_st = FT.build_weekly(F, wu_all, "us", cal)
    we = FT.build_weekend(F, cal, wkc)
    out["fto"] = pd.concat([kr, us, kw, uw, we], ignore_index=True, sort=False)
    out["fto_stock"] = pd.DataFrame(kr_st + us_st + kw_st + uw_st)
    return out


# ================================================================ 비교
KEYS = {"fa": ["win_id"], "fa_wkend": ["sat"], "fa_h40": ["market", "stock", "week_close_day"], "ft": ["win_id"],
        "fto": ["table", "win_id"], "fto_stock": ["table", "win_id", "stock"]}
# 행 마감보다 늦게 닫는 열(문서가 정한 강건판): H49 개장 1분 전 판은 미국 시가 직전(09:30 ET)에 닫는다
LATE_COLS = {"fto": {"h49_s_e_r0929", "h49_n_units_r0929"}}


def _close_col(name: str, df: pd.DataFrame) -> pd.Series:
    if name == "fto_stock":
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df["close_kst"])


def _num(s: pd.Series) -> np.ndarray | None:
    if s.dtype == bool or str(s.dtype) in ("boolean", "Int64", "Float64") or s.dtype.kind in "fiu":
        return pd.to_numeric(s.astype("Float64"), errors="coerce").to_numpy(float, na_value=np.nan)
    return None


def compare(base: dict, pert: dict, T: pd.Timestamp, row_close: dict | None = None) -> dict:
    """마감 ≤ T 행끼리 모든 공통 열을 비교한다. 값이 다른 열과 행 수를 돌려준다."""
    rep = {}
    for name, keys in KEYS.items():
        a, b = base[name].copy(), pert[name].copy()
        if name == "fto_stock":   # 종목 긴 표: 짝 창의 마감으로 고른다
            cl = pd.concat([base["fto"][["table", "win_id", "close_kst"]]]).drop_duplicates(["table", "win_id"])
            a = a.merge(cl, on=["table", "win_id"], how="left")
            b = b.merge(cl, on=["table", "win_id"], how="left")
        for df in (a, b):
            for k in keys:
                if k in ("week_close_day", "sat"):
                    df[k] = pd.to_datetime(df[k])
        a = a[pd.to_datetime(a["close_kst"]) <= T]
        b = b[pd.to_datetime(b["close_kst"]) <= T]
        a = a.drop_duplicates(keys).set_index(keys).sort_index()
        b = b.drop_duplicates(keys).set_index(keys).sort_index()
        missing = sorted(map(str, set(a.index) ^ set(b.index)))
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
        diff = {}
        for c in a.columns.intersection(b.columns):
            if c in LATE_COLS.get(name, set()):
                continue
            x, y = a[c], b[c]
            xn, yn = _num(x), _num(y)
            if xn is not None and yn is not None:
                bad = ~((np.isnan(xn) & np.isnan(yn)) | (np.abs(xn - yn) <= 1e-9 * np.maximum(1, np.abs(xn))))
            else:   # 결측끼리는 같다(문자열 변환이 결측을 남기는 pandas 판이 있어 따로 본다)
                xm, ym = pd.isna(x).to_numpy(), pd.isna(y).to_numpy()
                xs = np.array([str(v) for v in x.to_numpy(object)], dtype=object)
                ys = np.array([str(v) for v in y.to_numpy(object)], dtype=object)
                bad = ~((xm & ym) | (~xm & ~ym & (xs == ys)))
            if bad.any():
                diff[c] = {"rows": int(bad.sum()), "first": str(idx[np.flatnonzero(bad)[0]])}
        rep[name] = {"rows_compared": int(len(idx)), "key_mismatch": missing[:5], "n_key_mismatch": len(missing),
                     "cols_compared": int(len(a.columns.intersection(b.columns))), "diff": diff}
    return rep


# ================================================================ 교란
def _h(*parts) -> int:
    return int(hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _scramble_text(key: tuple) -> str:
    k = _h("t", *key)
    if k % 2:
        return LOADED[k % len(LOADED)] + f" 이번 {k % 97}번째"
    return "qxz" + "".join("qxzjwv"[(k >> (3 * i)) % 6] for i in range(10))


def perturb(rows: list[dict], units0: pd.DataFrame, T: pd.Timestamp, S: datetime, seed: int,
            delete: bool = False, refetch_frac: float = 0.3, n_new_posts: int = 300,
            keep_roots: bool = False) -> tuple[list[dict], dict]:
    """원 줄 교란. T 뒤 단위(로더 기준 시각 ≥ T)만 내용을 바꾸고, 다시 받은 줄의 글 필드·모든 수치 필드를 바꾼다."""
    rng = np.random.default_rng(seed)
    authors = units0["a"].dropna().unique()
    c0 = units0[units0["utype"] == "comment"]
    fut_full = set(zip(c0.loc[c0["dt_kst"] >= T, "no"], c0.loc[c0["dt_kst"] >= T, "a"],
                       c0.loc[c0["dt_kst"] >= T, "text"], c0.loc[c0["dt_kst"] >= T, "dt_kst"].dt.strftime("%Y-%m-%d %H:%M:%S")))
    to = c0[c0["dt_conf"] != "full"]
    tod_fut: dict[tuple, set] = defaultdict(set)
    for no, a, tx, dt in zip(to["no"], to["a"], to["text"], to["dt_kst"]):
        tod_fut[(no, a, tx, dt.strftime("%H:%M:%S"))].add(bool(dt >= T))
    stats = defaultdict(int)
    first_line: dict[int, tuple] = {}
    for r in sorted(rows, key=lambda r: (r["file"], r["line"])):
        first_line.setdefault(r["rec"]["no"], (r["file"], r["line"]))

    def rand_author(key) -> str:
        return str(authors[_h("a", *key) % len(authors)])

    def is_del(key) -> bool:
        return delete and _h("d", *key) % 5 == 0

    out = []
    for r in rows:
        rec = copy.deepcopy(r["rec"])
        no = rec["no"]
        pdt = datetime.strptime(rec["dt"], "%Y-%m-%d %H:%M")
        future_post = pdt >= T
        refetched = first_line[no] != (r["file"], r["line"])
        if future_post and is_del(("post", no)):
            stats["posts_deleted_rows"] += 1
            continue
        for f in NUMERIC_POST:
            rec[f] = int(rng.integers(0, 10 ** 5))
        if future_post or refetched:
            rec["ncomment"] = int(rng.integers(0, 3000))
            stats["ncomment_changed_rows"] += 1
        if future_post:
            rec["title"], rec["body"] = _scramble_text(("pt", no)), _scramble_text(("pb", no))
            rec["a"] = rand_author(("pa", no))
            stats["future_post_rows"] += 1
        elif refetched:
            rec["title"], rec["body"] = _scramble_text(("rt", no, r["line"])), _scramble_text(("rb", no, r["line"]))
            stats["refetched_rows_edited"] += 1
        cs = []
        ndel = 0
        for c in rec["comments"]:
            c["up"], c["down"] = int(rng.integers(0, 500)), int(rng.integers(0, 500))
            raw = c["dt"]
            fut = None
            if LD.FULL_RX.match(raw):
                fut = (no, c["a"], c["text"], raw) in fut_full
            elif LD.TOD_RX.match(raw):
                s = tod_fut.get((no, c["a"], c["text"], raw), set())
                if s == {True}:
                    fut = True
                elif s == {False}:
                    fut = False
                else:
                    stats["tod_ambiguous_kept"] += 1
                    fut = False
            if fut:
                key = (no, c["a"], c["text"])
                if is_del(("c",) + key) and not (keep_roots and int(c.get("depth") or 0) == 0):
                    stats["comments_deleted_appearances"] += 1
                    ndel += 1
                    continue
                c["text"], c["a"] = _scramble_text(("c",) + key), rand_author(("c",) + key)
                stats["future_comment_appearances_scrambled"] += 1
            cs.append(c)
        rec["comments"] = cs
        if ndel and not (future_post or refetched):
            # 지운 미래 댓글은 '처음부터 없던 댓글'로 본다: 그 줄의 원천 댓글 수도 같이 줄인다(잘림 표지 정의 그대로)
            rec["ncomment"] = max(0, int(rec.get("ncomment") or 0) - ndel)
        out.append({"file": r["file"], "line": r["line"], "rec": rec})

    # 다시 받은 줄 더하기: T 이전 글 일부에 T 뒤 댓글만 든 새 줄(파일 끝 = 가장 늦은 수집)
    last_day = S.date() - timedelta(days=1)
    hi_dt = datetime.combine(last_day, time(23, 59, 59))
    posts_past = units0[(units0["utype"] == "post") & (units0["dt_kst"] < T)][["no", "dt_kst"]]
    pick = posts_past.sample(frac=refetch_frac, random_state=seed)
    line0 = 10 ** 7
    recs_by_no = {}
    for r in rows:
        recs_by_no[r["rec"]["no"]] = r
    for k, (no, pdt) in enumerate(zip(pick["no"], pick["dt_kst"])):
        src = recs_by_no[no]
        rec = copy.deepcopy(src["rec"])
        lo_t = max(T, pd.Timestamp(pdt))
        span = max((pd.Timestamp(hi_dt) - lo_t).total_seconds(), 1)
        cs = []
        for j in range(int(rng.integers(1, 4))):
            t = lo_t + pd.Timedelta(seconds=float(rng.uniform(0, span)))
            cs.append({"dt": t.strftime("%Y-%m-%d %H:%M:%S"), "text": _scramble_text(("rf", no, j)),
                       "up": 0, "down": 0, "depth": 0, "a": rand_author(("rf", no, j))})
        rec.update({"title": _scramble_text(("rft", no)), "body": _scramble_text(("rfb", no)), "comments": cs,
                    "ncomment": int(rng.integers(0, 3000)), "views": 1, "up": 1, "down": 1})
        out.append({"file": src["file"], "line": line0 + k, "rec": rec})
    stats["refetch_rows_added"] = len(pick)
    # T 뒤 새 글(번호는 끝 번호 뒤)
    max_no = max(r["rec"]["no"] for r in rows)
    span = max((pd.Timestamp(hi_dt) - T).total_seconds(), 60)
    for k in range(n_new_posts):
        t = T + pd.Timedelta(seconds=float(rng.uniform(0, span)))
        t = t.floor("min")
        no = max_no + 1 + k
        cs = []
        for j in range(int(rng.integers(0, 6))):
            ct = t + pd.Timedelta(seconds=float(rng.uniform(0, max((pd.Timestamp(hi_dt) - t).total_seconds(), 1))))
            cs.append({"dt": ct.strftime("%Y-%m-%d %H:%M:%S"), "text": _scramble_text(("nc", no, j)), "up": 0,
                       "down": 0, "depth": 0, "a": rand_author(("nc", no, j))})
        rec = {"no": no, "dt": t.strftime("%Y-%m-%d %H:%M"), "title": _scramble_text(("nt", no)),
               "body": _scramble_text(("nb", no)), "views": 5, "up": 5, "down": 5, "ncomment": len(cs) + 3,
               "comments": cs, "a": rand_author(("np", no))}
        out.append({"file": t.strftime("%Y-%m-%d"), "line": line0 * 2 + k, "rec": rec})
    stats["new_posts_added"] = n_new_posts
    return out, dict(stats), max_no + n_new_posts


def unit_shift_check(u0: pd.DataFrame, u1: pd.DataFrame, T: pd.Timestamp) -> dict:
    """마감 ≤ T 창(국내·미국 본판)의 단위 집합: 교란 뒤 새로 들어온 단위가 없는가(빠지는 쪽만)."""
    out = {}
    for col in ("win_kr", "win_u"):
        def keyset(u):
            x = u[u[col].notna() & (u["dt_kst"] < T)]
            return set(zip(x["utype"], x["no"], x["a"], x["text"].fillna(""), x["dt_kst"].astype("datetime64[s]")))
        k0, k1 = keyset(u0), keyset(u1)
        out[col] = {"base": len(k0), "pert": len(k1), "added": len(k1 - k0), "removed": len(k0 - k1)}
        # 창이 바뀐 단위: 같은 (종류, 글, 작성자, 본문, 시각)의 창 id 비교
        m0 = u0[u0[col].notna()].set_index(["utype", "no", "a", "text", "dt_kst"])[col]
        m1 = u1[u1[col].notna()].set_index(["utype", "no", "a", "text", "dt_kst"])[col]
        m0 = m0[~m0.index.duplicated()]
        m1 = m1[~m1.index.duplicated()]
        j = m0.index.intersection(m1.index)
        out[col]["window_changed_same_key"] = int((m0.loc[j] != m1.loc[j]).sum())
    r0, r1 = _root_keys(u0), _root_keys(u1)
    jj = r0.index.intersection(r1.index)
    jj = jj[jj.get_level_values(3) < T]
    a, b = r0.loc[jj], r1.loc[jj]
    # 뿌리 '자리'가 바뀌었는가: 뿌리 시각으로 본다(교란 A는 미래 뿌리의 작성자·본문만 바꾸므로 시각은 그대로다)
    ch = a["r_dt"].astype(str).to_numpy() != b["r_dt"].astype(str).to_numpy()
    lat = pd.to_datetime(a["r_dt"]).to_numpy() > np.array(jj.get_level_values(3), dtype="datetime64[ns]")
    out["pre_T_comment_root_changed"] = {"compared": int(len(jj)), "changed": int(ch.sum()),
                                         "changed_and_base_root_dated_after_reply": int((ch & lat).sum())}
    return out


def _root_keys(u: pd.DataFrame) -> pd.DataFrame:
    """댓글 → 로더가 붙인 줄기 뿌리 댓글의 (작성자, 본문, 시각). 키 = (글, 작성자, 본문, 시각)."""
    c = u[u["utype"] == "comment"]
    m = c.set_index(["no", "cid"])[["a", "text", "dt_kst"]]
    rk = m.reindex(pd.MultiIndex.from_arrays([c["no"], c["root_cid"].astype("int64")])).reset_index(drop=True)
    out = c[["no", "a", "text", "dt_kst"]].reset_index(drop=True).copy()
    out["r_a"], out["r_t"], out["r_dt"] = rk["a"].to_numpy(), rk["text"].to_numpy(), rk["dt_kst"].to_numpy()
    out = out.set_index(["no", "a", "text", "dt_kst"])
    return out[~out.index.duplicated()]


def residual_checks(units: pd.DataFrame, fa: pd.DataFrame) -> dict:
    """문서가 정한 예외·생존 편향의 크기: (1) 잘림 표지(ncomment, 수집 때까지 댓글 수)와 마감 전 정보만 쓴 _pc 판,
    (2) 목록상 뿌리가 대댓글보다 늦은 시각인 대댓글(뿌리를 고친 시각·고정 댓글)."""
    out = {}
    p = units[units["utype"] == "post"]
    c = units[units["utype"] == "comment"]
    for mk, pidc, prec in (("kr", "pwin_kr", "pre_close_kr"), ("us", "pwin_u", "pre_close_u")):
        pw = p[p[pidc].notna()]
        cm = c[c[prec].astype(bool) & c[pidc].notna()]
        nroot = cm[cm["depth"].fillna(0) == 0].groupby("no").size()
        tf = pw["trunc_first"].astype(bool).to_numpy()
        tl = pw["trunc_last"].astype(bool).to_numpy()
        pc = pw["no"].map(nroot).fillna(0).to_numpy() >= FA.PAGE_ROOTS
        rt = (pw["dt_kst"] >= pd.Timestamp("2026-10-13")).to_numpy()
        out[f"trunc_{mk}"] = {"posts": int(len(pw)), "trunc_first": int(tf.sum()), "trunc_last": int(tl.sum()),
                              "first_ne_last": int((tf != tl).sum()), "trunc_pc": int(pc.sum()),
                              "trunc_first_not_pc": int((tf & ~pc).sum()), "pc_not_trunc_first": int((pc & ~tf).sum()),
                              "trunc_first_rate_retro": float(tf[~rt].mean()), "trunc_first_rate_realtime": float(tf[rt].mean()),
                              "trunc_pc_rate_retro": float(pc[~rt].mean()), "trunc_pc_rate_realtime": float(pc[rt].mean())}
        f = fa[fa["market"] == mk]
        for x, y in (("fight", "fight_pc"), ("trunc_share", "trunc_share_pc"), ("fight_adj", "fight_adj_pc")):
            m = f[x].notna() & f[y].notna()
            out[f"{mk}:{x}_vs_{y}"] = {"n": int(m.sum()),
                                       "pearson": float(np.corrcoef(f.loc[m, x], f.loc[m, y])[0, 1]) if m.sum() > 2 else None,
                                       "mean_abs_diff": float((f.loc[m, x] - f.loc[m, y]).abs().mean()),
                                       "max_abs_diff": float((f.loc[m, x] - f.loc[m, y]).abs().max())}
    rk = _root_keys(units)
    rep = c[c["depth"].fillna(0) > 0].set_index(["no", "a", "text", "dt_kst"])
    rep = rep[~rep.index.duplicated()]
    rr = rk.loc[rep.index]
    lag_h = (pd.to_datetime(rr["r_dt"]).to_numpy() - np.array(rep.index.get_level_values(3), dtype="datetime64[ns]"))         / np.timedelta64(1, "h")
    late = lag_h > 0
    out["reply_root_dated_after_reply"] = {"replies": int(len(rep)), "n": int(late.sum()),
                                           "lag_le_1h": int((late & (lag_h <= 1)).sum()),
                                           "lag_gt_24h": int((late & (lag_h > 24)).sum()),
                                           "in_kr_pre_close_fight_pool": int((late & rep["pre_close_kr"].astype(bool).to_numpy()
                                                                              & rep["pwin_kr"].notna().to_numpy()).sum())}
    return out


# ================================================================ 댓글 날짜(D)
def row_bounds(rows: list[dict], S: datetime, use_match: bool = True) -> tuple[dict, dict, dict]:
    """loader.resolve의 줄별 수집 날짜 범위 [lo, hi](맞춤 on/off)를 같은 규칙으로 다시 계산한다(검증용 복제)."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["file"], rows[i]["line"]))
    by_no = defaultdict(list)
    for i in order:
        by_no[rows[i]["rec"]["no"]].append(i)
    lo, hi, exact = {}, {}, {}
    fidx_by_no = {}
    for no, idxs in by_no.items():
        fidx = defaultdict(set)
        if use_match:
            for i in idxs:
                for c in rows[i]["rec"]["comments"]:
                    if LD.FULL_RX.match(c["dt"]):
                        fidx[(c["a"], c["text"], c["dt"][11:])].add(c["dt"][:10])
        fidx_by_no[no] = fidx
        for i in idxs:
            rec = rows[i]["rec"]
            P = LD._pdt(rec["dt"])
            tods = [LD._tod(c["dt"]) for c in rec["comments"] if LD.TOD_RX.match(c["dt"])]
            fulls = [LD._cdt(c["dt"]) for c in rec["comments"] if LD.FULL_RX.match(c["dt"])]
            l = P.date() + (timedelta(days=1) if tods and min(tods) < P.time() else timedelta(0))
            if fulls:
                l = max(l, max(fulls).date() + timedelta(days=1))
            h = S.date() - (timedelta(days=1) if tods and max(tods) > S.time() else timedelta(0))
            md = {fd for c in rec["comments"] if LD.TOD_RX.match(c["dt"]) for fd in fidx.get((c["a"], c["text"], c["dt"]), ())}
            if md:
                d = date.fromisoformat(max(md))
                exact[i] = d
                l = h = d
            lo[i], hi[i] = l, h
    by_file = defaultdict(list)
    for i in order:
        by_file[rows[i]["file"]].append(i)
    for idxs in by_file.values():
        run = None
        for i in idxs:
            if run is not None and i not in exact:
                lo[i] = max(lo[i], run)
            run = lo[i] if run is None else max(run, lo[i])
        run = None
        for i in reversed(idxs):
            if run is not None:
                hi[i] = min(hi[i], run)
            run = hi[i] if run is None else min(run, hi[i])
    return lo, hi, exact


def tod_dates(rows, lo, hi, use_match=True) -> dict[tuple, date]:
    """시각만 댓글 (글, 해시, 본문, 시각) → 놓을 날짜(loader와 같은 규칙: 맞춤 날짜, 아니면 나온 줄 hi의 최솟값)."""
    by_no = defaultdict(list)
    for i in sorted(range(len(rows)), key=lambda i: (rows[i]["file"], rows[i]["line"])):
        by_no[rows[i]["rec"]["no"]].append(i)
    out = {}
    for no, idxs in by_no.items():
        fidx = defaultdict(set)
        if use_match:
            for i in idxs:
                for c in rows[i]["rec"]["comments"]:
                    if LD.FULL_RX.match(c["dt"]):
                        fidx[(c["a"], c["text"], c["dt"][11:])].add(c["dt"][:10])
        groups = defaultdict(list)
        for i in idxs:
            for c in rows[i]["rec"]["comments"]:
                if LD.TOD_RX.match(c["dt"]):
                    groups[(c["a"], c["text"], c["dt"])].append(i)
        for k, ii in groups.items():
            if k in fidx:
                ds = fidx[k]
                out[(no,) + k] = date.fromisoformat(max(ds))
            else:
                out[(no,) + k] = min(hi[i] for i in ii)
    return out


def date_tests(base: dict, units0: pd.DataFrame) -> dict:
    rows, S = base["rows"], datetime.fromisoformat(base["man"]["S"])
    res = {}
    # 복제가 로더와 같은지: 시각만 댓글의 놓인 날짜
    lo, hi, exact = row_bounds(rows, S, True)
    d_on = tod_dates(rows, lo, hi, True)
    c0 = units0[(units0["utype"] == "comment") & (units0["dt_conf"] != "full")]
    loader_d = {(no, a, tx, dt.strftime("%H:%M:%S")): dt.date() for no, a, tx, dt in
                zip(c0["no"], c0["a"], c0["text"], c0["dt_kst"])}
    agree = sum(1 for k, v in d_on.items() if loader_d.get(k) == v)
    res["replica_agrees_with_loader"] = {"time_only_keys": len(d_on), "agree": agree,
                                         "loader_keys": len(loader_d)}
    # (1) 맞춤을 끄면: 맞춘 댓글의 정답 날짜보다 이르게 놓이는가
    lo0, hi0, _ = row_bounds(rows, S, False)
    d_off = tod_dates(rows, lo0, hi0, False)
    matched = c0[c0["dt_conf"] == "matched"]
    early = later = same = 0
    lag_days = []
    for no, a, tx, dt in zip(matched["no"], matched["a"], matched["text"], matched["dt_kst"]):
        k = (no, a, tx, dt.strftime("%H:%M:%S"))
        d = d_off.get(k)
        if d is None:
            continue
        if d < dt.date():
            early += 1
        elif d > dt.date():
            later += 1
            lag_days.append((d - dt.date()).days)
        else:
            same += 1
    res["no_match_vs_truth"] = {"matched_comments": int(len(matched)), "earlier_than_truth": early,
                                "later": later, "same": same,
                                "later_days_median": float(np.median(lag_days)) if lag_days else None,
                                "later_days_max": int(max(lag_days)) if lag_days else None}
    # 모든 시각만 키: 맞춤을 끈 날짜 ≥ 맞춘 날짜
    viol = sum(1 for k, v in d_on.items() if k in d_off and d_off[k] < v)
    res["no_match_vs_with_match_all_keys"] = {"keys": len(d_on), "earlier": viol}
    # (2) 다시 받은 줄을 모두 지운 정보 상태(첫 줄만)
    first = {}
    for r in sorted(rows, key=lambda r: (r["file"], r["line"])):
        first.setdefault(r["rec"]["no"], r)
    rows1 = list(first.values())
    lo1, hi1, _ = row_bounds(rows1, S, True)
    d1 = tod_dates(rows1, lo1, hi1, True)
    full_info = {}
    for no, a, tx, dt in zip(c0["no"], c0["a"], c0["text"], c0["dt_kst"]):
        full_info[(no, a, tx, dt.strftime("%H:%M:%S"))] = dt.date()
    e1 = sum(1 for k, v in d1.items() if k in full_info and v < full_info[k])
    l1 = sum(1 for k, v in d1.items() if k in full_info and v > full_info[k])
    res["first_rows_only_vs_full"] = {"keys": len(d1), "earlier": e1, "later": l1,
                                      "same": sum(1 for k, v in d1.items() if k in full_info and v == full_info[k])}
    # (3) 줄 순서 = 수집 순서: 같은 파일 안 맞춘 날짜가 줄 순서대로 줄어드는 쌍
    by_file = defaultdict(list)
    for i in sorted(range(len(rows)), key=lambda i: (rows[i]["file"], rows[i]["line"])):
        if i in exact:
            by_file[rows[i]["file"]].append(exact[i])
    inv = sum(sum(1 for x, y in zip(v, v[1:]) if y < x) for v in by_file.values())
    res["file_order_exact_inversions"] = {"rows_with_exact": len(exact), "adjacent_decreasing_pairs": inv}
    # (4) 수집 당일 규칙: 맞춘 줄(정해진 수집 날짜 X)에서 날짜 붙은 댓글이 X 이상인가, X−1인데 시각만 댓글보다 24시간 안인가
    same_or_after = within24 = rows_checked = 0
    for i, X in exact.items():
        rec = rows[i]["rec"]
        tods = [LD._tod(c["dt"]) for c in rec["comments"] if LD.TOD_RX.match(c["dt"])]
        fulls = [LD._cdt(c["dt"]) for c in rec["comments"] if LD.FULL_RX.match(c["dt"])]
        if not tods:
            continue
        rows_checked += 1
        same_or_after += sum(1 for f in fulls if f.date() >= X)
        coll_lo = datetime.combine(X, max(tods))      # 수집 시각 ≥ 가장 늦은 시각만 댓글
        within24 += int(any(f > coll_lo - timedelta(hours=24) for f in fulls))
    res["calendar_day_rule"] = {"rows_checked": rows_checked, "dated_comment_on_or_after_collect_day": same_or_after,
                                "rows_with_dated_comment_within_24h": within24}
    # 늦게 놓인(latest) 댓글이 든 창
    lt = units0[(units0["utype"] == "comment") & (units0["dt_conf"] == "latest")]
    res["latest_comments"] = {"n": int(len(lt)), "in_win_kr": int(lt["win_kr"].notna().sum()),
                              "in_win_u": int(lt["win_u"].notna().sum()),
                              "dates": {str(k): int(v) for k, v in lt["dt_kst"].dt.date.value_counts().sort_index().items()}}
    return res


# ================================================================ 기준선 함수(N)
def _ref_z(x, ok, n, nmin, positional):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if not (ok[i] and np.isfinite(x[i])):
            continue
        if positional:
            h = np.array([x[j] for j in range(max(0, i - n), i) if ok[j] and np.isfinite(x[j])])
        else:
            h = np.array([x[j] for j in range(i) if ok[j] and np.isfinite(x[j])])[-n:]
        if len(h) >= nmin and h.std(ddof=1) > 0:
            out[i] = (x[i] - h.mean()) / h.std(ddof=1)
    return out


def norm_tests(seed: int = SEED, n: int = 400, reps: int = 30) -> dict:
    """무작위 자료에서: 미래 값 교란 불변 + 직전 자료만 쓰는 참조 구현과 일치 + 자기 값 교란 시 기준선 불변."""
    rng = np.random.default_rng(seed)
    out = {}

    def check(name, f, ref=None):
        fut_bad = ref_bad = 0
        for _ in range(reps):
            x = rng.normal(size=n)
            x[rng.random(n) < 0.08] = np.nan
            wd = rng.integers(0, 5, n)
            ok = rng.random(n) > 0.1
            hol = rng.random(n) < 0.08
            y = f(x, wd, ok, hol)
            i = int(rng.integers(n // 3, n - 1))
            x2 = x.copy()
            x2[i:] = rng.normal(size=n - i) * 50 + 7
            y2 = f(x2, wd, ok, hol)
            fut_bad += int(not np.array_equal(y[:i], y2[:i], equal_nan=True))
            if ref is not None:
                r = ref(x, wd, ok, hol)
                ref_bad += int(not np.allclose(y, r, equal_nan=True, atol=1e-12))
        out[name] = {"reps": reps, "future_changed_past": fut_bad, "ref_mismatch": ref_bad if ref else None}

    # feat_text z_prev / pct_prev / rz8 / roll_mean_full
    check("feat_text.z_prev(60,40)", lambda x, wd, ok, hol: FX.z_prev(x, ok),
          lambda x, wd, ok, hol: _ref_z(x, ok, 60, 40, True))

    def ref_pct(x, wd, ok, hol, n=250, nmin=120):
        out = np.full(len(x), np.nan)
        for i in range(len(x)):
            if not (ok[i] and np.isfinite(x[i])):
                continue
            h = np.array([x[j] for j in range(max(0, i - n), i) if ok[j] and np.isfinite(x[j])])
            if len(h) >= nmin:
                out[i] = (np.sum(h < x[i]) + 0.5 * np.sum(h == x[i])) / len(h)
        return out
    check("feat_text.pct_prev(250,120)", lambda x, wd, ok, hol: FX.pct_prev(x, ok), ref_pct)

    def ref_rz8_text(x, wd, ok, hol):
        out = np.full(len(x), np.nan)
        for i in range(len(x)):
            if not (ok[i] and np.isfinite(x[i])):
                continue
            h = [x[j] for j in range(i) if wd[j] == wd[i] and ok[j] and not hol[j]][-8:]
            if len(h) == 8 and np.all(np.isfinite(h)):
                med = np.median(h)
                mad = np.median(np.abs(np.array(h) - med)) * 1.4826
                if mad > 0:
                    out[i] = np.clip((x[i] - med) / mad, -6, 6)
        return out
    check("feat_text.rz8", lambda x, wd, ok, hol: FX.rz8(x, ok, wd, hol), ref_rz8_text)

    def ref_rz8_act(x, wd, ok, hol, mn=8):
        out = np.full(len(x), np.nan)
        for i in range(len(x)):
            if not np.isfinite(x[i]):
                continue
            h = [x[j] for j in range(i) if wd[j] == wd[i] and ok[j] and np.isfinite(x[j])][-8:]
            if len(h) >= mn:
                med = np.median(h)
                mad = np.median(np.abs(np.array(h) - med))
                if mad > 0:
                    out[i] = np.clip((x[i] - med) / (1.4826 * mad), -6, 6)
        return out
    check("feat_activity.rz8(8)", lambda x, wd, ok, hol: FA.rz8(x, wd, ok, 8), ref_rz8_act)
    check("feat_activity.rz8(m4)", lambda x, wd, ok, hol: FA.rz8(x, wd, ok, 4),
          lambda x, wd, ok, hol: ref_rz8_act(x, wd, ok, hol, 4))
    def act_z_ref(x, wd, ok, hol):
        # z_trailing: 자기 값이 유한하면 계산(ok는 기준 편입 여부만), 기준 = 직전 ok·유한 값 60개
        out = np.full(len(x), np.nan)
        for i in range(len(x)):
            if not np.isfinite(x[i]):
                continue
            h = np.array([x[j] for j in range(i) if ok[j] and np.isfinite(x[j])])[-60:]
            if len(h) >= 40 and h.std(ddof=1) > 0:
                out[i] = (x[i] - h.mean()) / h.std(ddof=1)
        return out
    check("feat_activity.z_trailing(60,40)", lambda x, wd, ok, hol: FA.z_trailing(x, ok), act_z_ref)

    def expres_stable(x, wd, ok, hol):
        g = np.random.default_rng(3)
        xx = g.normal(size=len(x))
        return FA.expanding_resid(x, xx, ok, 20)[0]

    def expres_ref(x, wd, ok, hol):
        g = np.random.default_rng(3)
        xx = g.normal(size=len(x))
        out = np.full(len(x), np.nan)
        for i in range(len(x)):
            if not (np.isfinite(x[i]) and np.isfinite(xx[i])):
                continue
            m = np.array([j for j in range(i) if np.isfinite(x[j]) and np.isfinite(xx[j]) and ok[j]], int)
            if len(m) < 20:
                continue
            X = np.column_stack([np.ones(len(m)), xx[m]])
            beta = np.linalg.lstsq(X, x[m], rcond=None)[0]
            out[i] = x[i] - (beta[0] + beta[1] * xx[i])
        return out
    check("feat_activity.expanding_resid(20)", expres_stable, expres_ref)
    check("feat_topic.z_prior(60,40)", lambda x, wd, ok, hol: FT.z_prior(x, ok, 60, 40),
          lambda x, wd, ok, hol: _ref_z_valid(x, ok, 60, 40))
    check("feat_topic.z_prior(20,20)", lambda x, wd, ok, hol: FT.z_prior(x, ok, 20, 20),
          lambda x, wd, ok, hol: _ref_z_valid(x, ok, 20, 20))

    def swb_ref(x, wd, ok, hol):
        out = np.full(len(x), np.nan)
        for i in range(len(x)):
            h = [x[j] for j in range(i) if wd[j] == wd[i] and ok[j] and np.isfinite(x[j])][-8:]
            if len(h) >= 3:
                out[i] = np.median(h)
        return out
    check("feat_topic.same_weekday_base", lambda x, wd, ok, hol: FT.same_weekday_base(x, wd, ok), swb_ref)

    def tilt_ref(x, wd, ok, hol):
        okk = ok & np.isfinite(x)
        d = np.full(len(x), np.nan)
        for i in range(len(x)):
            v = [x[j] for j in range(i + 1) if okk[j]]
            k = len(v)
            if k - 20 >= 5:
                d[i] = np.mean(v[k - 5:k]) - np.mean(v[k - 25:k - 20])
        return d
    check("feat_topic.tilt_delta(dTilt)", lambda x, wd, ok, hol: FT.tilt_delta(x, ok)[1], tilt_ref)
    check("feat_topic.herd_events(120)", lambda x, wd, ok, hol: FT.herd_events(np.abs(x), ok))

    def lsm(x, wd, ok, hol):
        df = pd.DataFrame({"wd": wd, "kind": np.where(hol, "holiday", "weekday"), "comp": ok, "x": x})
        return LD._same_weekday_median(df.rename(columns={"wd": "weekday", "kind": "main_kind", "comp": "complete"}), "x")

    def lsm_ref(x, wd, ok, hol):
        out = np.full(len(x), np.nan)
        for i in range(len(x)):
            h = [x[j] for j in range(i) if wd[j] == wd[i] and ok[j] and not hol[j] and np.isfinite(x[j])][-8:]
            if len(h) >= 3:
                out[i] = np.median(h)
        return out
    check("loader._same_weekday_median", lsm, lsm_ref)
    return out


def _ref_z_valid(x, ok, n, nmin):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if not (ok[i] and np.isfinite(x[i])):
            continue
        h = np.array([x[j] for j in range(i) if ok[j] and np.isfinite(x[j])])[-n:]
        if len(h) >= nmin and h.std(ddof=1) > 0:
            out[i] = (x[i] - h.mean()) / h.std(ddof=1)
    return out


def realdata_norm_check(results: Path = RESULTS) -> dict:
    """실제 산출에서 정규화 열을 참조 구현(직전 자료만)으로 다시 만들어 맞춰 본다."""
    out = {}
    ft = pd.read_parquet(results / "feat_text.parquet")
    for mk in ("kr", "us"):
        f = ft[ft["market"] == mk].sort_values("close_kst").reset_index(drop=True)
        valid = f["valid"].astype(bool).to_numpy()
        for src, dst in (("S", "S_z60"), ("B", "B_z60"), ("D_SD", "D_SD_z60"), ("MC_A_all", "MC_A_all_z60")):
            ref = _ref_z(f[src].to_numpy(float), valid, 60, 40, True)
            got = f[dst].to_numpy(float)
            out[f"feat_text:{mk}:{dst}"] = {"n_valid": int(np.isfinite(got).sum()),
                                            "match": bool(np.allclose(ref, got, equal_nan=True))}
        p = FX.pct_prev(f["S"].to_numpy(float), valid)
        out[f"feat_text:{mk}:S_pct250"] = {"n_valid": int(np.isfinite(f["S_pct250"].to_numpy(float)).sum()),
                                           "match": bool(np.array_equal(p, f["S_pct250"].to_numpy(float), equal_nan=True))}
    fa = pd.read_parquet(results / "feat_activity.parquet")
    for mk in ("kr", "us"):
        f = fa[fa["market"] == mk].sort_values("close_kst").reset_index(drop=True)
        ok = f["base_ok"].astype(bool).to_numpy()
        wd = f["weekday"].to_numpy()
        for src, dst in (("A_raw", "A_rz8m4"), ("U_per_h", "U_rz8m4"), ("N_per_h", "N_rz8m4")):
            x = f[src].to_numpy(float)
            ref = np.full(len(x), np.nan)
            for i in range(len(x)):
                if not np.isfinite(x[i]):
                    continue
                h = [x[j] for j in range(i) if wd[j] == wd[i] and ok[j] and np.isfinite(x[j])][-8:]
                if len(h) >= 4:
                    med = np.median(h)
                    mad = np.median(np.abs(np.array(h) - med))
                    if mad > 0:
                        ref[i] = np.clip((x[i] - med) / (1.4826 * mad), -6, 6)
            got = f[dst].to_numpy(float)
            out[f"feat_activity:{mk}:{dst}"] = {"n_valid": int(np.isfinite(got).sum()),
                                                "match": bool(np.allclose(ref, got, equal_nan=True))}
    return out


# ================================================================ 수치 필드(F)
def numeric_field_scan() -> dict:
    rx = re.compile(r"""\[\s*["'](views|up|down)["']\s*\]|\.get\(\s*["'](views|up|down)["']|["'](views)["']""")
    hits = {}
    for p in sorted(CODE.glob("*.py")):
        if p.name in ("audit_leak.py",) or p.name.startswith("test_"):
            continue
        for k, line in enumerate(p.read_text("utf-8").splitlines(), 1):
            if rx.search(line):
                hits.setdefault(p.name, []).append(k)
    u = pd.read_parquet(RESULTS / "units.parquet")
    nc = [c for c in u.columns if c in ("views", "up", "down", "cup", "cdown")]
    return {"code_hits": hits, "units_numeric_cols": nc,
            "units_ncomment_cols": [c for c in u.columns if "ncomment" in c]}


# ================================================================ 실행
def cutoffs(b0: dict, quick: bool) -> list[pd.Timestamp]:
    wk = b0["windows_kr"]
    wk = wk[(wk["variant"] == "main") & wk["complete"]].sort_values("close_kst")
    wu = b0["windows_us"]
    wu = wu[(wu["variant"] == "main") & wu["complete"]].sort_values("close_kst")
    ts = [pd.Timestamp(wk["close_kst"].iloc[25]), pd.Timestamp(wu["close_kst"].iloc[45]),
          pd.Timestamp("2026-10-13 08:00"), pd.Timestamp(wu.loc[wu["win_id"] == "U2026-10-14", "close_kst"].iloc[0]),
          pd.Timestamp("2026-10-15 08:00")]
    return ts[2:4] if quick else ts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-perturb", action="store_true")
    a = ap.parse_args(argv)
    t0 = _time.time()
    base = load_base()
    S = datetime.fromisoformat(base["man"]["S"])
    b0 = run_loader(base["rows"], base)
    report = {"created": datetime.now().isoformat(timespec="seconds"), "tag": a.tag, "S": base["man"]["S"]}
    report["numeric_fields"] = numeric_field_scan()
    report["date_tests"] = date_tests(base, b0["units"])
    report["norm_tests"] = norm_tests()
    report["realdata_norm"] = realdata_norm_check()
    report["residual"] = residual_checks(b0["units"], pd.read_parquet(RESULTS / "feat_activity.parquet"))
    print("date/norm done", round(_time.time() - t0), flush=True)
    if not a.skip_perturb:
        tmp_root = Path(tempfile.mkdtemp(prefix="audit_leak_"))
        try:
            f0 = run_features(b0, tmp_root / "base")
            print("base features", round(_time.time() - t0), flush=True)
            # 하네스가 결과 폴더 산출을 그대로 다시 만드는지
            saved = {"fa": pd.read_parquet(RESULTS / "feat_activity.parquet"),
                     "ft": pd.read_parquet(RESULTS / "feat_text.parquet"),
                     "fto": pd.read_parquet(RESULTS / "feat_topic.parquet")}
            far = pd.Timestamp("2100-01-01")
            f0_legacy = run_features(legacy_trunc(b0), tmp_root / "base_legacy")
            report["harness_vs_saved"] = {}
            for nm, ff in (("fixed", f0), ("legacy_trunc_last", f0_legacy)):
                rep_saved = compare({**ff, **saved}, ff, far)
                report["harness_vs_saved"][nm] = {k: {"rows": v["rows_compared"], "cols": v["cols_compared"],
                                                      "diff_cols": sorted(v["diff"])}
                                                  for k, v in rep_saved.items() if k in saved}
            print("harness vs saved", json.dumps(report["harness_vs_saved"], ensure_ascii=False), flush=True)
            pert = {}
            for k, T in enumerate(cutoffs(b0, a.quick)):
                modes = [("A", False, False), ("B", True, False)]
                if T >= pd.Timestamp("2026-10-13"):          # 다시 받은 줄이 있는 실시간 구간: 고치기 전 동작도 본다
                    modes.append(("A_legacy", False, True))
                for tag, delete, legacy in modes:
                    rows_p, st, max_no = perturb(base["rows"], b0["units"], T, S, SEED + k, delete=delete)
                    bp = run_loader(rows_p, base, high=max(base["high"] or 0, max_no))
                    if legacy:
                        fp = run_features(legacy_trunc(bp), tmp_root / f"p{k}{tag}")
                        cmpr = compare(f0_legacy, fp, T)
                    else:
                        fp = run_features(bp, tmp_root / f"p{k}{tag}")
                        cmpr = compare(f0, fp, T)
                    pert[f"T{k}_{tag}"] = {
                        "T": str(T), "variant": {"A": "A(내용·수치·새 줄)", "B": "B(A + 지우기)",
                                                 "A_legacy": "A, 고치기 전 잘림 표지(trunc_last)"}[tag],
                        "perturb": st, "compare": cmpr, "units": unit_shift_check(b0["units"], bp["units"], T)}
                    nd = sum(len(v["diff"]) for v in cmpr.values())
                    print(f"T{k} {T} {tag}: diff cols {nd}", round(_time.time() - t0), flush=True)
                    shutil.rmtree(tmp_root / f"p{k}{tag}", ignore_errors=True)
                    if tag == "B" and nd:
                        rows_p, st, max_no = perturb(base["rows"], b0["units"], T, S, SEED + k, delete=True,
                                                     keep_roots=True)
                        bp = run_loader(rows_p, base, high=max(base["high"] or 0, max_no))
                        fp = run_features(bp, tmp_root / f"p{k}Bk")
                        cmpr = compare(f0, fp, T)
                        pert[f"T{k}_B_keeproots"] = {"T": str(T), "variant": "B, 깊이 0(뿌리) 미래 댓글은 지우지 않음",
                                                     "perturb": st, "compare": cmpr,
                                                     "units": unit_shift_check(b0["units"], bp["units"], T)}
                        nd = sum(len(v["diff"]) for v in cmpr.values())
                        print(f"T{k} {T} B_keeproots: diff cols {nd}", round(_time.time() - t0), flush=True)
                        shutil.rmtree(tmp_root / f"p{k}Bk", ignore_errors=True)
            report["perturb"] = pert
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
    name = f"audit_leak_{a.tag}.json" if a.tag else "audit_leak.json"
    (RESULTS / name).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), "utf-8")
    print("wrote", name, round(_time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
