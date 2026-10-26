"""바닥 사례 비교(설명용) 점검: compare.py와 따로 짠 계산으로 compare.md 숫자를 다시 만든다. 판정 아님.

① 라벨 날짜 표를 labels/·label_key에서 독립 재계산 → daily_labels.parquet와 대조, 묶음 평균을 daily.parquet·daily_labels·
   daily_topic에서 다시 내 compare.md 표 문자열과 대조, 후보·10월 대조 표 대조, 손으로 쓴 문장 숫자(+17.9% 등) 대조.
② 표본: 층(날짜 × 글/댓글) 크기, 조각 섞임(조각 안 날짜 수, 위치↔날짜 순위 상관), 조각 키, 남은 날짜 단서(계절·요일·상대 날짜).
③ 우연 수준: (a) 바닥 2일 + 급락 4일을 서로 바꾼 15가지 배정(지표 사이 상관 유지)에서 범위 밖 개수 순위,
   (b) 다른 인접 2일(라벨 51일 안, 바닥 이틀 제외)을 가짜 바닥으로 둔 범위 밖 개수 분포, (c) 정규·두꺼운 꼬리 모의 확률.
   (d) 후보 11개에서 7월 바닥 쪽 개수(가짜 바닥 쌍·라벨 51일 하루씩), compare.py에 넣은 같은 계산과 대조.
④ 단정·권유 문장 검색(낱말 목록, 사람이 다시 읽는 것을 대신하지 않는다).
산출: results/audit.json. 실행: py audit.py (compare.py·compare_report.py 다음)
"""
from __future__ import annotations

import glob
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BC = HERE.parent
ANT = BC.parent
PRE = ANT / "preview" / "results"
OUT = BC / "results"
T = pd.Timestamp
EMO = ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR", "HOPE", "REGRET", "ENVY"]
BOTTOM = [T("2026-07-29"), T("2026-07-30")]
DROP = [T("2026-06-23"), T("2026-07-02"), T("2026-07-08"), T("2026-07-13")]


def dr(a, b):
    return list(pd.date_range(a, b, freq="D"))


GROUPS = [("bottom", BOTTOM), ("pre", dr("2026-07-20", "2026-07-28")), ("post", dr("2026-07-31", "2026-08-07")),
          ("drop", DROP), ("sep", dr("2026-09-28", "2026-10-02")), ("oct_fall", dr("2026-10-06", "2026-10-14")),
          ("oct_low", [T("2026-10-15")]), ("oct_after", [T("2026-10-16"), T("2026-10-17")])]
KEYS = (["d_no_z20", "comments_per_post", "missing_rate", "new30_share", "MC_A", "LL_US_A", "LS_A", "KX_A", "ANG_A",
         "SEMI_A", "MEM_A", "TILT"] + [f"share_{e}" for e in EMO] + ["neutral_share", "mixed_share"]
        + [f"mean_{e}" for e in EMO] + ["P_bar", "P_share", "P_strong1"])
LABEL_KEYS = [k for k in KEYS if k.startswith(("share_", "mean_", "neutral", "mixed", "P_"))]
PCT = {"new30_share", "missing_rate", "P_share", "P_strong1", "neutral_share", "mixed_share"}


def fmt(k, v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if k.endswith("_A") or k.startswith("share_") or k in PCT:
        return f"{v * 100:.1f}%"
    if k == "P_bar":
        return f"{v:+.2f}"
    return f"{v:.2f}"


# ---------------------------------------------------------------- ① 라벨 날짜 표 독립 재계산
def my_daily_labels() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(BC / "labels" / "shard_*.json"))):
        for x in json.loads(Path(f).read_text("utf-8")):
            r = {"id": x["id"], "pred": int(x["pred"])}
            r.update({e: int(x["emo"][e]) for e in EMO})
            rows.append(r)
    lab = pd.DataFrame(rows)
    key = pd.read_parquet(BC / "label_key.parquet", columns=["id", "date", "a"])
    m = key.merge(lab, on="id", validate="1:1")
    out = []
    for d, sub in m.groupby("date"):
        auth = {}
        for a, s in sub.groupby("a"):
            emo_max = {e: int(s[e].max()) for e in EMO}
            n_on = (s[EMO] >= 1).sum(axis=1)
            preds = set(s["pred"].tolist())                    # (날짜, 작성자, 값) 쌍 1회
            auth[a] = (emo_max, bool((n_on == 0).all()), bool((n_on >= 2).any()), preds)
        n = len(auth)
        o = {"date": pd.Timestamp(d)}
        for e in EMO:
            v = np.array([z[0][e] for z in auth.values()], float)
            o[f"share_{e}"], o[f"mean_{e}"] = float((v >= 1).mean()), float(v.mean())
        o["neutral_share"] = sum(z[1] for z in auth.values()) / n
        o["mixed_share"] = sum(z[2] for z in auth.values()) / n
        pairs = [p for z in auth.values() for p in z[3]]
        pp = [p for p in pairs if p > 0]
        o["P_bar"] = (np.mean(pp) - 3) if pp else np.nan
        o["P_share"] = len(pp) / len(pairs)
        o["P_strong1"] = (sum(p == 1 for p in pp) / len(pp)) if pp else np.nan
        out.append(o)
    return pd.DataFrame(out)


def my_topic(dates) -> pd.DataFrame:
    u = pd.read_parquet(PRE / "units.parquet", columns=["a", "utype"])
    t = pd.read_parquet(PRE / "feat_topic_unit_hits.parquet", columns=["mem", "semi", "tus", "tkr"])
    h = pd.read_parquet(OUT / "unit_hits.parquet", columns=["uidx", "excl", "wdate"])
    h = h[h["excl"] == ""]
    ix = h["uidx"].to_numpy()
    V = pd.DataFrame({"wdate": h["wdate"].to_numpy(), "a": u["a"].to_numpy()[ix], "post": u["utype"].to_numpy()[ix] == "post",
                      "semi": t["semi"].to_numpy()[ix] > 0, "mem": t["mem"].to_numpy()[ix] > 0,
                      "tus": t["tus"].to_numpy()[ix] > 0, "tkr": t["tkr"].to_numpy()[ix] > 0})
    g = V.groupby(["wdate", "a"])[["semi", "mem"]].max().groupby(level=0).mean()
    P = V[V["post"]].groupby("wdate")[["tus", "tkr"]].sum()
    tilt = np.log(P["tus"] / P["tkr"]).where((P["tus"] > 0) & (P["tkr"] > 0))
    D = pd.DataFrame(index=pd.DatetimeIndex(dates))
    D["SEMI_A"], D["MEM_A"], D["TILT"] = g["semi"].reindex(D.index), g["mem"].reindex(D.index), tilt.reindex(D.index)
    return D


def md_tables(md: str) -> dict[str, list[str]]:
    """compare.md 2절 묶음 표: 지표 이름 → 칸 문자열 8개."""
    sec = md.split("## 2. 묶음별 지표")[1].split("◆ = ")[0]
    out = {}
    for line in sec.splitlines():
        if not line.startswith("| ") or line.startswith("| 지표") or line.startswith("| **"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        out[cells[0].replace(" ◆", "")] = cells[1:]
    return out


# ---------------------------------------------------------------- ③ 우연 수준
def outside(bvals, dvals):
    bvals = [x for x in bvals if np.isfinite(x)]
    dvals = [x for x in dvals if np.isfinite(x)]
    if len(bvals) < 2 or len(dvals) < 3:
        return None, None
    bm, lo, hi = np.mean(bvals), min(dvals), max(dvals)
    side = "위" if bm > hi else ("아래" if bm < lo else None)
    both = len(bvals) == 2 and (min(bvals) > hi or max(bvals) < lo)
    return side, both


def count_outside(X, keys, bdays, ddays):
    n_out = n_both = 0
    for k in keys:
        s = X[k]
        side, both = outside([s.get(d, np.nan) for d in bdays], [s.get(d, np.nan) for d in ddays])
        if side is None and both is None:
            continue
        n_out += side is not None
        n_both += bool(both)
    return n_out, n_both


def main() -> int:
    res: dict = {"purpose": "compare.md 점검(설명용, 판정 아님)"}
    D = pd.read_parquet(OUT / "daily.parquet")
    D["date"] = pd.to_datetime(D["date"]).dt.normalize()
    DL = pd.read_parquet(OUT / "daily_labels.parquet")
    DL["date"] = pd.to_datetime(DL["date"]).dt.normalize()
    TP = pd.read_parquet(OUT / "daily_topic.parquet")
    TP["date"] = pd.to_datetime(TP["date"]).dt.normalize()
    ct = json.loads((OUT / "compare_tables.json").read_text("utf-8"))
    md = (OUT / "compare.md").read_text("utf-8")

    # ① a. 라벨 날짜 표
    mine = my_daily_labels().set_index("date")
    theirs = DL.set_index("date")
    diff = {k: float(np.nanmax(np.abs(mine[k] - theirs[k]))) for k in LABEL_KEYS}
    res["labels_recompute_max_abs_diff"] = {k: v for k, v in diff.items() if v > 1e-12} or "모두 0"
    # ① b. 주제 묶음
    mt = my_topic(D["date"])
    tp = TP.set_index("date")
    res["topic_recompute_max_abs_diff"] = {k: float(np.nanmax(np.abs(mt[k] - tp[k]))) for k in ("SEMI_A", "MEM_A", "TILT")}
    # ① c. 묶음 평균 → compare.md 문자열
    X = D.merge(TP, on="date", how="left").merge(DL.drop(columns=["collection"]), on="date", how="left").set_index("date")
    tab = md_tables(md)
    names = {s["key"]: s["name"] for s in ct["separators"]}
    mism, n_cells = [], 0
    for k in KEYS:
        row = tab[names[k]]
        for j, (gid, days) in enumerate(GROUPS):
            v = X.loc[[d for d in days if d in X.index], k].dropna()
            mine_s = fmt(k, float(v.mean()) if len(v) else None)
            n_cells += 1
            if mine_s != row[j]:
                mism.append({"key": k, "group": gid, "md": row[j], "recomputed": mine_s})
    res["group_table_cells"] = n_cells
    res["group_table_mismatch"] = mism
    # ① d. 후보 표
    cand_md = {}
    sec3 = md.split("## 3.")[1].split("- 확장 급락일")[0]
    for line in sec3.splitlines():
        if line.startswith("| ") and not line.startswith("| 후보"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            cand_md[c[0]] = c
    ext = [T(s) for s in ct["ext_rule_days_used"]]
    lab_days = set(DL["date"])
    cmis, my_cands = [], []
    for k in KEYS:
        s = X[k]
        b = [s.get(d) for d in BOTTOM]
        dv = [s.get(d) for d in DROP if pd.notna(s.get(d))]
        bm, lo, hi = float(np.mean(b)), min(dv), max(dv)
        side = "위" if bm > hi else ("아래" if bm < lo else None)
        if not side:
            continue
        my_cands.append(k)
        both = min(b) > hi or max(b) < lo
        ev = [s.get(d) for d in ext if (k not in LABEL_KEYS or d in lab_days) and pd.notna(s.get(d))]
        ext_ok = bm > max(ev) if side == "위" else bm < min(ev)
        d3 = [s.get(d) for d in DROP[1:] if pd.notna(s.get(d))]
        w0623 = bm > max(d3) if side == "위" else bm < min(d3)
        allv = s.dropna()
        allv = allv[~allv.index.isin(BOTTOM)]
        more = int((allv >= bm).sum() if side == "위" else (allv <= bm).sum())
        row = cand_md.get(names[k])
        yn = lambda x: "예" if x else "아니오"
        mine_row = [names[k], side, f"{fmt(k, b[0])} · {fmt(k, b[1])}", f"{fmt(k, lo)}~{fmt(k, hi)}", yn(both), yn(ext_ok), yn(w0623)]
        if row is None or row[:7] != mine_row or not row[9].startswith(f"{more}/"):
            cmis.append({"key": k, "md": row, "recomputed": mine_row + [more]})
    res["candidates_recomputed"] = my_cands
    res["candidates_md"] = list(cand_md)
    res["candidate_table_mismatch"] = cmis
    # ① e. 10월 대조 표
    sec_o = md.split("### 10월 날들이")[1].split("- 사전 지표는")[0]
    omis = []
    for line in sec_o.splitlines():
        if not line.startswith("| ") or line.startswith("| 지표"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        k = next(kk for kk, nn in names.items() if nn == c[0])
        s = X[k]
        dv = [s.get(d) for d in DROP if pd.notna(s.get(d))]
        lo, hi = min(dv), max(dv)
        bm = float(np.mean([s.get(d) for d in BOTTOM]))
        side = "위" if bm > hi else "아래"

        def where(v):
            if pd.isna(v):
                return "—"
            if lo <= v <= hi:
                return f"{fmt(k, v)} 급락일 범위 안"
            return f"{fmt(k, v)} " + ("7월 바닥 쪽" if (v > hi) == (side == "위") else "반대쪽")
        fall = s.reindex(dr("2026-10-06", "2026-10-14")).dropna().mean()
        mine_row = [where(fall), where(s.get(T("2026-10-15"))), where(s.get(T("2026-10-16"))), where(s.get(T("2026-10-17")))]
        if c[3:7] != mine_row:
            omis.append({"key": k, "md": c[3:7], "recomputed": mine_row})
    res["october_table_mismatch"] = omis
    # ① f. 손으로 쓴 문장 숫자
    p = pd.read_parquet(PRE / "prices.parquet")
    k_ = p[p["ticker"] == "^KS11"].set_index("date")["close"].sort_index()
    k_.index = pd.to_datetime(k_.index)
    r_ = k_.pct_change()
    hand = {"7/31 KOSPI 등락": round(float(r_[T("2026-07-31")]) * 100, 1),
            "10/16 KOSPI 등락": round(float(r_[T("2026-10-16")]) * 100, 1),
            "7/15 KOSPI 등락": round(float(r_[T("2026-07-15")]) * 100, 1)}
    br = {}
    for kk in ("MC_A", "LS_A"):
        s = X[kk]
        x = s.to_numpy(float)
        ok = X["complete"].to_numpy(bool) & np.isfinite(x)
        rr = pd.Series(np.nan, index=X.index)
        for i in range(20, len(x)):
            if ok[i - 20:i].all():
                rr.iloc[i] = x[i] / x[i - 20:i].mean()
        br[kk] = {str(d.date()): round(float(rr[d]), 2) for d in BOTTOM + DROP}
    hand["배수판(값÷직전 20일 평균)"] = br
    u = pd.read_parquet(PRE / "units.parquet", columns=["utype", "dt_kst", "title", "text"])
    u["dt_kst"] = u["dt_kst"].astype("datetime64[ns]")
    w = u[(u["dt_kst"] >= T("2026-07-30 08:00")) & (u["dt_kst"] < T("2026-07-31 08:00"))]
    txt = (w["title"].fillna("") + " " + w["text"].fillna(""))
    late = w[(w["dt_kst"] >= T("2026-07-30 15:30"))]
    lt = txt.loc[late.index]
    hand["7/30 창 15:30 뒤 '야간 선물'·'야선' 글·댓글"] = int(lt.str.contains("야간\\s?선물|야선").sum())
    hand["7/30 창 15:30 뒤 '상한가' 글·댓글"] = int(lt.str.contains("상한가").sum())
    hand["7/30 창 15:30 뒤 단위 몫"] = round(len(late) / len(w), 3)
    res["hand_written_numbers"] = hand

    # ② 표본
    key = pd.read_parquet(BC / "label_key.parquet")
    strata = key.groupby(["date", "kind"]).size().unstack()
    res["strata"] = {"days": int(len(strata)), "post_min_max": [int(strata["글"].min()), int(strata["글"].max())],
                     "comment_min_max": [int(strata["댓글"].min()), int(strata["댓글"].max())]}
    rk = key["date"].rank()
    order = (key["shard"] - 1) * 300 + key["shard_pos"]
    res["shard_order_vs_date_spearman"] = round(float(np.corrcoef(rk, order.rank())[0, 1]), 4)
    res["shard_days"] = {"min": int(key.groupby("shard")["date"].nunique().min()),
                         "max": int(key.groupby("shard")["date"].nunique().max())}
    shard_keys = set()
    texts = {}
    for f in sorted(glob.glob(str(BC / "shards" / "shard_*.json"))):
        for x in json.loads(Path(f).read_text("utf-8")):
            shard_keys.add(tuple(sorted(x)))
            texts[x["id"]] = x["text"]
    res["shard_fields"] = sorted({k for t in shard_keys for k in t})
    lab_fields = set()
    for f in sorted(glob.glob(str(BC / "labels" / "shard_*.json"))):
        for x in json.loads(Path(f).read_text("utf-8")):
            lab_fields |= set(x)
    res["label_fields"] = sorted(lab_fields)
    kd = key.set_index("id")
    clues = {"여름(장마 빼고)": r"(?<!장)마?(?:폭염|더위|더운|무더위|여름|휴가|열대야)", "가을·연휴": r"가을|추워지|쌀쌀|단풍|연휴|국경절",
             "요일": r"[월화수목금토일]요일|[월화수목금]욜", "상대 날짜(오늘·어제·내일)": r"오늘|어제|내일"}
    cl = {}
    for nm, rx in clues.items():
        r = re.compile(rx)
        hit = [i for i, t in texts.items() if r.search(t)]
        by = kd.loc[hit, "block"].value_counts().to_dict() if hit else {}
        cl[nm] = {"units": len(hit), "share": round(len(hit) / len(texts), 4), "by_block": {k: int(v) for k, v in by.items()}}
    res["date_clues_in_shards"] = cl

    # ③ 우연 수준
    keys_checked = [k for k in KEYS]
    obs = count_outside(X, keys_checked, BOTTOM, DROP)
    six = BOTTOM + DROP
    perm = []
    for pair in itertools.combinations(six, 2):
        rest = [d for d in six if d not in pair]
        perm.append({"pair": [str(d.date()) for d in pair], "n_out": None, "n_both": None})
        perm[-1]["n_out"], perm[-1]["n_both"] = count_outside(X, keys_checked, list(pair), rest)
    n_out_all = np.array([p["n_out"] for p in perm])
    n_both_all = np.array([p["n_both"] for p in perm])
    res["observed"] = {"n_out": obs[0], "n_both": obs[1]}
    res["exchange_15"] = {"assignments": perm,
                          "rank_p_out": round(float((n_out_all >= obs[0]).mean()), 3),
                          "rank_p_both": round(float((n_both_all >= obs[1]).mean()), 3),
                          "median_out": float(np.median(n_out_all)), "median_both": float(np.median(n_both_all))}
    # (b) 가짜 바닥: 라벨 51일 안 인접 2일(둘 다 평일 거래일), 바닥 이틀·급락 4일과 겹치지 않음
    ld = sorted(lab_days)
    trade = set(D.loc[D["krx_session"], "date"])
    plac = []
    for a, b in zip(ld[:-1], ld[1:]):
        if (b - a).days != 1 or a not in trade or b not in trade:
            continue
        if {a, b} & set(BOTTOM + DROP):
            continue
        n_o, n_b = count_outside(X, keys_checked, [a, b], DROP)
        plac.append({"pair": [str(a.date()), str(b.date())], "n_out": n_o, "n_both": n_b})
    po = np.array([p["n_out"] for p in plac])
    pb = np.array([p["n_both"] for p in plac])
    res["placebo_adjacent_pairs"] = {"n_pairs": len(plac), "n_out_min_median_max": [int(po.min()), float(np.median(po)), int(po.max())],
                                     "n_both_min_median_max": [int(pb.min()), float(np.median(pb)), int(pb.max())],
                                     "share_pairs_n_out_ge_obs": round(float((po >= obs[0]).mean()), 3),
                                     "share_pairs_n_both_ge_obs": round(float((pb >= obs[1]).mean()), 3), "pairs": plac}
    # (b2) 후보 11개에서 '7월 바닥과 같은 쪽' 개수: 가짜 바닥 쌍(평균), 라벨 51일 하루씩
    cands = res["candidates_recomputed"]
    side_of = {}
    for k in cands:
        s = X[k]
        dv = [s.get(d) for d in DROP if pd.notna(s.get(d))]
        side_of[k] = ("위" if np.mean([s.get(d) for d in BOTTOM]) > max(dv) else "아래", min(dv), max(dv))

    def n_same(vals: dict) -> int:
        n = 0
        for k, (sd, lo, hi) in side_of.items():
            v = vals.get(k)
            if v is not None and np.isfinite(v) and ((sd == "위" and v > hi) or (sd == "아래" and v < lo)):
                n += 1
        return n
    for pp_ in plac:
        a, b = T(pp_["pair"][0]), T(pp_["pair"][1])
        pp_["n_same_side_of_11"] = n_same({k: float(np.nanmean([X.loc[a, k], X.loc[b, k]])) for k in cands})
    ps = np.array([pp_["n_same_side_of_11"] for pp_ in plac])
    res["placebo_adjacent_pairs"]["n_same_side_of_11_min_median_max"] = [int(ps.min()), float(np.median(ps)), int(ps.max())]
    res["placebo_adjacent_pairs"]["share_pairs_same_side_ge_9"] = round(float((ps >= 9).mean()), 3)
    single = {str(d.date()): n_same({k: X.loc[d, k] for k in cands}) for d in ld}
    oth = [v for d, v in single.items() if T(d) not in set(BOTTOM)]
    res["single_day_same_side_of_11"] = {"by_day": single, "others_min_median_max": [int(min(oth)), float(np.median(oth)), int(max(oth))],
                                         "n_other_days": len(oth),
                                         "other_days_ge_7": int(sum(v >= 7 for v in oth)), "other_days_ge_5": int(sum(v >= 5 for v in oth)),
                                         "rank_1016_from_top": int(sum(v > single["2026-10-16"] for v in oth)) + 1,
                                         "top_days": sorted(single.items(), key=lambda kv: -kv[1])[:10]}

    # (c) 모의 확률(분포 가정 민감도)
    rng = np.random.default_rng(20261019)
    sim = {}
    for nm, gen in (("정규", lambda s: rng.standard_normal(s)), ("t3", lambda s: rng.standard_t(3, s)),
                    ("로그정규", lambda s: rng.lognormal(0, 1, s))):
        x = gen((200_000, 6))
        ref, bb = x[:, :4], x[:, 4:]
        mb = bb.mean(1)
        sim[nm] = round(float(((mb > ref.max(1)) | (mb < ref.min(1))).mean()), 4)
    sim["두 날 모두 같은 쪽(정확, 분포 무관)"] = round(2 / math.comb(6, 2), 4)
    res["chance_sim_p_mean_outside_n4"] = sim

    # compare.py에 넣은 같은 계산과 대조
    ce = ct.get("chance_exchange_6days", {})
    mine_ex = sorted((tuple(a["pair"]), a["n_out"], a["n_both"]) for a in res["exchange_15"]["assignments"])
    theirs_ex = sorted((tuple(a["pair"]), a["n_out"], a["n_both"]) for a in ce.get("assignments", []))
    cp = {x["date"]: x["n_same_side"] for x in ct.get("placebo_days_same_side", {}).get("by_day", [])}
    res["cross_check_compare_py"] = {"exchange_same": mine_ex == theirs_ex,
                                     "same_side_by_day_same": cp == res["single_day_same_side_of_11"]["by_day"]}

    # ④ 단정·권유 문장
    pat = re.compile(r"바닥(?:이다|입니다|이에요|일 것|일 가능성이 높|을 찍었|확인)|저점(?:이다|입니다|확인|을 찍었)|매수|사라|사세요|"
                     r"담(?:아라|으세요|을 때)|들어가|기회|추천|반등할|오를 것")
    hits = []
    for i, line in enumerate(md.splitlines(), 1):
        for mm in pat.finditer(line):
            hits.append({"line": i, "match": mm.group(0), "text": line[:160]})
    res["assertive_phrase_hits"] = hits

    res["inputs_sha256"] = {p_.name: hashlib.sha256(p_.read_bytes()).hexdigest() for p_ in
                            (OUT / "daily.parquet", OUT / "daily_labels.parquet", OUT / "daily_topic.parquet",
                             OUT / "compare_tables.json", OUT / "compare.md", BC / "label_key.parquet")}
    (OUT / "audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), "utf-8")
    show = {k: v for k, v in res.items() if k not in ("inputs_sha256",)}
    show["exchange_15"] = {k: v for k, v in res["exchange_15"].items() if k != "assignments"} | {
        "n_out_by_assignment": [(a["pair"], a["n_out"], a["n_both"]) for a in res["exchange_15"]["assignments"]]}
    show["placebo_adjacent_pairs"] = {k: v for k, v in res["placebo_adjacent_pairs"].items() if k != "pairs"}
    print(json.dumps(show, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
