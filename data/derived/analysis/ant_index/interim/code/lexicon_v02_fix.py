"""사전 v0.2 고정: 두 라벨러의 가림 표본 라벨로 묶음 정밀도·κ·단어 정밀도·반어율·BULL/BEAR 정확도를 내고,
HYPOTHESES.md §2.7·§4 규칙(적중 10회 이상 단어의 정밀도 50% 미만 → 뺀다)만 적용해 v0.2를 쓴다.

입력(읽기만): results/blind_samples.json, results/labels_1.json, results/labels_2.json,
             results/blind_key.parquet(id·층·적중·사전 점수 열만. 날짜 열은 읽지 않는다),
             results/lexicon_v02_candidate.json, results/blind_summary.json(모집단 단어 적중 수)
출력: results/lexicon_v02.json(층 → 단어 목록), results/lexicon_v02_full.json(규칙 포함 판),
      results/lexicon_v02_precision.json(수치), results/lexicon_v02_precision.md(표)
가격 자료는 읽지 않는다. 새 단어를 더하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

MIN_WORD_HITS = 10        # §2.7 '적중 10회 이상인 단어'
WORD_PREC_MIN = 0.50      # §2.7 단어 정밀도 50% 이상
BUNDLE_PREC_MIN = 0.60    # §2.7 묶음 정밀도 60% 이상
IRONY_MAX = 0.30          # §2.7 반어율 30% 이하
SCORE_ACC_MIN = 0.60      # H06 강세·약세 분류 정확도 60%
KAPPA_FLAG = 0.60         # §2.7 LLM κ 0.6(참고 표시만, 사전 판정 기준 아님)

RULE_ITEMS = {"MOCK_BEAR": ["주식전문가(단독)", "ㅋ4개 이상+2인칭"]}   # lexicon_v02.py 특수 규칙 2개(§4 MOCK 약세론 10)

USES = {   # HYPOTHESES.md §4 '쓰는 가설'
    "E": "S(H01·H14·H15·H51·H49·H53), H05, H19", "F": "S, H05, H24", "PX": "감정 셈 빼기용·메아리 계기",
    "CAP": "H25", "HES": "H25", "LL_KR": "국내판 보조", "LL_US": "H31", "LS": "H31 비율판(보조)",
    "MC": "H30, H33", "NOV": "H35 보조", "FO": "H05, H21", "RES": "H26", "ANG": "H05, H26", "KX": "H47",
    "KR": "H47", "BUY": "통제", "BULL": "H06, H09~H12, H22, H37", "BEAR": "H06, H09~H12, H22, H37",
    "MOCK_BEAR": "H13", "MOCK_BULL": "H13", "REBUT": "H22", "REBUT_FEAR": "H22", "CHASE_BUY": "H43",
    "CHASE_SELL": "H43", "CONV_BULL": "H37 보조", "CONV_BEAR": "H37 보조", "DIST": "H30", "IPO": "H45",
    "MEM": "H50", "SEMI": "H10", "TILT_US": "H46", "TILT_KR": "H46",
    "ALIAS_SEC": "H38, H39+H48, H40, H41, H43", "ALIAS_SKH": "H38, H39+H48, H40, H41, H43",
    "ALIAS_BOTH": "H38, H39+H48, H40, H41, H43", "ALIAS_US": "H39+H48, H40, H41, H43",
}

SPACE_CH = set(" \t\r　")


def compact(s: str) -> str:
    return "".join(ch for ch in s if ch not in SPACE_CH).lower()


def yes(v) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v in ("예", "true", "True"):
        return True
    if v in ("아니오", "false", "False"):
        return False
    raise ValueError(f"알 수 없는 값: {v!r}")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def kappa(a: list, b: list) -> tuple[float | None, float]:
    """Cohen κ(범주 수 무관)와 관측 일치율. 기대 일치가 1이면 κ는 정의되지 않는다(None)."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if abs(1 - pe) < 1e-12:
        return None, po
    return (po - pe) / (1 - pe), po


def direction(score: float) -> str:
    return "강세" if score > 0 else ("약세" if score < 0 else "중립")


def post_score(hits: list[dict], drop: dict[str, set[str]]) -> float:
    s = 0
    for h in hits:
        if compact(h["word"]) in drop.get(h["stratum"], set()):
            continue
        s += h["sign"] * (2 if h["in_title"] else 1)
    return float(max(-3, min(3, s)))


def main() -> int:
    samples = json.loads((RES / "blind_samples.json").read_text(encoding="utf-8"))
    l1 = {r["id"]: r for r in json.loads((RES / "labels_1.json").read_text(encoding="utf-8"))}
    l2 = {r["id"]: r for r in json.loads((RES / "labels_2.json").read_text(encoding="utf-8"))}
    cand_bytes = (RES / "lexicon_v02_candidate.json").read_bytes()
    cand = json.loads(cand_bytes.decode("utf-8"))
    summary = json.loads((RES / "blind_summary.json").read_text(encoding="utf-8"))
    key = pd.read_parquet(RES / "blind_key.parquet", columns=["id", "set", "stratum", "hits_json", "dict_score"])
    key = key.set_index("id")

    assert set(l1) == set(l2), "두 라벨 파일의 id가 다르다"
    ids_samples = {s["id"] for s in samples["samples"]} | {s["id"] for s in samples["score_samples"]}
    assert ids_samples == set(l1), "라벨 id와 표본 id가 다르다"

    strata = list(dict.fromkeys(s["stratum"] for s in cand["spec"]))
    spec_words: dict[str, list[str]] = defaultdict(list)
    for s in cand["spec"]:
        spec_words[s["stratum"]].extend(s["words"])
    display: dict[tuple[str, str], str] = {}
    for st in strata:
        for w in spec_words[st] + RULE_ITEMS.get(st, []):
            display.setdefault((st, compact(w)), w)

    # ---------------------------------------------------------------- 묶음(층)별
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for s in samples["samples"]:
        by_stratum[s["bundle"]].append(s)

    consist = Counter()          # 단위 판정 = 적중 판정의 OR 인지
    hit_rows = []                # 적중 하나씩
    bundle_rows = {}
    for st in strata:
        units = by_stratum[st]
        a = [yes(l1[u["id"]]["in_sense"]) for u in units]
        b = [yes(l2[u["id"]]["in_sense"]) for u in units]
        ia = [yes(l1[u["id"]]["irony"]) for u in units]
        ib = [yes(l2[u["id"]]["irony"]) for u in units]
        n = len(units)
        both = sum(x and y for x, y in zip(a, b))
        neither = sum((not x) and (not y) for x, y in zip(a, b))
        d12 = sum(x and not y for x, y in zip(a, b))
        d21 = sum(y and not x for x, y in zip(a, b))
        k, po = kappa(a, b)
        kirony, _ = kappa(ia, ib)
        ir_both = sum(x and y for x, y in zip(ia, ib))
        ir_any = sum(x or y for x, y in zip(ia, ib))
        lo, hi = wilson(both, n)
        bundle_rows[st] = dict(n=n, both_yes=both, both_no=neither, dis_1y2n=d12, dis_1n2y=d21,
                               l1_yes=sum(a), l2_yes=sum(b), precision=both / n, prec_ci=(lo, hi),
                               upper=(both + d12 + d21) / n, kappa=k, agree=po,
                               irony_both=ir_both, irony_any=ir_any, irony_l1=sum(ia), irony_l2=sum(ib),
                               irony_rate_both=ir_both / n, irony_rate_any=ir_any / n, kappa_irony=kirony)
        for u in units:
            r1, r2 = l1[u["id"]], l2[u["id"]]
            multi = len(u["hits"]) >= 2
            for lab, r in (("1", r1), ("2", r2)):
                if multi:
                    hs = r.get("hits_in_sense") or {}
                    ok = set(hs) == {str(h["k"]) for h in u["hits"]}
                    consist[(lab, "keys_ok")] += ok
                    consist[(lab, "keys_bad")] += (not ok)
                    if ok:
                        orv = any(yes(v) for v in hs.values())
                        consist[(lab, "or_match" if orv == yes(r["in_sense"]) else "or_mismatch")] += 1
            for h in u["hits"]:
                if multi:
                    y1 = yes((r1.get("hits_in_sense") or {}).get(str(h["k"])))
                    y2 = yes((r2.get("hits_in_sense") or {}).get(str(h["k"])))
                else:
                    y1, y2 = yes(r1["in_sense"]), yes(r2["in_sense"])
                note = h.get("note", "")
                src = note.split(" ")[0] if note else st     # 옮겨 온 적중은 원래 묶음 목록의 단어다
                hit_rows.append(dict(stratum=st, src=src, uid=u["id"], k=h["k"], word=h["word"],
                                     key=compact(h["word"]), note=note, y1=bool(y1), y2=bool(y2)))

    # ---------------------------------------------------------------- 단어별(옮겨 온 적중 제외)
    pop = summary["word_population_hits"]
    pop_key: dict[str, Counter] = {st: Counter() for st in strata}
    for st, d in pop.items():
        for w, c in d.items():
            pop_key[st][compact(w)] += c
    word_rows = []
    agg: dict[tuple[str, str], dict] = {}
    for h in hit_rows:
        if h["note"]:
            continue
        g = agg.setdefault((h["stratum"], h["key"]), dict(n=0, both=0, either=0, units=defaultdict(list)))
        g["n"] += 1
        g["both"] += h["y1"] and h["y2"]
        g["either"] += h["y1"] or h["y2"]
        g["units"][h["uid"]].append(h["y1"] and h["y2"])
    removed: dict[str, set[str]] = defaultdict(set)
    for (st, kk), g in sorted(agg.items(), key=lambda x: (strata.index(x[0][0]), -x[1]["n"])):
        nu = len(g["units"])
        bu = sum(any(v) for v in g["units"].values())
        prec = g["both"] / g["n"]
        drop = g["n"] >= MIN_WORD_HITS and prec < WORD_PREC_MIN
        drop_unit = nu >= MIN_WORD_HITS and bu / nu < WORD_PREC_MIN       # 민감도: 단위·단어 하나로 묶은 판
        drop_either = g["n"] >= MIN_WORD_HITS and g["either"] / g["n"] < WORD_PREC_MIN   # 민감도: 한쪽 이상 예
        if drop:
            removed[st].add(kk)
        word_rows.append(dict(stratum=st, word=display.get((st, kk), kk), key=kk, n=g["n"], both=g["both"],
                              either=g["either"], precision=prec, n_units=nu, both_units=bu,
                              pop_hits=pop_key.get(st, Counter()).get(kk, 0), drop=drop,
                              drop_unit_dedup=drop_unit, drop_either=drop_either))

    # 옮겨 온 적중(규칙 이동) 따로
    moved = [h for h in hit_rows if h["note"]]
    moved_rows = Counter((h["stratum"], h["note"], h["word"], h["y1"] and h["y2"]) for h in moved)

    # 빼고 난 뒤(같은 표본, 낙관 편향) 추정
    for st in strata:
        units = defaultdict(list)
        for h in hit_rows:
            if h["stratum"] != st:
                continue
            if h["key"] in removed.get(h["src"], set()):
                continue
            units[h["uid"]].append(h["y1"] and h["y2"])
        kept = len(units)
        yes_kept = sum(any(v) for v in units.values())
        tot_pop = sum(pop_key.get(st, Counter()).values())
        rem_pop = sum(pop_key.get(st, Counter()).get(kk, 0) for kk in removed.get(st, set()))
        bundle_rows[st].update(kept_units=kept, kept_yes=yes_kept,
                               prec_after=(yes_kept / kept) if kept else float("nan"),
                               n_removed=len(removed.get(st, set())),
                               pop_hits_total=tot_pop, pop_hits_removed=rem_pop)

    # 판정
    for st, r in bundle_rows.items():
        why = []
        if r["precision"] < BUNDLE_PREC_MIN:
            why.append(f"묶음 정밀도 {r['precision']:.0%} < 60%")
        if r["irony_rate_any"] > IRONY_MAX:
            why.append(f"반어율 {r['irony_rate_any']:.0%} > 30%")
        r["fail"] = bool(why)
        r["fail_reason"] = "; ".join(why)
        r["kappa_low"] = (r["kappa"] is not None and r["kappa"] < KAPPA_FLAG)

    # ---------------------------------------------------------------- BULL/BEAR 정확도
    def score_block(rows: list[tuple[str, str | None, str | None]]) -> dict:
        """rows: (사전 방향, 라벨러1 극성, 라벨러2 극성)"""
        nz = [r for r in rows if r[0] != "중립"]
        strict = sum(r[1] == r[2] == r[0] for r in nz)
        agreed = [r for r in nz if r[1] == r[2]]
        agreed_ok = sum(r[1] == r[0] for r in agreed)
        all_strict = sum(r[1] == r[2] == r[0] for r in rows)
        all_agreed = [r for r in rows if r[1] == r[2]]
        gold_dir = [r for r in all_agreed if r[1] in ("강세", "약세")]
        recall = sum(r[0] == r[1] for r in gold_dir)
        opp = sum(r[1] == r[2] and r[1] in ("강세", "약세") and r[0] in ("강세", "약세") and r[1] != r[0] for r in nz)
        return dict(n=len(rows), n_nonzero=len(nz), nz_strict=strict,
                    nz_acc_strict=(strict / len(nz)) if nz else float("nan"),
                    nz_acc_ci=wilson(strict, len(nz)) if nz else (float("nan"),) * 2,
                    nz_agreed=len(agreed), nz_agreed_ok=agreed_ok,
                    nz_acc_agreed=(agreed_ok / len(agreed)) if agreed else float("nan"),
                    nz_disagree=len(nz) - len(agreed), nz_opposite=opp,
                    all_acc_strict=all_strict / len(rows), all_agreed=len(all_agreed),
                    all_acc_agreed=(sum(r[1] == r[0] for r in all_agreed) / len(all_agreed)) if all_agreed else float("nan"),
                    gold_directional=len(gold_dir), recall=(recall / len(gold_dir)) if gold_dir else float("nan"),
                    dict_dirs=dict(Counter(r[0] for r in rows)))

    s200 = []
    recompute_bad = 0
    for s in samples["score_samples"]:
        kr = key.loc[s["id"]]
        hits = json.loads(kr["hits_json"])
        sc_c = post_score(hits, {})
        if abs(sc_c - float(kr["dict_score"])) > 1e-9:
            recompute_bad += 1
        sc_v = post_score(hits, removed)
        s200.append((s["id"], sc_c, sc_v, l1[s["id"]]["polarity"], l2[s["id"]]["polarity"]))
    pol_k, pol_po = kappa([r[3] for r in s200], [r[4] for r in s200])
    score = dict(
        recompute_mismatch=recompute_bad,
        cand=score_block([(direction(r[1]), r[3], r[4]) for r in s200]),
        v02=score_block([(direction(r[2]), r[3], r[4]) for r in s200]),
        labeler_kappa=pol_k, labeler_agree=pol_po,
        labeler_dist={"1": dict(Counter(r[3] for r in s200)), "2": dict(Counter(r[4] for r in s200))},
        changed_by_v02=sum(direction(r[1]) != direction(r[2]) for r in s200),
    )
    prec_bb = {}
    for st in ("BULL", "BEAR"):
        rows = []
        for u in by_stratum[st]:
            rows.append((direction(float(key.loc[u["id"]]["dict_score"])), l1[u["id"]]["polarity"], l2[u["id"]]["polarity"]))
        blk = score_block(rows)
        kk, po = kappa([r[1] for r in rows], [r[2] for r in rows])
        blk.update(labeler_kappa=kk, labeler_agree=po)
        prec_bb[st] = blk
    h06_fail = score["v02"]["nz_acc_strict"] < SCORE_ACC_MIN
    h06_fail_cand = score["cand"]["nz_acc_strict"] < SCORE_ACC_MIN

    # ---------------------------------------------------------------- v0.2 사전 쓰기
    v02_map: dict[str, list[str]] = {}
    for st in strata:
        words = [w for w in spec_words[st] if compact(w) not in removed.get(st, set())]
        words += [w for w in RULE_ITEMS.get(st, []) if compact(w) not in removed.get(st, set())]
        v02_map[st] = words
    map_bytes = (json.dumps(v02_map, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    (RES / "lexicon_v02.json").write_bytes(map_bytes)
    map_sha = hashlib.sha256(map_bytes).hexdigest()

    full_spec = []
    for s in cand["spec"]:
        s2 = dict(s)
        s2["words"] = [w for w in s["words"] if compact(w) not in removed.get(s["stratum"], set())]
        full_spec.append(s2)
    full = {
        "version": "v0.2 (정밀도 점검 뒤 고정, 2026-10-16)",
        "base": {"file": "lexicon_v02_candidate.json",
                 "sha256": hashlib.sha256(cand_bytes).hexdigest()},
        "spec": full_spec,
        "rule_items": {st: [w for w in ws if compact(w) not in removed.get(st, set())] for st, ws in RULE_ITEMS.items()},
        "rule_items_candidate": RULE_ITEMS,
        "matching": ("후보 매처(lexicon_v02.py, base.sha256)를 그대로 쓰고 removed의 단어만 막는다. 막힌 단어는 가장 긴 표현 먼저 "
                     "규칙에서 자리를 차지한 채 버려진다(짧은 단어가 그 자리를 다시 잡지 않음). 규칙으로 옮겨 가는 적중은 원래 묶음 "
                     "기준으로 막는다. 구현: interim/code/lexicon_v02_final.py load_v02()"),
        "rules": cand["rules"],
        "removed": {st: sorted(display.get((st, kk), kk) for kk in ks) for st, ks in removed.items() if ks},
        "status": {st: ("측정 실패: " + bundle_rows[st]["fail_reason"]) if bundle_rows[st]["fail"] else "통과"
                   for st in strata},
        "h06_score_accuracy": "측정 실패" if h06_fail else "통과",
        "lexicon_v02_json_sha256": map_sha,
    }
    full_bytes = (json.dumps(full, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode("utf-8")
    (RES / "lexicon_v02_full.json").write_bytes(full_bytes)
    full_sha = hashlib.sha256(full_bytes).hexdigest()

    stats = dict(bundles=bundle_rows, words=word_rows,
                 moved=[dict(stratum=a, note=b, word=c, both_yes=d, n=v) for (a, b, c, d), v in moved_rows.items()],
                 consistency={f"{a}:{b}": v for (a, b), v in consist.items()},
                 score200=score, precision_bullbear=prec_bb, h06_fail=h06_fail, h06_fail_cand=h06_fail_cand,
                 sha256={"lexicon_v02.json": map_sha, "lexicon_v02_full.json": full_sha,
                         "lexicon_v02_candidate.json": hashlib.sha256(cand_bytes).hexdigest(),
                         "labels_1.json": hashlib.sha256((RES / "labels_1.json").read_bytes()).hexdigest(),
                         "labels_2.json": hashlib.sha256((RES / "labels_2.json").read_bytes()).hexdigest(),
                         "blind_samples.json": hashlib.sha256((RES / "blind_samples.json").read_bytes()).hexdigest(),
                         "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                 params=dict(MIN_WORD_HITS=MIN_WORD_HITS, WORD_PREC_MIN=WORD_PREC_MIN, BUNDLE_PREC_MIN=BUNDLE_PREC_MIN,
                             IRONY_MAX=IRONY_MAX, SCORE_ACC_MIN=SCORE_ACC_MIN))

    cand_counts = {st: len(spec_words[st]) + len(RULE_ITEMS.get(st, [])) for st in strata}
    ver = verify_population(strata, removed)
    stats["verify"] = ver
    (RES / "lexicon_v02_precision.json").write_bytes(
        (json.dumps(stats, ensure_ascii=False, indent=1, default=str) + "\n").encode("utf-8"))
    write_md(strata, bundle_rows, word_rows, removed, stats, v02_map, display, cand_counts)
    print(json.dumps({"lexicon_v02.json": map_sha, "lexicon_v02_full.json": full_sha,
                      "removed": {st: len(v) for st, v in removed.items() if v},
                      "fail": [st for st in strata if bundle_rows[st]["fail"]],
                      "h06_fail": h06_fail, "consistency": stats["consistency"],
                      "recompute_mismatch": recompute_bad,
                      "verify": {k: v for k, v in ver.items() if k != "strata"}}, ensure_ascii=False))
    return 0


def verify_population(strata: list[str], removed: dict[str, set[str]]) -> dict:
    """가림 표본과 같은 틀(blind_sample.load_frame)에서 후보판·v0.2판 매처를 돌려
    ① 후보판 단어 적중 수가 blind_summary와 같은지 ② v0.2 적중 = 후보 적중 − 뺀 단어 적중(새 적중 0)인지
    ③ v0.2 글 점수 = 남은 적중으로 다시 센 점수인지 확인하고, 층별 적중 단위·적중 수를 낸다."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import blind_sample as BS          # noqa: E402
    import lexicon_v02 as LX           # noqa: E402
    from lexicon_v02_final import load_v02  # noqa: E402

    analyze_v02 = load_v02()
    summary = json.loads((RES / "blind_summary.json").read_text(encoding="utf-8"))
    e, _ = BS.load_frame(RES)
    rule_names = {w for ws in RULE_ITEMS.values() for w in ws}

    def src_key(h):
        src = h.origin.split(" ")[0] if h.origin else h.stratum
        return src, (h.word if h.word in rule_names else compact(h.word))

    def blocked(h):
        src, kk = src_key(h)
        return (kk in removed.get(src, set())) or (h.word in rule_names and compact(h.word) in removed.get(src, set()))

    pop_c: dict[str, Counter] = {st: Counter() for st in strata}
    units_c, units_v, hits_c, hits_v = Counter(), Counter(), Counter(), Counter()
    bad_subset = bad_score = new_hits = 0
    n = 0
    for t, b, ut in zip(e["title"].tolist(), e["text"].tolist(), e["utype"].tolist()):
        n += 1
        title = t if ut == "post" else None
        rc = LX.analyze(title, b, ut)
        rv = analyze_v02(title, b, ut)
        if rc.excluded != rv.excluded:
            bad_subset += 1
            continue
        if rc.excluded:
            continue
        hc = [h for h in rc.hits if BS._scope_ok(h.stratum, ut)]
        hv = [h for h in rv.hits if BS._scope_ok(h.stratum, ut)]
        for h in hc:
            pop_c[h.stratum][h.word] += 1
        exp = [(h.stratum, h.word, h.start, h.end, h.sign) for h in hc if not blocked(h)]
        got = [(h.stratum, h.word, h.start, h.end, h.sign) for h in hv]
        if sorted(exp) != sorted(got):
            bad_subset += 1
            new_hits += len(set(got) - set(exp))
        if ut == "post":
            s = sum(h.sign * (2 if h.in_title else 1) for h in rv.hits if h.stratum in ("BULL", "BEAR"))
            if rv.score != max(-3, min(3, s)):
                bad_score += 1
        for st in {h.stratum for h in hc}:
            units_c[st] += 1
        for st in {h.stratum for h in hv}:
            units_v[st] += 1
        for h in hc:
            hits_c[h.stratum] += 1
        for h in hv:
            hits_v[h.stratum] += 1
    want = summary["word_population_hits"]
    pop_match = all(dict(pop_c[st]) == want.get(st, {}) for st in strata)
    return dict(frame_units=n, candidate_sha=LX.lexicon_sha256(), pop_hits_match_summary=pop_match,
                v02_not_equal_cand_minus_removed=bad_subset, v02_new_hits=new_hits, v02_score_mismatch=bad_score,
                strata={st: dict(units_cand=units_c[st], units_v02=units_v[st], hits_cand=hits_c[st],
                                 hits_v02=hits_v[st]) for st in strata})


def pct(x: float | None, d: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.{d}f}%"


def fk(x: float | None) -> str:
    return "정의 안 됨" if x is None else f"{x:.2f}"


def write_md(strata, B, W, removed, stats, v02_map, display, cand_counts) -> None:
    sc = stats["score200"]
    L = []
    a = L.append
    a("# 사전 v0.2 정밀도 점검 결과 (2026-10-16)")
    a("")
    a("> 동결 문서 `HYPOTHESES.md` §2.7·§4, `lexicon_draft.md` §8을 따랐다. 두 라벨러(labels_1·labels_2)의 가림 표본 라벨만 썼고 가격 자료는 읽지 않았다.")
    a("> 코드 `interim/code/lexicon_v02_fix.py`. 해석 기록은 `interim/notes_lexicon_v02.md`.")
    a("")
    a("## 지문")
    a("| 파일 | SHA-256 |")
    a("|---|---|")
    for f, h in stats["sha256"].items():
        a(f"| {f} | `{h}` |")
    a("")
    a("## 정의")
    a("| 항목 | 정의 |")
    a("|---|---|")
    a("| 정답 | 두 라벨러가 모두 '예'인 단위(불일치는 정답이 아니며 따로 셌다) |")
    a("| 묶음 정밀도 | 둘 다 예 ÷ 표본 단위(60개, CONV_BEAR 53개). 단어를 빼기 **전** 표본으로 판정한다. 괄호는 Wilson 95% 구간(참고) |")
    a("| 상한 | 한쪽 이상 예 ÷ 표본 단위(참고) |")
    a("| κ | 두 라벨러 in_sense의 Cohen κ. 한쪽 범주만 나오면 '정의 안 됨' |")
    a("| 반어율 | 한쪽 이상이 반어로 본 단위 ÷ 표본 단위(더 엄한 쪽으로 판정). 둘 다 반어도 함께 적었다 |")
    a("| 단어 정밀도 | 적중 하나씩(같은 단위 안 같은 단어도 따로) 둘 다 예 ÷ 라벨된 적중. 규칙으로 옮겨 온 적중(CAP→E·HES)은 뺐다 |")
    a("| 단어 빼기 | 라벨된 적중 10회 이상이고 단어 정밀도 50% 미만(§2.7). 공백만 다른 같은 표현은 함께 뺀다. 새 단어는 더하지 않았다 |")
    a("| 뺀 뒤 추정 | 남은 단어 적중이 하나라도 있는 표본 단위에서, 남은 적중 중 하나라도 둘 다 예인 비율. 같은 표본으로 고른 뒤 다시 잰 값이라 낙관 편향이 있다(판정에 쓰지 않음) |")
    a("| 측정 실패 | 묶음 정밀도 < 60% 또는 반어율 > 30% |")
    a("")
    a("## 묶음별 결과")
    a("| 묶음 | n | 둘 다 예 | 불일치(1예·2아니오 / 1아니오·2예) | 정밀도 (95%) | 상한 | κ | 일치율 | 반어(한쪽 / 둘 다) | 반어율 | 뺀 단어 | 같은 틀 적중 단위 후보 → v0.2 | 뺀 뒤 추정(남은 단위) | 판정 | 쓰는 가설 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st in strata:
        r = B[st]
        lo, hi = r["prec_ci"]
        vs = stats["verify"]["strata"][st]
        share_txt = f"{vs['units_cand']:,} → {vs['units_v02']:,} ({pct(1 - vs['units_v02'] / vs['units_cand']) if vs['units_cand'] else '—'} 줄어듦)"
        verdict = ("**측정 실패**: " + r["fail_reason"]) if r["fail"] else "통과"
        if r["kappa_low"]:
            verdict += " (κ < 0.6)"
        a(f"| {st} | {r['n']} | {r['both_yes']} | {r['dis_1y2n']} / {r['dis_1n2y']} | {pct(r['precision'])} ({pct(lo)}~{pct(hi)}) | "
          f"{pct(r['upper'])} | {fk(r['kappa'])} | {pct(r['agree'])} | {r['irony_any']} / {r['irony_both']} | {pct(r['irony_rate_any'])} | "
          f"{r['n_removed']} | {share_txt} | {pct(r['prec_after'])} ({r['kept_units']}) | {verdict} | {USES.get(st, '')} |")
    a("")
    fails = [st for st in strata if B[st]["fail"]]
    a(f"- 측정 실패 {len(fails)}개: {', '.join(fails) if fails else '없음'}.")
    a(f"- 통과 {len(strata) - len(fails)}개.")
    a(f"- κ < 0.6(참고 표시, 사전 판정 기준 아님): {', '.join(st for st in strata if B[st]['kappa_low']) or '없음'}.")
    a("")
    a("## 뺀 단어")
    a("| 묶음 | 단어 | 라벨된 적중 | 둘 다 예 | 단어 정밀도 | 한쪽 이상 예 | 모집단 적중 | 단위로 묶은 판도 빼짐 | 한쪽 이상 예 기준도 빼짐 |")
    a("|---|---|---|---|---|---|---|---|---|")
    for w in W:
        if w["drop"]:
            a(f"| {w['stratum']} | {w['word']} | {w['n']} | {w['both']} | {pct(w['precision'])} | {pct(w['either'] / w['n'])} | "
              f"{w['pop_hits']} | {'예' if w['drop_unit_dedup'] else '아니오'} | {'예' if w['drop_either'] else '아니오'} |")
    a("")
    a(f"- 합계 {sum(1 for w in W if w['drop'])}개.")
    a("")
    a("## 남긴 단어 중 적중 10회 이상")
    a("| 묶음 | 단어 | 라벨된 적중 | 둘 다 예 | 단어 정밀도 | 모집단 적중 |")
    a("|---|---|---|---|---|---|")
    for w in W:
        if (not w["drop"]) and w["n"] >= MIN_WORD_HITS:
            a(f"| {w['stratum']} | {w['word']} | {w['n']} | {w['both']} | {pct(w['precision'])} | {w['pop_hits']} |")
    a("")
    diff = [w for w in W if w["drop"] != w["drop_unit_dedup"] or (w["drop"] != w["drop_either"])]
    a("## 판정이 바뀌는 민감도")
    if diff:
        a("| 묶음 | 단어 | 주 판정 | 단위로 묶은 판 | 한쪽 이상 예 기준 | 적중 / 단위 | 정밀도(적중 / 단위) |")
        a("|---|---|---|---|---|---|---|")
        for w in diff:
            a(f"| {w['stratum']} | {w['word']} | {'뺌' if w['drop'] else '남김'} | {'뺌' if w['drop_unit_dedup'] else '남김'} | "
              f"{'뺌' if w['drop_either'] else '남김'} | {w['n']} / {w['n_units']} | {pct(w['precision'])} / {pct(w['both_units'] / w['n_units'])} |")
    else:
        a("- 없음")
    a("")
    a("## 규칙으로 옮겨 온 적중(단어 정밀도에서 뺌)")
    a("| 층 | 규칙 | 단어 | 둘 다 예 | 수 |")
    a("|---|---|---|---|---|")
    for m in stats["moved"]:
        a(f"| {m['stratum']} | {m['note']} | {m['word']} | {'예' if m['both_yes'] else '아니오'} | {m['n']} |")
    a("")
    a("## BULL/BEAR 정확도")
    a("사전 방향 = 글 점수 부호(clip(강세 − 약세, ±3), 제목 2배, 부정이면 뒤집음). 맞음 = 두 라벨러의 글 극성이 모두 사전 방향과 같음(불일치는 따로 셈).")
    a("")
    a("| 표본 | 사전 판 | n | 사전 점수 ≠ 0 | 맞음(둘 다) | 정확도(점수 ≠ 0, 95%) | 라벨 일치 단위만 | 불일치 | 정반대 | 3범주 정확도(전체, 둘 다) | 라벨 일치 단위만(전체) | 재현율(합의 강세·약세 글 중 사전이 같은 부호) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, blk in (("score200", sc["cand"]), ("score200", sc["v02"]),
                     ("정밀도 BULL", stats["precision_bullbear"]["BULL"]), ("정밀도 BEAR", stats["precision_bullbear"]["BEAR"])):
        ver = "v0.2" if blk is sc["v02"] else "후보"
        lo, hi = blk["nz_acc_ci"]
        a(f"| {lab} | {ver} | {blk['n']} | {blk['n_nonzero']} | {blk['nz_strict']} | {pct(blk['nz_acc_strict'])} ({pct(lo)}~{pct(hi)}) | "
          f"{pct(blk['nz_acc_agreed'])} ({blk['nz_agreed_ok']}/{blk['nz_agreed']}) | {blk['nz_disagree']} | {blk['nz_opposite']} | "
          f"{pct(blk['all_acc_strict'])} | {pct(blk['all_acc_agreed'])} ({blk['all_agreed']}) | {pct(blk['recall'])} ({blk['gold_directional']}) |")
    a("")
    a(f"- score200 라벨러 극성 κ {fk(sc['labeler_kappa'])}, 일치율 {pct(sc['labeler_agree'])}. 분포 1: {sc['labeler_dist']['1']}, 2: {sc['labeler_dist']['2']}.")
    pb = stats["precision_bullbear"]
    a(f"- 정밀도 표본 극성 κ: BULL {fk(pb['BULL']['labeler_kappa'])}, BEAR {fk(pb['BEAR']['labeler_kappa'])}.")
    a(f"- v0.2에서 방향이 바뀐 score200 글: {sc['changed_by_v02']}개. 저장된 사전 점수와 다시 계산한 후보 점수가 다른 글: {sc['recompute_mismatch']}개.")
    a(f"- H06 판정(score200, v0.2, 점수 ≠ 0 글의 정확도 60% 기준): **{'측정 실패' if stats['h06_fail'] else '통과'}**. 후보 판으로는 {'측정 실패' if stats['h06_fail_cand'] else '통과'}.")
    a("")
    v = stats["verify"]
    a("## v0.2 매처 점검(가림 표본과 같은 틀, 가격 없음)")
    a(f"- 틀 단위 {v['frame_units']:,}개(급락 4일·열어 본 8일 제외, 2026-07-13~10-11). 후보 매처 지문 `{v['candidate_sha']}`.")
    a(f"- 후보판 단어 적중 수가 blind_summary와 같음: {'예' if v['pop_hits_match_summary'] else '아니오'}.")
    a(f"- v0.2 적중 ≠ 후보 적중 − 뺀 단어 적중인 단위: {v['v02_not_equal_cand_minus_removed']}개, v0.2에서 새로 생긴 적중: {v['v02_new_hits']}개, 글 점수 불일치: {v['v02_score_mismatch']}개.")
    a("- 뺀 단어는 자리를 차지한 채 버려진다(`lexicon_v02_final.py`). 예: '레버리지'를 빼도 그 안의 '레버'가 다시 잡지 않는다. 막지 않고 목록에서만 지우면 '레버'가 '레버리지' 자리를 잡아 빼기가 무효가 된다.")
    a("")
    a("## 라벨 형식 점검")
    a(f"- 적중 2개 이상 단위의 '단위 판정 = 적중 판정의 OR' 확인: {stats['consistency']}.")
    a("")
    a("## v0.2 단어 수")
    a("| 묶음 | 후보 | v0.2 |")
    a("|---|---|---|")
    for st in strata:
        a(f"| {st} | {cand_counts[st]} | {len(v02_map[st])} |")
    (RES / "lexicon_v02_precision.md").write_bytes(("\n".join(L) + "\n").encode("utf-8"))


if __name__ == "__main__":
    sys.exit(main())
