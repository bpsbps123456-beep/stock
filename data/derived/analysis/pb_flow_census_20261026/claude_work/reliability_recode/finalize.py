"""Write reliability.csv and summary stats from compare_rows.json plus manual cause labels."""
import collections
import csv
import glob
import json
import math
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\geunt\OneDrive\Desktop\주식"
W = os.path.join(ROOT, r"data\derived\analysis\pb_flow_census_20261026\claude_work")
R = os.path.join(ROOT, r"research\pb_flow_census_20261026")
FIELDS = ["bank", "flow_direction", "long_flow", "short_flow", "family", "lookback_months", "note_date"]
rows = json.load(open(os.path.join(HERE, "compare_rows.json"), encoding="utf-8"))

RULE, INFO = "rule_interpretation", "information_gap"
CAUSES = {
    ("2843b81e72e4#0", "flow_direction"): (RULE, "Codex는 숏 커버 주도 순매수를 흐름 공란으로 둠; 원문 'Net buying ... largest level since October last year'는 명시적 순매수라 규칙상 net_buy"),
    ("2843b81e72e4#0", "lookback_months"): (RULE, "같은 이유로 Codex가 순매수 lookback(2022-10 이후, 9개월)을 비움; 매크로 숏 커버 'since November 2020'(32개월) 선택 여지도 있음"),
    ("2843b81e72e4#1", "note_date"): (RULE, "원문 'note dated July 13'인데 Codex가 note_date를 비워 공개일 대용(7/14)이 들어감"),
    ("ad98d9050c54#0", "family"): (RULE, "에너지 비중이 2020-05 저점 근처: 재부호화는 S(극단 숏·저노출), Codex는 '낮은 보유 비중은 극단성 근거 아님'으로 N"),
    ("ad98d9050c54#0", "lookback_months"): (RULE, "'approaching a May 2020 low'를 lookback(40개월)으로 볼지 여부"),
    ("89c0590362e8#1", "family"): (RULE, "지배 사실 선택: Codex는 매도 흐름(N), 재부호화는 2018년 이후 기록 근처 노출(T)"),
    ("cd7a6d5b86e5#0", "note_date"): (RULE, "'report released on Wednesday'를 8/16으로 환산할지; Codex는 비워 공개일 대용"),
    ("1c0b45b1f49a#0", "note_date"): (RULE, "월간 노트: Codex는 대상 월말(10/31)을 명시 날짜로, 재부호화는 공개일 대용(11/1)"),
    ("0200d6b3c0ac#0", "long_flow"): (RULE, "'long positions in IT reached the largest in eight months'(보유 수준)를 롱 매수로 볼지"),
    ("7eee58bd5f5f#0", "flow_direction"): (INFO, "Codex는 발췌만 보고 '순방향 미확정'; 재부호화는 article_reads 본문의 'generally net sellers of global stocks' 사용(범위도 IT 대 세계로 다름)"),
    ("7eee58bd5f5f#0", "note_date"): (RULE, "프로토콜이 대상 주 종료일(11/16)과 발행일(11/17)을 둘 다 허용"),
    ("e3dc68ad1bf8#0", "note_date"): (INFO, "같은 기사에 GS 보고서 두 개(금 7/28, 월 7/31); 숏 청산 사실이 어느 보고서인지 원문이 모호"),
    ("6ebf9522baab#0", "family"): (RULE, "다지역 노트의 지배 사실: 기존은 유럽 5년 최대 매도(D), 재부호화는 미국 재편입(R)"),
    ("cd3b6e80c7a4#1", "flow_direction"): (INFO, "기사 속 'separate note'(GS 미국 노트)를 재부호화가 놓쳐 세계 노트 하나만 만듦; 기존 #1은 미국 별도 노트"),
    ("cd3b6e80c7a4#1", "long_flow"): (INFO, "위와 같음(단위 차이)"),
    ("cd3b6e80c7a4#1", "family"): (INFO, "위와 같음(단위 차이)"),
    ("44c26ac783a2#0", "family"): (INFO, "기존은 같은 그룹 기사 본문의 'leverage now stands at a five-year high'를 봄(T); 재부호화 입력은 5월 매수 속도만(R)"),
    ("44c26ac783a2#0", "lookback_months"): (INFO, "위와 같음: 레버리지 5년 최고(60) 대 매수 2025-06 이후 최빠름(12)"),
    ("44c26ac783a2#0", "note_date"): (RULE, "월간 노트: 기존은 대상 월말(5/31), 재부호화는 공개일 대용(6/5)"),
    ("2da96dc2e812#0", "family"): (RULE, "둘 다 전사문 열람; 순노출 3·5년 2~3백분위를 기존은 '상당한 디레버리징' D, 재부호화는 극단 저노출 S"),
    ("e7bac6103b58#0", "bank"): (INFO, "재부호화 입력은 제목뿐(은행명 없음, Google News 링크 열람 실패); 기존은 247wallst 원 주소를 열람해 GS 확인"),
    ("e7bac6103b58#0", "flow_direction"): (INFO, "위와 같음: 본문의 숏 매도:롱 매수 5:1을 재부호화는 못 봄"),
    ("e7bac6103b58#0", "long_flow"): (INFO, "위와 같음"),
    ("e7bac6103b58#0", "lookback_months"): (INFO, "위와 같음: 'near five-year high'(60)"),
    ("881ee06554a6#0", "long_flow"): (RULE, "개별주 순매수(롱 매수·숏 커버 주도)를 북미 행에 넣을지; 기존은 '지역 구분 없는 사실이라 na' 선례"),
    ("881ee06554a6#0", "short_flow"): (RULE, "위와 같음"),
    ("5d371ca00cc3#0", "note_date"): (RULE, "'Through Thursday'를 1/15로 환산할지; 재부호화는 6ebf9522baab에서는 환산해 자기 일관성도 없음"),
}
INFO_LEVEL = {
    "code_screen_items": "snippet_or_title", "code_reading_items": "evidence_text", "reading_queue": "evidence_text",
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


csv_rows = []
for r in rows:
    dis = [f for f in FIELDS if not r[f + "_agree"]]
    causes = [CAUSES[(r["key"], f)] for f in dis]
    lvl = INFO_LEVEL[r["input_file"]]
    if r["info"].startswith("title only"):
        lvl = "title_only"
    d = {
        "sample_no": r["sample_no"], "existing_coder": r["coder"], "row_key": r["key"], "id": r["id"],
        "title": r["title"], "input_file": r["input_file"], "recode_input_level": lvl,
        "recode_webfetch": r["webfetch"], "match_method": r["match"],
        "existing_rows_same_id": r["n_existing_rows_same_id"], "recode_notes_same_id": r["n_recode_notes"],
        "recode_variant": r["recode_variant"], "recode_confidence": r["recode_confidence"],
    }
    for f in FIELDS:
        fmt = lambda v: int(v) if isinstance(v, float) and v.is_integer() else v
        d[f + "_existing"] = fmt(r[f + "_existing"])
        d[f + "_recode"] = fmt(r[f + "_recode"])
        d[f + "_agree"] = r[f + "_agree"]
    d["lookback_exact"] = r["lookback_months_exact"]
    d["note_date_within3d"] = r["note_date_within3d"]
    d["existing_note_date_basis"] = r["existing_basis"]
    d["recode_note_date_basis"] = r["recode_basis"]
    d["fields_agree"] = sum(r[f + "_agree"] for f in FIELDS)
    d["disagree_fields"] = ";".join(dis)
    d["disagree_causes"] = ";".join(f"{f}={c[0]}" for f, c in zip(dis, causes))
    d["cause_detail_ko"] = " | ".join(f"{f}: {c[1]}" for f, c in zip(dis, causes))
    io = {f: r[f + "_recode_input_only"] for f in FIELDS if f + "_recode_input_only" in r}
    d["recode_input_only_values"] = ";".join(f"{k}={v}" for k, v in io.items())
    csv_rows.append(d)

out_csv = os.path.join(R, "reliability.csv")
with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
    w.writeheader()
    w.writerows(csv_rows)

stats = {}
for grp in ["all", "claude", "codex"]:
    sub = [r for r in rows if grp == "all" or r["coder"] == grp]
    g = {"n": len(sub)}
    for f in FIELDS:
        k = sum(r[f + "_agree"] for r in sub)
        lo, hi = wilson(k, len(sub))
        e = {"agree": k, "rate": round(k / len(sub), 3), "ci95": [round(lo, 2), round(hi, 2)]}
        if f not in ("lookback_months", "note_date"):
            e["kappa"] = round(kappa([(r[f + "_existing"], r[f + "_recode"]) for r in sub]), 2)
        if f == "lookback_months":
            e["exact"] = sum(r["lookback_months_exact"] for r in sub)
            both = [r for r in sub if r["lookback_months_existing"] != "" and r["lookback_months_recode"] != ""]
            e["both_numeric"] = len(both)
            e["presence_agree"] = sum((r["lookback_months_existing"] == "") == (r["lookback_months_recode"] == "") for r in sub)
        if f == "note_date":
            e["within3d"] = sum(r["note_date_within3d"] for r in sub)
            e["same_week"] = sum(1 for r in sub if r["note_date_existing"] and r["note_date_recode"] and
                                 date.fromisoformat(r["note_date_existing"]).isocalendar()[:2] == date.fromisoformat(r["note_date_recode"]).isocalendar()[:2])
        g[f] = e
    g["all7_agree_rows"] = sum(all(r[f + "_agree"] for f in FIELDS) for r in sub)
    stats[grp] = g

# bank agreement excluding rows matched on bank
nb = [r for r in rows if r["match"] != "matched_on_bank"]
stats["bank_excl_matched"] = {"n": len(nb), "agree": sum(r["bank_agree"] for r in nb)}
# input-only counterfactual (no WebFetch)
cf = {}
for f in FIELDS:
    k = 0
    for r in rows:
        if f + "_recode_input_only" in r:
            a, b = r[f + "_existing"], r[f + "_recode_input_only"]
            if f == "lookback_months":
                ok = (a == "" and b == "") or (a != "" and b != "" and abs(float(a) - float(b)) <= 1)
            else:
                ok = a == b
            k += int(ok)
        else:
            k += r[f + "_agree"]
    cf[f] = k
stats["input_only_counterfactual_agree"] = cf
# cause tally
tally = collections.Counter()
by_field = collections.defaultdict(collections.Counter)
for (key, f), (c, _) in CAUSES.items():
    tally[c] += 1
    by_field[f][c] += 1
stats["cause_tally"] = dict(tally)
stats["cause_by_field"] = {f: dict(v) for f, v in by_field.items()}
# by input level
lv = collections.defaultdict(lambda: [0, 0])
for d in csv_rows:
    lv[d["recode_input_level"]][0] += d["fields_agree"]
    lv[d["recode_input_level"]][1] += len(FIELDS)
stats["field_agree_by_input_level"] = {k: f"{a}/{b}" for k, (a, b) in lv.items()}
stats["rows_by_input_level"] = dict(collections.Counter(d["recode_input_level"] for d in csv_rows))
stats["family_confusion"] = dict(collections.Counter(f"{r['family_existing']}->{r['family_recode']}" for r in rows))
stats["flow_confusion"] = dict(collections.Counter(f"{r['flow_direction_existing'] or 'blank'}->{r['flow_direction_recode'] or 'blank'}" for r in rows))

# supplementary: existing Claude vs existing Codex on ids coded by both (not blind; no recode)
claude_by = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(W, "coded_*.json"))):
    for x in json.load(open(f, encoding="utf-8"))["rows"]:
        if x["eligible"] == "yes":
            claude_by[x["id"]].append(x)
codex_by = collections.defaultdict(list)
for x in json.load(open(os.path.join(R, "reviewed_notes.json"), encoding="utf-8")):
    codex_by[x["source_id"]].append(x)
sup = []
for sid in sorted(set(claude_by) & set(codex_by)):
    for cx in codex_by[sid]:
        cands = [c for c in claude_by[sid] if c["bank"] == cx["bank"]]
        if not cands:
            sup.append({"id": sid, "bank_codex": cx["bank"], "matched": False})
            continue
        c = cands[0]
        e = {"id": sid, "bank": cx["bank"], "matched": True}
        for f in ["flow_direction", "long_flow", "short_flow", "family", "lookback_months"]:
            a = c.get(f) if c.get(f) not in (None, "", "na") else ""
            b = cx.get(f) if cx.get(f) not in (None, "", "na") else ""
            if f == "lookback_months":
                a = "" if a == "" else round(float(a), 2)
                b = "" if b == "" else round(float(b), 2)
            e[f] = int(a == b)
        sup.append(e)
m = [e for e in sup if e["matched"]]
stats["supplement_claude_vs_codex_existing"] = {
    "codex_rows_on_shared_ids": len(sup), "matched_by_bank": len(m),
    **{f: sum(e[f] for e in m) for f in ["flow_direction", "long_flow", "short_flow", "family", "lookback_months"]}}
json.dump(stats, open(os.path.join(HERE, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
print(out_csv, len(csv_rows))
