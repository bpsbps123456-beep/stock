"""실적 추세 연구 — 보도자료 정독 결과(워크플로 journal) → results/extract/<티커>.json + 품질 점검.

실행: py data/derived/analysis/earnings_trend/code/parse_extract.py <journal.jsonl> [<journal2.jsonl> ...]

- extract:<T> 에이전트 결과에 adjudicate:<T> 판정(두 독립 추출이 다른 숫자)을 덮어쓴다.
- 인용문은 원문 파일에 그대로 있는지 기계로 확인한다(공백·따옴표·대소문자만 정규화).
  없는 인용은 qc에 남기고, 그 인용에 기댄 정성 판단(촉매·쇼스타퍼·내러티브)은 features.py가 비운다.
- 매출은 XBRL 분기값과 1% 넘게 다르면 표시한다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sec import ROOT, RESULTS  # noqa: E402

QUOTE_FIELDS = [("tone_quote",), ("catalyst", "quote"), ("showstopper", "quote"), ("narrative_quote",),
                ("next_q_guidance", "quote"), ("fy_guidance", "quote"), ("profit_guidance", "quote"),
                ("backlog", "quote"), ("prior_guided_actual", "quote")]


def norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = s.replace("|", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def load_journal(paths: list[str]) -> dict[str, list]:
    """label → 결과 객체. 같은 label이 여러 번이면 처음 것을 쓴다
    (이어받기가 끝난 에이전트를 다시 돌린 경우가 있어, 추출·검증·판정이 서로 맞는 첫 실행을 정본으로 둔다)."""
    out: dict[str, object] = {}
    for p in paths:
        labels: dict[str, str] = {}  # 'started' 줄에만 label이 있고 'result' 줄은 key로 잇는다
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("type") == "started" and rec.get("label"):
                labels[rec["key"]] = rec["label"]
                continue
            if rec.get("type") != "result":
                continue
            res = rec.get("result")
            if isinstance(res, str):
                try:
                    res = json.loads(res)
                except json.JSONDecodeError:
                    pass
            label = labels.get(rec.get("key"))
            if label and res is not None and label not in out:
                out[label] = res
    return out


def get(d: dict, path: tuple):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def apply_fix(ev: dict, fix: dict) -> None:
    try:
        val = json.loads(fix["value_json"])
    except (json.JSONDecodeError, TypeError):
        val = fix["value_json"]
    field = fix["field"]
    if field == "missing_in_B":
        return
    if "." in field:
        a, b = field.split(".", 1)
        if isinstance(ev.get(a), dict):
            ev[a][b] = val
        return
    if field in ("next_q_guidance", "fy_guidance") and isinstance(val, dict) and isinstance(ev.get(field), dict):
        merged = dict(ev[field])
        merged.update(val)
        val = merged
    ev[field] = val


def main() -> None:
    journals = sys.argv[1:]
    res = load_journal(journals)
    sec = pd.read_csv(RESULTS / "sec_events.csv", dtype=str).set_index("accession")
    q = pd.read_csv(RESULTS / "quarters.csv")
    outdir = RESULTS / "extract"
    outdir.mkdir(parents=True, exist_ok=True)
    by_ticker: dict[str, dict] = {}
    for label, obj in res.items():
        if not label.startswith("extract:") or not isinstance(obj, dict):
            continue
        key = label.split(":", 1)[1]
        t = obj.get("ticker") or re.sub(r"[AB]$", "", key)
        fixes = (res.get(f"adjudicate:{key}") or {}).get("fixes", []) if isinstance(res.get(f"adjudicate:{key}"), dict) else []
        evs = {e["accession"]: e for e in obj["events"]}
        for f in fixes:
            if f.get("accession") in evs:
                apply_fix(evs[f["accession"]], f)
        rec = by_ticker.setdefault(t, {"ticker": t, "events": {}, "fixes": [], "qc": {"bad_quotes": [], "revenue_mismatch": []}})
        rec["events"].update(evs)
        rec["fixes"].extend(fixes)
    total_q, bad_q = 0, 0
    for t, rec in by_ticker.items():
        texts = {}
        for acc, ev in rec["events"].items():
            if acc not in sec.index:
                continue
            files = sec.loc[acc, "files"].split(";")
            texts[acc] = norm(" ".join((ROOT / f).read_text(encoding="utf-8") for f in files))
            for path in QUOTE_FIELDS:
                qt = get(ev, path)
                if not qt:
                    continue
                total_q += 1
                if norm(qt) not in texts[acc]:
                    bad_q += 1
                    rec["qc"]["bad_quotes"].append({"accession": acc, "field": ".".join(path), "quote": qt[:120]})
            # 매출 대조(XBRL 분기값, 분기말 ±7일)
            pe = ev.get("period_end") or ""
            if pe and ev.get("revenue_musd"):
                qt_ = q[(q["ticker"] == t)]
                d = (pd.to_datetime(qt_["period_end"]) - pd.Timestamp(pe)).abs().dt.days
                m = qt_[d <= 7]
                if len(m) and pd.notna(m["revenue"].iloc[0]):
                    x = m["revenue"].iloc[0] / 1e6
                    if abs(x / ev["revenue_musd"] - 1) > 0.01:
                        rec["qc"]["revenue_mismatch"].append({"accession": acc, "pr": ev["revenue_musd"], "xbrl": round(x, 1)})
        rec["events"] = sorted(rec["events"].values(), key=lambda e: e.get("filing_date", ""))
        (outdir / f"{t}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    isq = [{"accession": e["accession"], "ticker": t, "is_q": bool(e.get("is_quarterly_results"))}
           for t, r in by_ticker.items() for e in r["events"]]
    pd.DataFrame(isq).to_csv(RESULTS / "is_quarterly.csv", index=False)  # events.py가 핵심 표본 거르기에 쓴다
    rows = [{"ticker": t, "events": len(r["events"]), "fixes": len(r["fixes"]), "bad_quotes": len(r["qc"]["bad_quotes"]),
             "revenue_mismatch": len(r["qc"]["revenue_mismatch"])} for t, r in sorted(by_ticker.items())]
    qc = pd.DataFrame(rows)
    qc.to_csv(RESULTS / "extract_qc.csv", index=False)
    print(qc.to_string(index=False))
    print(f"인용 {total_q}개 중 원문 불일치 {bad_q}개 ({bad_q / max(total_q, 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
