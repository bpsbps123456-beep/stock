"""Display every candidate title and record an explicit human review batch.

Suggestions organize the reading queue only. Nothing is decided until the
reviewer has read the displayed titles and confirms or overrides the batch.
"""
import argparse
import csv
import json
from pathlib import Path
import re

NEWS = Path(__file__).resolve().parents[2]
STATE = NEWS / "v2/title_review_state.json"
REASONS = {"single_stock", "not_a_change", "pb_flow", "other_asset", "other_region", "duplicate", "unrelated", "out_of_window"}


def rows():
    with (NEWS / "pending_review_B1.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def suggestion(title):
    t = title.lower()
    broad = bool(re.search(r"s&p|s＆p|snp|nasdaq|나스닥|주식|증시|반도체|기술주|semiconductor|equities|stocks|tech sector|growth sector|u\.s\. equity|us equity|global equity|market outlook|market strategy|asset allocation|자산배분", t))
    if re.search(r"prime brokerage|prime broker|hedge funds? (sold|sell|buy|bought|cut|add)|hedge-fund (selling|buying)|헤지펀드.*(매도|매수|청산)|자금흐름", t):
        return "pb_flow"
    if re.search(r"\bgold\b|bitcoin|\bcrypto\b|treasur|\bbond|\boil\b|원유|금값|비트코인|채권|환율", t) and not broad and not re.search(r"stock|share", t):
        return "other_asset"
    if broad:
        return "not_a_change"
    if re.search(r"price target|target price|(?:raises|cuts|lifts|hikes).*(?:target|\bpt\b)|upgrade|downgrade|overweight|underweight|neutral|rating|상향|하향|목표가|목표주가|투자의견|비중확대|비중축소|중립", t):
        return "single_stock"
    return "unrelated"


def state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"batches": [], "decisions": {}}


def display(start, end):
    data = rows()
    for i in range(start, min(end, len(data)) + 1):
        row = data[i-1]
        print(f"{i:04d} [{suggestion(row['title'])}] {row['title']}")


def confirm(start, end, overrides):
    data, current = rows(), state()
    overrides = json.loads(overrides)
    for i in range(start, min(end, len(data)) + 1):
        row = data[i-1]
        reason = overrides.get(str(i), suggestion(row["title"]))
        decision = "include" if reason == "include" else "exclude"
        if reason != "include" and reason not in REASONS:
            raise ValueError(reason)
        current["decisions"][row["candidate_id"]] = dict(
            candidate_id=row["candidate_id"], decision=decision,
            reason_code="" if decision == "include" else reason,
            note=f"제목 직접 판정 batch {start}-{end}; " + ("의견 변경 또는 확인할 광역 자산 의견으로 본문 검증 대상" if decision == "include" else row["title"]),
            title=row["title"], review_index=i)
    current["batches"].append(dict(start=start, end=end, method="all_displayed_titles_read_and_suggestion_confirmed_or_overridden", overrides=overrides))
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (NEWS / "review_B1_v2.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "decision", "reason_code", "note", "title", "review_index"])
        writer.writeheader()
        writer.writerows(sorted(current["decisions"].values(), key=lambda r:r["review_index"]))
    print("Explicitly reviewed", len(current["decisions"]), "of", len(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["display", "confirm"])
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--overrides", default="{}")
    args = parser.parse_args()
    display(args.start, args.end) if args.command == "display" else confirm(args.start, args.end, args.overrides)
