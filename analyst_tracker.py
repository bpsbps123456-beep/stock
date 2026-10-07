"""애널리스트 콜 로그 채점기 (Path-A 정본: docs/analyst_system.md)

사용법:
  py analyst_tracker.py verify     # scoreable+day 콜의 5d/20d·MFE/MAE·놓친 상승 채점
  py analyst_tracker.py scorecard  # 인물별 성적표 출력
"""
import argparse
import glob
import json
import os
import sys
from datetime import date

import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "data", "analyst_log")
BENCHMARK = "^GSPC"

# ── 판정 튜닝 포인트 (사용자 조정 영역) ──────────────────────────────
# 20거래일 판정 기준 수익률의 절대값이 이 값(%) 미만이면 hit/miss 대신 '중립'.
HIT_DEADBAND_PCT = 0.0
# 약세 콜이 방향을 맞혔더라도 놓친 상승이 이 값(%) 이상이면 나쁜 콜 플래그.
BEARISH_MISSED_UPSIDE_FLAG_PCT = 10.0
# ────────────────────────────────────────────────────────────────

ANALYST_NAMES = {"gabriel": "테런스 가브리엘", "wilson": "마이크 윌슨", "catone": "유제니오 카토네"}


def load_calls():
    calls = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            calls.append((path, json.load(f)))
    return calls


def is_scoreable(call):
    return bool(call.get("scoreable")) and call.get("date_precision") == "day"


def fetch_closes(tickers, start):
    df = yf.download(sorted(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(sorted(tickers)[0])
    return df


def window_return(series, entry_idx, n, entry_close):
    exit_idx = entry_idx + n
    if exit_idx >= len(series):
        return None
    return series.iloc[exit_idx] / entry_close - 1.0


def score_call(call, closes):
    target = call["target"]
    if target not in closes.columns:
        return None
    s = closes[target].dropna()
    pub = pd.Timestamp(call["published"])
    idx = s.index.searchsorted(pub)
    if idx >= len(s):
        return None
    entry_close = float(s.iloc[idx])
    bearish = call["direction"] == "bearish"

    bench = closes[BENCHMARK].dropna()
    b_idx = bench.index.searchsorted(pub)
    b_entry = float(bench.iloc[b_idx]) if b_idx < len(bench) else None
    use_bench = target != BENCHMARK and b_entry is not None

    v = {
        "status": "pending",
        "entry_date": s.index[idx].strftime("%Y-%m-%d"),
        "entry_close": round(entry_close, 2),
        "d5": None,
        "d20": None,
        "mfe_pct": None,
        "mae_pct": None,
        "missed_upside_pct": None,
        "hit_20d": None,
        "flag": None,
        "verified_at": date.today().isoformat(),
    }

    for n in (5, 20):
        ret = window_return(s, idx, n, entry_close)
        if ret is None:
            continue
        block = {"target_pct": round(ret * 100, 2)}
        if use_bench:
            b_ret = window_return(bench, b_idx, n, b_entry)
            if b_ret is not None:
                block["bench_pct"] = round(b_ret * 100, 2)
                block["excess_pct"] = round((ret - b_ret) * 100, 2)
        v["d5" if n == 5 else "d20"] = block

    seg = s.iloc[idx + 1: idx + 21]
    if len(seg):
        max_up = float(seg.max()) / entry_close - 1.0
        max_dn = float(seg.min()) / entry_close - 1.0
        v["mfe_pct"] = round((-max_dn if bearish else max_up) * 100, 2)
        v["mae_pct"] = round((max_up if bearish else -max_dn) * 100, 2)
        if bearish:
            v["missed_upside_pct"] = round(max_up * 100, 2)

    if v["d20"]:
        basis = v["d20"].get("excess_pct") if call.get("relative") else v["d20"]["target_pct"]
        if basis is None:
            pass
        elif abs(basis) < HIT_DEADBAND_PCT:
            v["hit_20d"] = "neutral"
        else:
            favorable = basis < 0 if bearish else basis > 0
            v["hit_20d"] = "hit" if favorable else "miss"
        if bearish and v["missed_upside_pct"] is not None \
                and v["missed_upside_pct"] >= BEARISH_MISSED_UPSIDE_FLAG_PCT:
            v["flag"] = f"놓친 상승 {v['missed_upside_pct']}% ≥ {BEARISH_MISSED_UPSIDE_FLAG_PCT}%"
        v["status"] = "complete"
    elif v["d5"] or v["mfe_pct"] is not None:
        v["status"] = "partial"

    return v


def cmd_verify():
    calls = load_calls()
    todo = [(p, c) for p, c in calls
            if is_scoreable(c) and c.get("verification", {}).get("status") != "complete"]
    if not todo:
        print("채점 대기 콜 없음.")
        return
    tickers = {c["target"] for _, c in todo} | {BENCHMARK}
    start = min(pd.Timestamp(c["published"]) for _, c in todo) - pd.Timedelta(days=10)
    closes = fetch_closes(tickers, start.strftime("%Y-%m-%d"))
    missing = [t for t in tickers if t not in closes.columns or closes[t].dropna().empty]
    if missing:
        print(f"경고: 가격 없음 → {missing}")
    updated = 0
    for path, call in todo:
        v = score_call(call, closes)
        if v is None:
            print(f"  건너뜀(가격 없음): {os.path.basename(path)}")
            continue
        call["verification"] = v
        with open(path, "w", encoding="utf-8") as f:
            json.dump(call, f, ensure_ascii=False, indent=2)
        updated += 1
        print(f"  {os.path.basename(path)}: {v['status']} (hit_20d={v['hit_20d']})")
    print(f"채점 갱신 {updated}건 / 대기 {len(todo)}건")


def fmt_window(call, block):
    if not block:
        return "—"
    if call.get("relative") and "excess_pct" in block:
        return f"{block['excess_pct']:+.2f}(초과)"
    return f"{block['target_pct']:+.2f}"


def cmd_scorecard():
    calls = [c for _, c in load_calls()]
    if not calls:
        print("기록된 콜 없음.")
        return
    for src, label in (("user_provided_summary", "참고 베이스라인(백필)"), ("websearch", "실시간 수집")):
        rows = [c for c in calls if c.get("source") == src]
        if not rows:
            continue
        print(f"\n## {label} — {len(rows)}건\n")
        print("| 인물 | 발표일 | 대상 | 방향 | 5d | 20d | 적중 | 놓친상승 | 플래그 |")
        print("|---|---|---|---|---:|---:|---|---:|---|")
        for c in rows:
            v = c.get("verification") or {}
            mu = v.get("missed_upside_pct")
            print("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                ANALYST_NAMES.get(c["analyst"], c["analyst"]), c["published"],
                c["target"], c["direction"],
                fmt_window(c, v.get("d5")), fmt_window(c, v.get("d20")),
                v.get("hit_20d") or "—",
                f"{mu:+.2f}" if mu is not None else "—",
                v.get("flag") or "—"))
        print()
        for key, name in ANALYST_NAMES.items():
            sub = [c for c in rows if c["analyst"] == key]
            if not sub:
                continue
            hits = sum(1 for c in sub if (c.get("verification") or {}).get("hit_20d") == "hit")
            misses = sum(1 for c in sub if (c.get("verification") or {}).get("hit_20d") == "miss")
            judged = hits + misses
            rate = f"{hits}/{judged} ({hits / judged * 100:.0f}%)" if judged else "채점분 없음"
            print(f"- {name}: 콜 {len(sub)}건, 20d 방향 적중 {rate}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["verify", "scorecard"])
    args = ap.parse_args()
    if args.command == "verify":
        cmd_verify()
    else:
        cmd_scorecard()


if __name__ == "__main__":
    main()
