"""두 펀드(TIME 456600, KoAct 2ETFQ1)의 실제 매매 주기 측정.

질문: 우리가 따라가는 펀드매니저는 하루·주·월 중 어느 단위로 종목을 바꾸나?
비중을 빼는 방식은 한 번에 정리인가, 며칠에 걸친 점진 축소인가?

입력: data/normalized/etf_holdings/*.json (CU당 수량 PDF, 2026-01-02 ~ 2026-10-08)
매매 판정: gbrc.etf.flows.quantity_flows 를 인접 두 날짜에 적용(설정·환매 공통 배율과
액면분할을 제거한 '선별 수량 변화'). ±5% 이상이면 매수/매도, 새로 생기면 신규, 사라지면 편출.
이 정의는 저장소 정본(docs/etf_system.md 실매매 기준)과 같다.

출력: results_fund_trades.json (같은 폴더)
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from gbrc.etf.flows import quantity_flows  # noqa: E402

SNAP_DIR = ROOT / "data" / "normalized" / "etf_holdings"
OUT = Path(__file__).with_name("results_fund_trades.json")
FUNDS = {"456600": "TIME 글로벌AI인공지능액티브", "2ETFQ1": "KoAct 미국나스닥성장기업액티브"}
WINDOW = 20  # 편출 전·신규 후 관찰 거래일 수
BIG_W = 1.0  # 편출 분석은 직전 20일 중 최대 비중이 1% 이상이던 종목만


def load(fund: str) -> list[dict]:
    rows = []
    for path in sorted(SNAP_DIR.glob(f"*_{fund}.json")):
        snap = json.loads(path.read_text(encoding="utf-8"))
        if (snap.get("quality") or {}).get("status") != "valid":
            continue
        rows.append(snap)
    rows.sort(key=lambda s: s["as_of_date"])
    return rows


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def analyse(fund: str) -> dict:
    snaps = load(fund)
    dates = [s["as_of_date"] for s in snaps]
    days = []  # 인접 쌍별 결과
    per_key = defaultdict(dict)  # key -> {pair_idx: flow}
    for i in range(1, len(snaps)):
        res = quantity_flows([snaps[i - 1], snaps[i]])
        buy_pp = sell_pp = 0.0
        kinds = defaultdict(int)
        for key, fl in res.flows.items():
            per_key[key][i] = fl
            kinds[fl.kind] += 1
            if fl.kind in ("매수", "신규"):
                buy_pp += abs(fl.trade_pp or 0.0)
            elif fl.kind in ("매도", "편출"):
                sell_pp += abs(fl.trade_pp or 0.0)
        days.append({
            "date": dates[i],
            "buy_pp": buy_pp,
            "sell_pp": sell_pp,
            "one_way_pp": (buy_pp + sell_pp) / 2,
            "n_buy": kinds["매수"], "n_sell": kinds["매도"],
            "n_new": kinds["신규"], "n_exit": kinds["편출"],
            "n_trade": kinds["매수"] + kinds["매도"] + kinds["신규"] + kinds["편출"],
        })

    n = len(days)
    one_way = [d["one_way_pp"] for d in days]
    trade_days = [d for d in days if d["n_trade"] > 0]
    inout_days = [d for d in days if d["n_new"] + d["n_exit"] > 0]

    # 월말 집중도: 각 달의 마지막 2개 표본일
    by_month = defaultdict(list)
    for d in days:
        by_month[d["date"][:7]].append(d)
    month_end = []
    for m, ds in by_month.items():
        month_end += ds[-2:]
    me_share_turn = sum(d["one_way_pp"] for d in month_end) / max(sum(one_way), 1e-12)
    me_share_days = len(month_end) / n
    q_end = [d for d in month_end if d["date"][5:7] in ("03", "06", "09")]
    qe_share_turn = sum(d["one_way_pp"] for d in q_end) / max(sum(one_way), 1e-12)
    qe_share_days = len(q_end) / n

    # 주 단위: 신규·편출이 한 건이라도 있는 주의 비율
    import datetime as dt
    weeks = defaultdict(int)
    for d in days:
        y, w, _ = dt.date.fromisoformat(d["date"]).isocalendar()
        weeks[(y, w)] += d["n_new"] + d["n_exit"]
    weeks_with_inout = sum(1 for v in weeks.values() if v > 0) / len(weeks)

    # 요일 분포(매매일 비율)
    wd = defaultdict(lambda: [0, 0])
    for d in days:
        k = dt.date.fromisoformat(d["date"]).weekday()
        wd[k][1] += 1
        wd[k][0] += 1 if d["n_trade"] > 0 else 0

    # 편출 직전 20거래일: 이미 얼마나 줄여 놨나
    exits = []
    for key, fmap in per_key.items():
        for i, fl in fmap.items():
            if fl.kind != "편출":
                continue
            factor, sells, wmax, span = 1.0, 0, fl.prev_weight, 0
            j = i - 1
            while j >= 1 and span < WINDOW:
                f = fmap.get(j)
                if f is None or f.status != "held" or f.selective_pct is None:
                    break
                factor *= 1 + f.selective_pct / 100
                sells += 1 if f.kind == "매도" else 0
                wmax = max(wmax, f.weight, f.prev_weight)
                span += 1
                j -= 1
            if wmax < BIG_W or span < 5:
                continue
            exits.append({
                "key": key, "name": fl.name, "exit_date": dates[i], "span": span,
                "w_max_prior": round(wmax, 3), "w_last": round(fl.prev_weight, 3),
                "pre_sold_frac": round(max(0.0, 1 - factor), 3), "sell_days": sells,
            })

    # 신규 후 20거래일: 첫날에 최종 수량의 몇 %를 샀나
    entries = []
    for key, fmap in per_key.items():
        for i, fl in fmap.items():
            if fl.kind != "신규" or i == 1:
                continue
            factor, buys, span = 1.0, 0, 0
            j = i + 1
            while j <= n and span < WINDOW:
                f = fmap.get(j)
                if f is None or f.status != "held" or f.selective_pct is None:
                    break
                factor *= 1 + f.selective_pct / 100
                buys += 1 if f.kind == "매수" else 0
                span += 1
                j += 1
            if span < WINDOW:
                continue
            w_end = fmap[i + span].weight
            if w_end < BIG_W:
                continue
            entries.append({
                "key": key, "name": fl.name, "entry_date": dates[i],
                "w_first": round(fl.weight, 3), "w_after20": round(w_end, 3),
                "first_day_frac": round(min(1.0, 1 / factor) if factor > 0 else 1.0, 3),
                "buy_days": buys,
            })

    # 보유 구간 길이(표본 안에서 시작·종료가 모두 관측된 구간)
    spells, censored = [], 0
    for key, fmap in per_key.items():
        start = None
        present0 = 0 in fmap or (1 in fmap and fmap[1].status in ("held", "exited"))
        if present0:
            start = 0
        for i in range(1, n + 1):
            fl = fmap.get(i)
            if fl and fl.kind == "신규":
                start = i
            if fl and fl.kind == "편출" and start is not None:
                if start > 0:
                    spells.append(i - start)
                else:
                    censored += 1
                start = None
        if start is not None:
            censored += 1

    pre = [e["pre_sold_frac"] for e in exits]
    first = [e["first_day_frac"] for e in entries]
    return {
        "fund": fund, "name": FUNDS[fund],
        "sample": {"first": dates[0], "last": dates[-1], "pairs": n},
        "trade_day_share": round(len(trade_days) / n, 3),
        "inout_day_share": round(len(inout_days) / n, 3),
        "weeks_with_inout_share": round(weeks_with_inout, 3),
        "median_names_traded_per_day": st.median(d["n_trade"] for d in days),
        "mean_daily_one_way_turnover_pp": round(st.mean(one_way), 3),
        "median_daily_one_way_turnover_pp": round(st.median(one_way), 3),
        "annualized_one_way_turnover_pct": round(st.mean(one_way) * 252, 1),
        "top5_days_turnover_share": round(sum(sorted(one_way)[-5:]) / sum(one_way), 3),
        "month_end_last2": {"turnover_share": round(me_share_turn, 3), "day_share": round(me_share_days, 3)},
        "quarter_end_last2_mar_jun_sep": {"turnover_share": round(qe_share_turn, 3), "day_share": round(qe_share_days, 3)},
        "weekday_trade_day_share": {k: round(v[0] / v[1], 3) for k, v in sorted(wd.items())},
        "exits_big": {
            "n": len(exits),
            "median_pre_sold_frac": pct(pre, 0.5),
            "share_pre_sold_ge_50pct": round(sum(p >= 0.5 for p in pre) / len(pre), 3) if pre else None,
            "share_pre_sold_lt_10pct": round(sum(p < 0.1 for p in pre) / len(pre), 3) if pre else None,
            "median_sell_days_prior20": pct([e["sell_days"] for e in exits], 0.5),
            "median_w_max_prior": pct([e["w_max_prior"] for e in exits], 0.5),
            "median_w_last": pct([e["w_last"] for e in exits], 0.5),
            "by_size": {
                label: {
                    "n": len(sub),
                    "median_pre_sold_frac": pct([e["pre_sold_frac"] for e in sub], 0.5),
                    "share_pre_sold_ge_50pct": round(sum(e["pre_sold_frac"] >= 0.5 for e in sub) / len(sub), 3) if sub else None,
                    "median_sell_days_prior20": pct([e["sell_days"] for e in sub], 0.5),
                }
                for label, sub in (
                    ("wmax_1to3", [e for e in exits if e["w_max_prior"] < 3]),
                    ("wmax_ge3", [e for e in exits if e["w_max_prior"] >= 3]),
                )
            },
            "events": sorted(exits, key=lambda e: e["exit_date"]),
        },
        "entries_big": {
            "n": len(entries),
            "median_first_day_frac": pct(first, 0.5),
            "share_first_day_ge_80pct": round(sum(f >= 0.8 for f in first) / len(first), 3) if first else None,
            "median_buy_days_next20": pct([e["buy_days"] for e in entries], 0.5),
            "events": sorted(entries, key=lambda e: e["entry_date"]),
        },
        "holding_spells_closed": {
            "n": len(spells), "censored": censored,
            "median_days": pct(spells, 0.5), "p25": pct(spells, 0.25), "p75": pct(spells, 0.75),
        },
        "top_turnover_days": sorted(days, key=lambda d: -d["one_way_pp"])[:5],
    }


def main() -> None:
    out = {
        "source": "data/normalized/etf_holdings (CU당 수량 PDF), gbrc.etf.flows.quantity_flows ±5% 기준",
        "window_days": WINDOW, "big_weight_pct": BIG_W,
        "funds": [analyse(f) for f in FUNDS],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for f in out["funds"]:
        brief = {k: v for k, v in f.items() if k not in ("exits_big", "entries_big", "top_turnover_days")}
        print(json.dumps(brief, ensure_ascii=False))
        ex = {k: v for k, v in f["exits_big"].items() if k not in ("events",)}
        en = {k: v for k, v in f["entries_big"].items() if k != "events"}
        print("EXITS", json.dumps(ex, ensure_ascii=False))
        print("ENTRIES", json.dumps(en, ensure_ascii=False))
        print("TOPDAYS", [(d["date"], round(d["one_way_pp"], 2), d["n_new"], d["n_exit"]) for d in f["top_turnover_days"]])


if __name__ == "__main__":
    main()
