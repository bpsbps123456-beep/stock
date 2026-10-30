"""H0 눈가림 재구현 (BRIEF3 H0, V1 = 'L 최저' 기준만).

lib.py·h0.py·test_lib.py를 쓰지 않고 Alpaca 1분봉 parquet를 직접 읽는다.

규칙(BRIEF3 H0 + 오케스트레이터 체결 규칙):
- L(종목별) = 09:30 봉(또는 앞 매매 청산 다음 봉)부터 지금 봉까지 저가 최저.
- 신호: 봉 t 종가 >= L*1.03. SOXL 신호 -> SOXS 매수, SOXS 신호 -> SOXL 매수.
  같은 봉에 둘 다 뜨면 더 많이 오른 쪽(종가/L 큰 쪽)의 반대를 산다(건수 기록).
- 결정 봉 시작 <= 10:44(w90) / 09:44(std).
- 매수가 = 봉 t+1 종가 * (1 + 반 호가[산 종목, t+1 시작 시각 구간]).
- 목표: t+2 봉부터 창 끝 봉까지 고가 >= 매수가*1.01*1.0002 이면 매수가*1.01에 매도(반 호가 없음).
- 못 닿으면 창 끝 봉(w90 10:59, std 09:59) 종가 * (1 - 반 호가)로 매도.
- 들고 있는 동안 신호 무시. 청산 봉 다음 봉부터 두 종목 L을 다시 잰다.
- 하루 수익 = 그날 매매 (1+r) 곱 - 1. 손익 % 단위(1.0 = 1%).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/sox_open_switch"
CACHE = os.path.join(ROOT, "cache", "alp1m")
RES = os.path.join(ROOT, "results")

RISE = 1.03
TGT = 1.01
PIERCE = 1.0002

SCHEMES = {
    # 창 끝 봉 시작(분), 마지막 결정 봉 시작(분) — 자정 기준 분
    "w90": {"end_bar": 10 * 60 + 59, "last_decide": 10 * 60 + 44},
    "std": {"end_bar": 9 * 60 + 59, "last_decide": 9 * 60 + 44},
}
OPEN_MIN = 9 * 60 + 30


def bucket(tod):
    if tod < 570:
        return "pre"
    if tod < 575:
        return "open5"
    if tod < 600:
        return "early"
    return "late"


def load():
    cost = json.load(open(os.path.join(ROOT, "cost_table.json"), encoding="utf-8"))
    out = {}
    for s in ("SOXL", "SOXS"):
        df = pd.read_parquet(os.path.join(CACHE, f"{s}.parquet"))
        tod = df.index.hour * 60 + df.index.minute
        df = df[(tod >= OPEN_MIN) & (tod < 11 * 60)].copy()
        df["tod"] = df.index.hour * 60 + df.index.minute
        df["date"] = df.index.date
        out[s] = df
    return out, cost


def run_day(bars, cost, scheme, stats):
    """bars: {sym: DataFrame(그날 09:30~10:59)}. 반환: 매매 목록."""
    end_bar = SCHEMES[scheme]["end_bar"]
    last_decide = SCHEMES[scheme]["last_decide"]
    # 두 종목 모두 있는 분만 쓴다(빈 분은 건너뛰고 센다)
    common = bars["SOXL"].index.intersection(bars["SOXS"].index)
    for s in ("SOXL", "SOXS"):
        stats["skipped_min"] += len(bars[s].index) - len(common)
    tods = (common.hour * 60 + common.minute).to_numpy()
    keep = tods <= end_bar
    common = common[keep]
    tods = tods[keep]
    n = len(common)
    if n == 0:
        return []
    H = {s: bars[s].loc[common, "High"].to_numpy(float) for s in bars}
    Lo = {s: bars[s].loc[common, "Low"].to_numpy(float) for s in bars}
    C = {s: bars[s].loc[common, "Close"].to_numpy(float) for s in bars}
    # 창 끝 봉 위치: end_bar 시작 봉(없으면 그 전 마지막 봉)
    end_idx = n - 1

    trades = []
    i = 0
    Lmin = {"SOXL": np.inf, "SOXS": np.inf}
    while i < n:
        if tods[i] > last_decide:
            break
        for s in ("SOXL", "SOXS"):
            Lmin[s] = min(Lmin[s], Lo[s][i])
        sigL = C["SOXL"][i] >= Lmin["SOXL"] * RISE
        sigS = C["SOXS"][i] >= Lmin["SOXS"] * RISE
        if not (sigL or sigS):
            i += 1
            continue
        if sigL and sigS:
            stats["both_signal"] += 1
            rL = C["SOXL"][i] / Lmin["SOXL"]
            rS = C["SOXS"][i] / Lmin["SOXS"]
            riser = "SOXL" if rL >= rS else "SOXS"
        else:
            riser = "SOXL" if sigL else "SOXS"
        buy = "SOXS" if riser == "SOXL" else "SOXL"
        f = i + 1  # 체결 봉
        if f > end_idx:
            break
        hs_in = cost[buy][bucket(tods[f])]
        px = C[buy][f] * (1 + hs_in)
        tgt_px = px * TGT
        exit_idx = None
        ret = None
        hit = False
        mae = 0.0
        for j in range(f + 1, end_idx + 1):
            mae = min(mae, Lo[buy][j] / px - 1)
            if H[buy][j] >= tgt_px * PIERCE:
                exit_idx = j
                ret = TGT - 1
                hit = True
                break
        if exit_idx is None:
            exit_idx = end_idx
            hs_out = cost[buy][bucket(tods[end_idx])]
            ret = C[buy][end_idx] * (1 - hs_out) / px - 1
        trades.append({
            "riser": riser,
            "buy": buy,
            "sig_t": common[i],
            "fill_t": common[f],
            "exit_t": common[exit_idx],
            "rise_pct": (C[riser][i] / Lmin[riser] - 1) * 100,
            "entry_px": px,
            "hit": hit,
            "ret_pct": ret * 100,
            "mae_pct": mae * 100,
            "min_to_hit": (exit_idx - f) if hit else np.nan,
        })
        # 청산 다음 봉부터 L 다시 재기
        i = exit_idx + 1
        Lmin = {"SOXL": np.inf, "SOXS": np.inf}
    return trades


def boot_ci(x, n=10000, seed=20261029):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    data, cost = load()
    allday = sorted(set(data["SOXL"]["date"]) & set(data["SOXS"]["date"]))
    days = [d for d in allday
            if (pd.Timestamp("2025-01-01").date() <= d <= pd.Timestamp("2025-12-31").date())
            or (pd.Timestamp("2026-01-02").date() <= d <= pd.Timestamp("2026-10-28").date())]
    g = {s: dict(tuple(data[s].groupby("date"))) for s in data}
    summary = {}
    for scheme in SCHEMES:
        stats = {"skipped_min": 0, "both_signal": 0}
        rows, all_tr = [], []
        for d in days:
            bars = {s: g[s].get(d, data[s].iloc[0:0]) for s in data}
            tr = run_day(bars, cost, scheme, stats)
            for t in tr:
                t["date"] = d
            all_tr += tr
            dr = (np.prod([1 + t["ret_pct"] / 100 for t in tr]) - 1) * 100 if tr else 0.0
            rows.append({"date": d, "period": "2025" if d.year == 2025 else "2026",
                         "n_trades": len(tr), "n_hit": sum(t["hit"] for t in tr),
                         "ret_pct": dr})
        daily = pd.DataFrame(rows)
        daily.to_csv(os.path.join(RES, f"h0_blind_daily_{scheme}.csv"), index=False)
        trades = pd.DataFrame(all_tr)
        trades.to_csv(os.path.join(RES, f"h0_blind_trades_{scheme}.csv"), index=False)
        out = {"skipped_min": stats["skipped_min"], "both_signal": stats["both_signal"]}
        segs = {
            "2026": ("2026-01-02", "2026-10-28"),
            "2026H1": ("2026-01-02", "2026-06-30"),
            "2026H2": ("2026-07-01", "2026-10-28"),
            "2025": ("2025-01-01", "2025-12-31"),
        }
        for k, (a, b) in segs.items():
            a, b = pd.Timestamp(a).date(), pd.Timestamp(b).date()
            dd = daily[(daily.date >= a) & (daily.date <= b)]
            tt = trades[(trades.date >= a) & (trades.date <= b)] if len(trades) else trades
            lo, hi = boot_ci(dd.ret_pct.to_numpy())
            out[k] = {
                "days": int(len(dd)),
                "active_days": int((dd.n_trades > 0).sum()),
                "mean_daily_pct": float(dd.ret_pct.mean()),
                "ci95": [lo, hi],
                "n_trades": int(len(tt)),
                "hit_rate": float(tt.hit.mean()) if len(tt) else float("nan"),
                "mean_trade_pct": float(tt.ret_pct.mean()) if len(tt) else float("nan"),
                "miss_mean_pct": float(tt.loc[~tt.hit, "ret_pct"].mean()) if len(tt) else float("nan"),
                "buy_SOXS": int((tt.buy == "SOXS").sum()) if len(tt) else 0,
                "buy_SOXL": int((tt.buy == "SOXL").sum()) if len(tt) else 0,
                "median_min_to_hit": float(tt.min_to_hit.median()) if len(tt) else float("nan"),
                "mean_mae_pct": float(tt.mae_pct.mean()) if len(tt) else float("nan"),
            }
        summary[scheme] = out
    with open(os.path.join(RES, "h0_blind_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
