"""Evaluate continuous, aligned daily strategy ledgers. Rates are stored as fractions."""

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PERIODS = {
    "full": ("2026-01-01", "2026-12-31"),
    "h1": ("2026-01-01", "2026-06-30"),
    "h2": ("2026-07-01", "2026-12-31"),
    "jul_aug": ("2026-07-01", "2026-08-31"),
    "sep_latest": ("2026-09-01", "2026-12-31"),
}


def stats(nav, initial):
    values = np.r_[initial, np.asarray(nav, dtype=float)]
    return values[-1] / initial - 1, np.min(values / np.maximum.accumulate(values) - 1)


def summarize(curve):
    result = {}
    for period, (start, end) in PERIODS.items():
        ix = np.flatnonzero((curve.index >= start) & (curve.index <= end))
        if not len(ix):
            raise ValueError(f"Missing sessions in {period}")
        window = curve.iloc[ix]
        for column, suffix in [("krw", ""), ("krw10", "_10bp"), ("usd", "_usd")]:
            before = curve[column].iloc[ix[0] - 1] if ix[0] else 1.0
            ret, dd = stats(window[column], before)
            result[f"{period}_return{suffix}"] = ret
            result[f"{period}_mdd{suffix}"] = dd
        result[f"{period}_sessions"] = len(window)
        result[f"{period}_exposure"] = window.exposure.mean()
        result[f"{period}_turn"] = window.turn.sum()
    return result


def choose(eligible, fallback):
    if eligible.empty:
        return fallback
    tied = eligible.h1_return >= eligible.h1_return.max() - 1e-12
    return eligible.index[tied].min()


def matched_control(base, target_exposure):
    """Hindsight diagnostic, daily close rebalance between the base in USD and USD cash."""
    window = base.loc["2026-07-01":]
    normal = window.exposure.mean()
    if normal <= 0 or target_exposure > normal + 1e-9:
        return np.nan, np.nan, np.nan
    fraction = min(target_exposure / normal, 1.0)
    daily_returns = base.usd.pct_change().loc[window.index]
    nav = (1 + fraction * daily_returns).cumprod()
    return fraction, *stats(nav, 1.0)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                     "|" + "|".join("---" for _ in headers) + "|",
                     *["| " + " | ".join(row) + " |" for row in rows]])


def pct(value):
    return "해당 없음" if pd.isna(value) else f"{value * 100:+.2f}%"


def evaluate(curves, configs):
    """Write report, full grid and selected ledgers, and return the selected key(s)."""
    if set(curves) != set(configs):
        raise ValueError("Curve and configuration keys differ")
    base = curves["base"]
    for key, curve in curves.items():
        if not (curve.index.is_monotonic_increasing and curve.index.is_unique
                and curve.index.equals(base.index)):
            raise ValueError(f"Misaligned sessions for {key}")
        if not np.isfinite(curve[["krw", "krw10", "usd", "turn", "exposure"]]).all().all():
            raise ValueError(f"Incomplete ledger for {key}")
        if (curve[["krw", "krw10", "usd"]] <= 0).any().any():
            raise ValueError(f"Non-positive NAV for {key}")

    frame = pd.DataFrame.from_dict({key: {**configs[key], **summarize(curves[key])}
                                    for key in sorted(curves)}, orient="index")
    frame.index.name = "strategy"
    ref = frame.loc["base"]
    frame["h1_risk_eligible"] = frame.h1_mdd >= ref.h1_mdd - 1e-9
    candidates = frame.drop(["base", "half"])
    eligible = candidates.loc[candidates.h1_risk_eligible]
    pool = frame.loc[frame.index.isin([*eligible.index, "base"])]
    global_key = choose(pool, "base")
    retain = pool.loc[(pool.h1_return >= 0.75 * ref.h1_return - 1e-12) | (pool.index == "base")]
    shallow = retain.loc[retain.h1_mdd >= retain.h1_mdd.max() - 1e-12]
    stability_key = choose(shallow, "base")
    by_family = {family: choose(subset.loc[subset.h1_risk_eligible], "base")
                 for family, subset in candidates.groupby("family", sort=True)}
    diagnostic_best = candidates.index[candidates.h2_return >= candidates.h2_return.max() - 1e-12].min()
    winners = (candidates.h2_return > ref.h2_return + 1e-9) & (candidates.h2_mdd > ref.h2_mdd + 1e-9)
    winners10 = ((candidates.h2_return_10bp > ref.h2_return_10bp + 1e-9)
                 & (candidates.h2_mdd_10bp > ref.h2_mdd_10bp + 1e-9))
    frame["h2_matched_fraction"] = np.nan
    frame["h2_matched_return_usd"] = np.nan
    frame["h2_matched_mdd_usd"] = np.nan
    matched_columns = ["h2_matched_fraction", "h2_matched_return_usd", "h2_matched_mdd_usd"]
    for key, row in frame.iterrows():
        frame.loc[key, matched_columns] = matched_control(base, row.h2_exposure)
    frame["global_selection"] = frame.index == global_key
    frame["stability_selection"] = frame.index == stability_key
    frame["family_selection"] = frame.index.isin(by_family.values())
    frame.to_csv(HERE / "candidates.csv")
    selected = list(dict.fromkeys(["base", "half", global_key, stability_key, *by_family.values(), diagnostic_best]))
    pd.concat({key: curves[key] for key in selected}, names=["strategy", "date"]).to_csv(
        HERE / "selected_daily.csv")

    def results_row(label, key):
        row = frame.loc[key]
        return [label, key, pct(row.h1_return), pct(row.h2_return), pct(row.h2_mdd),
                pct(row.full_return), pct(row.full_mdd)]

    listed = [("현행", "base"), ("고정 절반 대조", "half"), ("상반기 수익 우선", global_key),
              ("상반기 안정성 우선", stability_key),
              *[(f"{family} 상반기 선택", key) for family, key in by_family.items()],
              ("7월 이후 수익 사후 1위", diagnostic_best)]
    rows = [results_row(label, key) for label, key in listed]
    lines = [
        "# 7종목 계좌의 시장 이탈·종목 교체 실험",
        "",
        f"미국 거래일 {base.index[0]}~{base.index[-1]}의 {len(base)}거래일을 연속 운용했다. "
        f"현행·고정 절반 대조 2개와 후보 {len(candidates)}개를 비교했다. "
        "상반기에 최대낙폭이 현행보다 깊지 않은 후보 중 원화 수익률이 가장 높은 규칙을 선택했다. "
        "안정성 우선 선택에서는 같은 낙폭 조건에 상반기 현행 이익의 75%를 지키는 조건을 더하고 "
        "그중 낙폭이 가장 얕은 규칙을 골랐다. 낙폭 동률이면 수익이 높은 쪽을 택했다. "
        "두 전체 선택에는 현행도 포함했다. 유형별로 자격을 충족한 후보가 없으면 현행을 표시했다. "
        "수익이 같으면 규칙 이름 오름차순으로 골랐다. 고정 절반은 대조용이다. "
        "각 가격선·재진입·교체 조건은 [실험 방법](methods.md)에서 확인할 수 있다.",
        "",
        f"현행은 상반기 {pct(ref.h1_return)}, 7월 이후 {pct(ref.h2_return)}, "
        f"연초부터 누적 {pct(ref.full_return)}였다. 수익 우선 규칙 **{global_key}**은 "
        f"7월 이후 {pct(frame.loc[global_key, 'h2_return'])}, "
        f"같은 구간 최대낙폭 {pct(frame.loc[global_key, 'h2_mdd'])}였다.",
        "",
        "## 구간 성과",
        "",
        table(["선정 근거", "규칙", "상반기", "7월 이후", "7월 이후 최대낙폭", "연초 누적",
               "연초 최대낙폭"], rows),
        "",
        f"상반기 낙폭 조건에 든 후보는 {len(eligible)}/{len(candidates)}개다. "
        f"7월 이후 현행 수익과 낙폭을 모두 개선한 후보는 {int(winners.sum())}/{len(candidates)}개이고, "
        f"거래대금 편도 10bp 비용을 현행과 후보에 함께 적용하면 {int(winners10.sum())}개다. "
        "여러 설정이 같은 날짜와 종목을 공유한다. 후보 수를 독립적인 성공 횟수로 해석할 수 없다. "
        "하반기 사후 1위는 전체 탐색의 상단을 보여 준다.",
        "",
        "## 약세 구간과 거래 비용",
        "",
        table(["규칙", "7~8월", "9월 이후", "7월 이후 투자 비중 평균", "7월 이후 매수·매도 합계¹",
               "7월 이후 10bp 수익", "7월 이후 10bp 최대낙폭"],
              [[key, pct(frame.loc[key, "jul_aug_return"]), pct(frame.loc[key, "sep_latest_return"]),
                f"{100*frame.loc[key, 'h2_exposure']:.1f}%", f"{frame.loc[key, 'h2_turn']:.1f}배",
                pct(frame.loc[key, "h2_return_10bp"]), pct(frame.loc[key, "h2_mdd_10bp"])]
               for key in selected]),
        "",
        "¹ 매일 매수 비중과 매도 비중의 절댓값 합을 누적했다. "
        "수수료 0인 기본 조건과 비교하기 위해 거래대금 편도 0.10%를 적용했다. "
        "표의 구간 수익과 낙폭에는 구간 직전 종가를 포함하며 연초에만 초기 자산 1을 쓴다.",
        "",
        "## 같은 금액을 계속 투자했을 때",
        "",
        "하반기 각 후보의 평균 투자 비중을 현행의 평균 투자 비중으로 나눈 비율만큼 매일 종가에 "
        "현행 모형을 보유하고 나머지를 달러 현금으로 두는 대조다. 하반기를 보고 정한 비율이므로 "
        "현금 보유 자체로 설명되는 방어 효과를 가늠하는 사후 계산이다. 대조의 거래 비용은 0이며, "
        "비율이 100%를 넘으면 대조 수익은 비워 둔다.",
        "",
        table(["규칙", "후보 달러 수익", "비중 맞춘 대조 달러 수익", "후보 달러 최대낙폭",
               "대조 달러 최대낙폭"], [[key, pct(frame.loc[key, "h2_return_usd"]),
                pct(frame.loc[key, "h2_matched_return_usd"]), pct(frame.loc[key, "h2_mdd_usd"]),
                pct(frame.loc[key, "h2_matched_mdd_usd"])] for key in selected]),
        "",
        "2026년 가격은 기존 연구에서 여러 번 사용했고, 7월 이후 약세를 알고 설계한 실험이다. "
        "상반기만으로 선택해 뒤쪽 성과를 분리했어도 독립적인 실시간 검증은 앞으로 쌓일 거래일에서 가능하다. "
        "각 규칙의 설정과 모든 구간 수익·낙폭은 [전체 후보](candidates.csv), "
        "선택 규칙과 대조군의 일별 자산·매매량은 [일별 원장](selected_daily.csv)에 담았다.",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"global": global_key, "stability": stability_key, "by_family": by_family, "candidate_count": len(candidates),
            "h2_best": diagnostic_best, "h2_double_wins": int(winners.sum())}
