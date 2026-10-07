"""The stop/review rule shown in the desk's risk panel (docs/desk_system.md §5).

Fixed on 2026-09-27 before any live trade, so that a drawdown does not get to move it.
Evidence: switching to the TIME ETF after lagging it cut the 7-stock rule's 2023-2025
return from +245% to +113% (20 sessions, -10%p, KRX closes aligned to the US day they
price). Lagging alone therefore never stops the plan; only leaving the range the rule was
validated on changes what the user does.
"""

from __future__ import annotations

# Desk model 2026-01-02..09-25 (zero cost, open fills, close-marked KRW, ETF aligned):
# worst drawdown and worst 20-session gap to the TIME ETF. Frozen — do not refit after a
# bad stretch; `py -m desk validate` checks they still describe the model.
BACKTEST_2026 = {"mdd": -0.303, "gap20": -0.081}
DD_MARGIN = 0.05   # out of range once 5%p deeper than the 2026 worst
SHORT_LAG = -0.10  # 20-session gap that gets the "keep going" note
LONG_WINDOW = 120  # sessions (about six months) behind the ETF before re-evaluating


def describe() -> list[str]:
    """The rule in words, shown under the alerts even when nothing fires."""
    return [f"최근 20거래일 수익률이 TIME ETF보다 {-SHORT_LAG:.0%}p 이상 낮아도 매매 규칙을 유지합니다. "
            f"(2026년 최대 격차 {BACKTEST_2026['gap20'] * 100:+.1f}%p, 2023~2025년에도 자주 나타난 구간)",
            f"내 계좌 또는 모델의 고점 대비 하락률이 {BACKTEST_2026['mdd'] - DD_MARGIN:.1%} 이하가 되면 "
            f"(2026년 최대 하락률 {BACKTEST_2026['mdd']:.1%}보다 5%p 더 하락) 규칙대로 매매하되 추가 입금은 보류하고 투자 금액을 다시 점검합니다.",
            f"최근 {LONG_WINDOW}거래일 수익률이 TIME ETF보다 낮으면 이 전략을 계속 사용할지 재평가합니다."]


def my_pause_rule(ctx: dict) -> list[dict]:
    """멈춤·점검 기준. 성과 부진만으로는 멈추지 않고, 검증 범위를 벗어날 때만 행동을 바꾼다.

    ctx 키: account_drawdown(입출금 보정, 기록이 없으면 None), model_drawdown,
    model_vs_etf_20d, model_vs_etf_120d(이력이 짧으면 None)
    """
    out = []
    dd, who = min((d, w) for d, w in ((ctx["account_drawdown"], "내 계좌"), (ctx["model_drawdown"], "모델"))
                  if d is not None)
    if dd <= BACKTEST_2026["mdd"] - DD_MARGIN:
        out.append({"level": "serious", "title": f"{who} 고점 대비 {dd:.1%} — 추가 입금 보류 기준 도달",
                    "body": f"2026년 검증 기간의 최대 하락률({BACKTEST_2026['mdd']:.1%})보다 5%p 이상 더 하락했습니다. "
                            "검증한 위험 범위를 벗어났으므로 매매 규칙은 유지하되 추가 입금은 보류하세요. 감당할 수 있는 손실을 기준으로 투자 금액을 다시 점검하세요."})
    long_gap = ctx["model_vs_etf_120d"]
    if long_gap is not None and long_gap < 0:
        out.append({"level": "serious", "title": f"최근 {LONG_WINDOW}거래일 TIME ETF 대비 수익률 격차 {long_gap * 100:+.1f}%p",
                    "body": "최근 약 반년의 수익률이 TIME ETF보다 낮습니다. 이 전략을 계속 사용할지 재평가하세요. "
                            "2026년 검증 기간에는 없었던 구간이며, 2023~2025년에는 전체 거래일의 약 40%에서 나타났습니다."})
    elif ctx["model_vs_etf_20d"] is not None and ctx["model_vs_etf_20d"] <= SHORT_LAG:
        out.append({"level": "warning",
                    "title": f"최근 20거래일 TIME ETF 대비 {ctx['model_vs_etf_20d'] * 100:+.1f}%p — 매매 규칙 유지",
                    "body": "2023~25년에는 이런 구간이 자주 있었습니다. 그때마다 ETF로 옮겼다면 수익이 +245%에서 +113%로 "
                            "줄었습니다. 현재 매매 규칙을 유지합니다."})
    return out
