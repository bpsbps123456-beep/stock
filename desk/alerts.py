"""The stop/review rule shown in the desk's risk panel (docs/desk_system.md §5).

Fixed on 2026-09-27 before any live trade, so that a drawdown does not get to move it.
Evidence: switching to the TIME ETF after lagging it cut the 7-stock rule's 2023-2025
return from +245% to +113% (20 sessions, -10%p, KRX closes aligned to the US day they
price). Lagging alone therefore never stops the plan; only leaving the range the rule was
validated on changes what the user does.
"""

from __future__ import annotations

# #774 model (semiconductor budget 60%, config.CODEX_774_RULE) 2026-01-02..09-25 (zero cost,
# open fills, close-marked KRW, ETF aligned): worst drawdown and worst 20-session gap to the
# TIME ETF. Frozen — do not refit after a bad stretch or a rule change; `py -m desk validate`
# checks they still describe the #774 replay. The live 50% rule (2026-10-10) is reported there.
BACKTEST_2026 = {"mdd": -0.303, "gap20": -0.081}
BASIS = "#774·반도체 60% 기준"
# §3.1 switch path (SOXX 100-day line -> month-start FINVIZ basket) on the 2026-11-21 weekly rule
# (60-session residual), KRW, 10bp one way, ETF aligned to the KRX close that prices each US day:
# worst drawdown 2026-09-28 and worst 20-session gap to the TIME ETF (ending 2026-05-07), through
# 2026-11-19. Frozen 2026-11-21 by user decision ("기본값 유지 + 멈춤 기준 고정") before living
# through a drawdown on this path; `py -m desk validate` re-derives them from the research ledger
# data/derived/analysis/desk_etf_validation_20261121/results/paired_daily.csv.
BACKTEST_SWITCH_2026 = {"mdd": -0.282, "gap20": -0.110}
SWITCH_BASIS = "§3.1 전환·60일 잔차·원화 편도 10bp 기준, 2026-11-21 고정"
DD_MARGIN = 0.05   # out of range once 5%p deeper than the 2026 worst
SHORT_LAG = -0.10  # 20-session gap that gets the "keep going" note
LONG_WINDOW = 120  # sessions (about six months) behind the ETF before re-evaluating


def describe(switch_applied: bool = False) -> list[str]:
    """The rule in words, shown under the alerts even when nothing fires."""
    lines = [f"최근 20거래일 수익률이 TIME ETF보다 {-SHORT_LAG:.0%}p 이상 낮아도 매매 규칙을 유지합니다. "
            f"(2026년 최대 격차 {BACKTEST_2026['gap20'] * 100:+.1f}%p({BASIS}), 2023~2025년에도 자주 나타난 구간)",
            f"내 계좌 또는 7종목 전략(모의)의 고점 대비 하락률이 {BACKTEST_2026['mdd'] - DD_MARGIN:.1%} 이하가 되면 "
            f"(2026년 최대 하락률 {BACKTEST_2026['mdd']:.1%}({BASIS}, 2026-09-27 고정)보다 5%p 더 하락) "
            "규칙대로 매매하되 추가 입금은 보류하고 투자 금액을 다시 점검합니다.",
            f"최근 {LONG_WINDOW}거래일 수익률이 TIME ETF보다 낮으면 이 전략을 계속 사용할지 재평가합니다."]
    if switch_applied:
        sw = BACKTEST_SWITCH_2026
        lines.append(f"주문 기준이 SOXX 100일선 전환(§3.1)일 때 내 계좌는 전환 경로 기준으로 봅니다. 고점 대비 하락률 "
                     f"{sw['mdd'] - DD_MARGIN:.1%} 이하(2026년 최대 {sw['mdd']:.1%}보다 5%p 더 하락)면 추가 입금을 보류하고 "
                     f"투자 금액을 다시 점검하고, 최근 20거래일 TIME ETF 대비 격차가 {(sw['gap20'] - DD_MARGIN) * 100:+.1f}%p 이하"
                     f"(2026년 최대 {sw['gap20'] * 100:+.1f}%p보다 5%p 더 벌어짐)면 검증 범위 밖이라 점검합니다({SWITCH_BASIS}).")
    return lines


def my_pause_rule(ctx: dict) -> list[dict]:
    """멈춤·점검 기준. 성과 부진만으로는 멈추지 않고, 검증 범위를 벗어날 때만 행동을 바꾼다.

    ctx 키: account_drawdown(입출금 보정, 기록이 없으면 None), model_drawdown,
    model_vs_etf_20d, model_vs_etf_120d(이력이 짧으면 None). 선택 키: switch_applied(주문 기준이 §3.1 전환이면
    True — 내 계좌를 전환 경로 기준으로 본다), account_vs_etf_20d(내 계좌 20거래일 TIME ETF 대비, 이력이 짧으면 None).
    """
    out = []
    switch = bool(ctx.get("switch_applied"))
    acct_ref, acct_basis = (BACKTEST_SWITCH_2026, SWITCH_BASIS) if switch else (BACKTEST_2026, BASIS)
    breaches = []
    if ctx["account_drawdown"] is not None and ctx["account_drawdown"] <= acct_ref["mdd"] - DD_MARGIN:
        breaches.append((ctx["account_drawdown"], "내 계좌", acct_ref["mdd"], acct_basis))
    if ctx["model_drawdown"] is not None and ctx["model_drawdown"] <= BACKTEST_2026["mdd"] - DD_MARGIN:
        breaches.append((ctx["model_drawdown"], "7종목 전략(모의)", BACKTEST_2026["mdd"], BASIS))
    if breaches:
        dd, who, ref, basis = min(breaches)
        out.append({"level": "serious", "title": f"{who} 고점 대비 {dd:.1%} — 추가 입금 보류 기준 도달",
                    "body": f"2026년 검증 기간의 최대 하락률({ref:.1%}, {basis})보다 5%p 이상 더 하락했습니다. "
                            "검증한 위험 범위를 벗어났으므로 매매 규칙은 유지하되 추가 입금은 보류하세요. 감당할 수 있는 손실을 기준으로 투자 금액을 다시 점검하세요."})
    acct_gap = ctx.get("account_vs_etf_20d")
    if switch and acct_gap is not None and acct_gap <= BACKTEST_SWITCH_2026["gap20"] - DD_MARGIN:
        out.append({"level": "warning",
                    "title": f"내 계좌 최근 20거래일 TIME ETF 대비 {acct_gap * 100:+.1f}%p — 전환 경로 검증 범위 밖",
                    "body": f"전환 경로의 2026년 최대 격차({BACKTEST_SWITCH_2026['gap20'] * 100:+.1f}%p, {SWITCH_BASIS})보다 5%p 이상 더 벌어졌습니다. "
                            "매매 규칙은 유지하되 주문 기준·체결이 규칙대로였는지 점검하세요."})
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
