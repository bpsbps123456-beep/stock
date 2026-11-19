"""Summarize the paired frequency comparisons and the fixed forward split."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
SCOPE = {"broad": "대형주에서 상대거래량 순", "time": "TIME 안에서 상대거래량 순",
         "desk": "기존 120일 순위에 자격 조건 추가", "blend": "기존과 대형주 검색식 절반씩",
         "switch": "SOXX 이탈 때 대형주로 교체", "exit": "대형주 검색식에 SOXX 대피 추가"}
LEVEL = {"ma": "이평선 3개", "technical": "가격·거래량·RSI 포함", "quality": "가격·거래량·재무",
         "ownership": "가격·거래량·재무·기관 근사"}
FREQ = {"daily": "매일", "weekly": "매주", "monthly": "매월"}
CONTROL = {"base": "기존 데스크 7종목", "base_SOXX100": "기존 + SOXX 100일선 대피",
           "base_half": "기존 목표 절반", "cash": "달러 현금", "SPY": "S&P 500 ETF 보유"}


def label(key):
    if key in CONTROL:
        return CONTROL[key]
    scope, mask, freq = key.split("_")
    return f"{SCOPE[scope]} · {LEVEL[mask]} · {FREQ[freq]}"


def table(stats, keys, include_midcaps=False):
    lines = ["| 전략 | 상반기 | 7월 이후 | 7월 이후 최대낙폭 | 연초 누적 | 7월 이후 주식비중 | 7월 이후 누적 거래대금/자산 | 7월 이후 10bp 차감 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in dict.fromkeys(keys):
        row = stats.loc[key]
        name = label(key).replace("대형주","대·중형주") if include_midcaps else label(key)
        lines.append(f"| {name} | {row.h1_return:+.2%} | {row.h2_return:+.2%} | "
                     f"{row.h2_mdd:+.2%} | {row.full_return:+.2%} | {row.h2_exposure:.1%} | "
                     f"{row.h2_turn:.1f}배 | {row.h2_return_10bp:+.2%} |")
    return "\n".join(lines)


def chart(stats, expanded, level):
    font = next((s for s in ["Malgun Gothic","AppleGothic","Noto Sans CJK KR"]
                 if s in font_manager.get_font_names()), None)
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1,2,figsize=(10.8,4.7),layout="constrained",sharey=True)
    labels = list(FREQ.values()) if font else ["Daily","Weekly","Monthly"]
    titles = ["대형주", "대형주 + 연초 중형주"] if font else ["Large cap", "Large and mid cap"]
    x = np.arange(3)
    for ax, sample, title in zip(axes, (stats, expanded), titles):
        selected = sample.loc[[f"broad_{level}_{f}" for f in FREQ]]
        for offset, field, color, label in [
                (-.18, "h2_return", "#3d8275", "수수료 0" if font else "Zero fee"),
                (.18, "h2_return_10bp", "#355882", "편도 0.1% 차감" if font else "10bp each side")]:
            bars = ax.bar(x+offset, selected[field]*100, width=.34, color=color, label=label)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
        ax.axhline(stats.loc["base","h2_return_10bp"]*100, lw=1.3, ls="--", color="#a9513f",
                   label="기존 7종목 · 편도 0.1%" if font else "Desk, 10bp")
        ax.set_xticks(x, labels=labels)
        ax.set_ylim(-45,0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.22)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("원화 수익률 (%)" if font else "KRW return (%)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("검색식 상위 7종목의 교체 주기별 성과 · 7월 1일~11월 17일" if font else
                 "Screened top seven by rebalance frequency · July 1 to November 17")
    fig.savefig(HERE/"comparison.png",dpi=155)
    plt.close(fig)


def write_report(stats, picks, ledgers, market):
    nearest = "ownership" if "broad_ownership_daily" in stats.index else "quality"
    selected = picks["all"]
    candidates = stats.drop(index=list(CONTROL))
    b = stats.loc["base"]
    outperform = (candidates.h2_return > b.h2_return) & (candidates.h2_mdd > b.h2_mdd)
    netout = (candidates.h2_return_10bp > b.h2_return_10bp) & (candidates.h2_mdd_10bp > b.h2_mdd_10bp)
    rows = [f"# FINVIZ 조건으로 종목을 교체한 2026년 모형 성과", "",
            f"1월 2일~11월 17일 {len(ledgers['base'])}거래일, 후보 {len(candidates)}개와 대조 5개를 비교했다. "
            "당시 미국 중대형주 전체의 FINVIZ 일별 합격 명단은 확보하지 못했다. 아래 숫자는 복원한 대형주·TIME 모집단에서의 원화 모형 수익률이다. "
            "가격 신호를 마감 때 확정하고 다음 정규장 시가에 거래했다. 날짜별 상대거래량은 직전 63거래일 평균 대비 거래량이다.", "",
            f"상반기 낙폭이 기존 이내인 후보에서 거래금액 편도 10bp를 차감한 상반기 수익으로 고른 결과는 **{label(selected)}**다. "
            f"이 규칙은 7월 이후 {stats.loc[selected,'h2_return']:+.2%}, 10bp 차감 후 {stats.loc[selected,'h2_return_10bp']:+.2%}였다. "
            f"같은 기간 기존 원화 수익은 {b.h2_return:+.2%}였다. 뒤 구간의 수익과 낙폭을 모두 개선한 후보는 기본 수수료에서 "
            f"{int(outperform.sum())}개, 10bp를 적용하면 {int(netout.sum())}개였다. 후보들은 같은 종목·날짜를 공유한다.", "",
            "## 원래 검색식에 가장 가깝게 복원한 대형주 조건", "",
            table(stats, ["base","SPY","cash","base_SOXX100", *[f"broad_{nearest}_{f}" for f in FREQ]]), "",
            "## 같은 재무 조건을 유지하고 두 방식 혼합", "",
            table(stats, [f"{scope}_{nearest}_{freq}" for scope in ("desk","time","blend","exit","switch") for freq in FREQ]), "",
            "## 월별·일별 교체에서 단계별 필터 효과", "",
            table(stats, [f"broad_{level}_{freq}" for level in LEVEL for freq in FREQ
                          if f"broad_{level}_{freq}" in stats.index]), "",
            "## 상반기에 고른 규칙과 뒤 구간", "",
            table(stats, ["base","base_half", *picks.values()]), "",
            "상반기 낙폭 기준에 맞는 후보가 없는 유형은 기존 전략으로 돌아간다. 선택 결과는 [유형별 선택](selection.csv)에 있다. "
            "최종 우열은 7~8월과 9~11월의 일관성, 달러 성과와 투자 비중 차이까지 함께 봐야 한다. "
            "당시 신고·발표 시점을 보존했어도 이미 여러 번 본 2026년 자료에 대한 재분석이다.", "",
            "## 재현 자료", "",
            "- [사전 비교 방법](protocol.md) · [전체 성과와 비용 민감도](results.csv) · [거래일별 선정 근거](decisions.csv) · [원화·달러 일별 자산](daily.csv)",
            "- [월초 모집단 원자료와 공개 시각](broad_sources.json) · [기술 지표 출처 및 종목별 누락](technical_inputs/sources.json)",
            "- 저장소에서 `py data/derived/analysis/finviz_rotation_20261119/compare.py`로 재계산한다."]
    rows.extend(["- [시점별 SEC 재무 자료·부채비율 정의](fundamental_inputs/method.json)",
                 "- [거래 중단·합병 처리 근거](technical_inputs/terminal_events.csv)"])
    for path in sorted(HERE.glob("*")):
        if path.name.startswith(("fundament", "ownership")) and path.suffix in (".json", ".md"):
            rows.append(f"- [{path.name}]({path.name})")
    off = pd.Series(~market, index=ledgers["base"].index)
    rows.extend(["", "## 이탈 후 재배치 기회와 7월 이후 월별 손익", "",
                 f"SOXX 100일선 아래에서 운용 방식이 갈린 날은 상반기 {off.loc[:'2026-06-30'].sum()}거래일, "
                 f"7월 이후 {off.loc['2026-07-01':].sum()}거래일이었다. 상반기에서 이탈 때의 교체 주기를 고를 기회가 많지 않았다.", "",
                 "| 전략 | 7월 | 8월 | 9월 | 10월 | 11월 17일까지 |",
                 "|---|---:|---:|---:|---:|---:|"])
    for key in dict.fromkeys(["base","base_SOXX100","SPY",selected,*[f"broad_{nearest}_{f}" for f in FREQ],
                              *[f"switch_{nearest}_{f}" for f in FREQ]]):
        series = ledgers[key].krw
        month_end = series.groupby(series.index.str[:7]).last().pct_change().loc["2026-07":]
        rows.append("| " + " | ".join([label(key), *[f"{v:+.2%}" for v in month_end]]) + " |")
    if nearest == "ownership":
        alternate = pd.read_csv(HERE/"sensitivity.csv", index_col=0)
        rows.extend(["", "## 이평선과 기관 지분 조건의 기여", "",
                     "아래는 같은 대형주 명단·상대거래량 순위·주문 시각으로 조건 하나를 바꿨을 때의 7월 이후 원화 수익이다. "
                     "기관 기준 해제는 기관 지분 자체가 확인되는 종목 안에서 문턱만 없앴다. 부채비율 대안은 FINVIZ 도움말의 유동부채/자본 식이다.", "",
                     "| 교체 | 전체 근사 | 이평선 3개 해제 | 기관 40% 문턱 해제 | 도움말 부채비율 |",
                     "|---|---:|---:|---:|---:|"])
        for f in FREQ:
            returns = [stats.loc[f"broad_ownership_{f}","h2_return"],
                       *[alternate.loc[f"{key}_{f}","h2_return"] for key in
                         ("without_ma","ownership_known_only","help_debt_and_ownership")]]
            rows.append("| " + " | ".join([FREQ[f], *[f"{v:+.2%}" for v in returns]]) + " |")
        rows.extend(["", "SOXX 이탈 시 교체 모형의 기본값은 미리 갱신해 온 검색식 바스켓의 그날 시가 비중을 인수한다. "
                     "다른 실행 해석으로, 이탈 당일에도 직전 종가로 검색을 새로 실시하고 그 뒤 정기일에 갱신했다. "
                     f"월간 방식의 7월 이후 원화 수익은 미리 갱신한 경우 {stats.loc['switch_ownership_monthly','h2_return']:+.2%}, "
                     f"이탈 시 즉시 다시 검색한 경우 {alternate.loc['rescreen_on_exit_monthly','h2_return']:+.2%}였다."])
    day, week = [f"broad_{nearest}_{f}" for f in ("daily","weekly")]
    def gap(cost):
        values = [(1 + stats.loc[key,"h2_return"]) *
                  np.prod(1-cost*ledgers[key].loc["2026-07-01":,"turn"].to_numpy()) for key in (day,week)]
        return values[0]-values[1]
    if gap(0)*gap(.01) < 0:
        lo, hi = 0., .01
        for _ in range(45):
            mid = (lo+hi)/2
            if gap(mid)*gap(0) > 0:
                lo = mid
            else:
                hi = mid
        rows += ["", f"7월 이후 대형주 매일·매주 교체의 원화 수익이 같아지는 편도 거래비용은 "
                 f"약 {(lo+hi)/2*10000:.1f}bp였다. 실제 비용을 추정한 값이 아니라 모형 거래금액으로 역산한 경계다."]
    decisions = pd.read_csv(HERE/"decisions.csv")
    passed = decisions[(decisions["filter"] == nearest) & (decisions.frequency == "daily") &
                       (decisions.trade_open >= "2026-07-01")]
    rows += ["", "## 7월 이후에 실제로 고를 수 있었던 종목 수", "",
             "| 후보 모집단 | 거래일 수 | 통과 종목 중앙값 | 통과 0개인 날 | 7개 미만인 날 |",
             "|---|---:|---:|---:|---:|"]
    for name, group in passed.groupby("universe"):
        if name in ("broad","time"):
            n = group.eligible_count
            rows.append(f"| {SCOPE[name]} | {len(n)} | {n.median():g} | {(n==0).sum()} | {(n<7).sum()} |")
    supplement = HERE/"midcaps/results.csv"
    if supplement.exists():
        expanded = pd.read_csv(supplement, index_col=0)
        rows += ["", "## 시작 당시 S&P 400 중형주까지 포함한 교차 확인", "",
                 "위 대형주 모집단에 연초 기준 중형주 명단을 합쳐 비교했다. 아래 행의 상대거래량 순위는 이 합집합 안의 순위다. "
                 "이 추가 표는 상반기 후보 선택과 별도로 계산했다. 월중 상장폐지·합병 처리와 가격 누락은 중형주 연구 기록을 따른다.", "",
                 table(expanded,[f"{scope}_ownership_{f}" for scope in ("broad","blend","switch")
                                 for f in FREQ if f"{scope}_ownership_{f}" in expanded.index], include_midcaps=True), "",
                 "확대 모집단의 15개 선별 방식·6개 혼합을 재계산하려면 같은 폴더의 `add_midcaps.py`를 먼저 실행한다. "
                 "[확대 모집단별 전체 성과](midcaps/results.csv) · [연초 중형주 명단 근거](midcaps/membership_source.json)"]
        ed, em = expanded.loc["broad_ownership_daily"], expanded.loc["broad_ownership_monthly"]
        rows[2:2] = [
            f"중형주까지 포함해 매일 7종목을 바꿨을 때 상반기는 {ed.h1_return:+.2%}, 7월 이후는 {ed.h2_return:+.2%}, "
            f"연초 누적은 {ed.full_return:+.2%}였다. 기존 전략은 같은 순서로 {b.h1_return:+.2%}, {b.h2_return:+.2%}, "
            f"{b.full_return:+.2%}였다. 하반기 편도 10bp 차감 수익은 매일 {ed.h2_return_10bp:+.2%}, "
            f"매월 {em.h2_return_10bp:+.2%}였다. 빈번한 교체의 이득은 모집단과 거래비용에 민감했다.", ""]
    if supplement.exists():
        rows += ["", "![모집단과 거래비용에 따른 교체 주기별 성과](comparison.png)"]
        chart(stats, expanded, nearest)
    (HERE/"report.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
