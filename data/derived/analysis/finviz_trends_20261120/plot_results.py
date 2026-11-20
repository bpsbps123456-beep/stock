"""Plot the four pre-specified account paths at 10 bp using stored, reconciled ledgers."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# label, display name, color, line style, occasional identifying marker
SERIES = {
    "baseline": ("현행 기준", "#263E53", "-", None),
    "anchored_h1": ("상반기 고정 선발", "#007B6F", (0, (6, 2.7)), "o"),
    "rolling": ("직전 분기마다 선발", "#AE5B09", (0, (5, 2, 1.3, 2)), "s"),
    "hindsight": ("연중 사후 최고", "#856F99", (0, (2, 2.2)), None),
}


def accounts():
    curves = pd.read_csv(HERE / "selected_curves.csv", parse_dates=["date"])
    results = pd.read_csv(HERE / "selected_results.csv")
    labels = list(SERIES)
    chosen = curves[curves.execution.eq("timely_10bp") & curves.label.isin(labels)]
    nav = chosen.pivot(index="date", columns="label", values="nav_krw").sort_index()[labels]
    assert nav.shape == (222, 4) and np.isfinite(nav).all().all() and (nav > 0).all().all()
    assert nav.index[0] == pd.Timestamp("2026-01-02") and nav.index[-1] == pd.Timestamp("2026-11-18")
    capital = pd.DataFrame([[1.] * len(labels)], index=[pd.Timestamp("2025-12-31")], columns=labels)
    ytd = pd.concat([capital, nav]) * 100
    h2 = nav.loc["2026-06-30":] / nav.loc["2026-06-30"] * 100
    stored = results[results.execution.eq("timely_10bp")].set_index(["label", "period"])
    for key in labels:
        for period, path in [("YTD", ytd), ("H2", h2)]:
            np.testing.assert_allclose(path[key].iloc[-1] / 100 - 1, stored.at[(key, period), "return"],
                                       rtol=0, atol=2e-8)
    return ytd, h2


def align_labels(values, low, high):
    """Keep endpoint labels apart if two portfolios finish near each other."""
    values = values.sort_values(kind="stable")
    gap = (high - low) * .066
    levels = []
    for value in values:
        levels.append(max(value, levels[-1] + gap) if levels else value)
    if levels[-1] > high:
        levels[-1] = high
        for i in range(len(levels) - 2, -1, -1):
            levels[i] = min(levels[i], levels[i + 1] - gap)
    return dict(zip(values.index, levels))


def pane(ax, frame, title, marks):
    ax.set_title(title, fontsize=13, loc="left", fontweight="bold", pad=13)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DFE4EA", linewidth=.65)
    ax.axhline(100, color="#87939D", lw=1)
    for mark in marks:
        ax.axvline(pd.Timestamp(mark), color="#B2BBC1", linestyle=(0, (3, 4)), lw=.7)
    for key, (_, color, style, marker) in SERIES.items():
        ax.plot(frame.index, frame[key], color=color, linestyle=style,
                lw=1.8 if key != "hindsight" else 1.5, marker=marker, ms=2.5,
                markevery=(0 if key == "anchored_h1" else 4, max(9, len(frame) // 11)))
    minimum, maximum = frame.to_numpy().min(), frame.to_numpy().max()
    padding = max((maximum - minimum) * .085, 1.)
    low, high = minimum - padding, maximum + padding
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    last = frame.iloc[-1]
    label_y = align_labels(last, low+padding*.23, high-padding*.23)
    stop = frame.index[-1]
    delta = (stop - frame.index[0]) * .026
    for key, y in last.items():
        color = SERIES[key][1]
        ax.annotate(f"{y-100:+.1f}%", xy=(stop, y), xytext=(stop + delta, label_y[key]),
                    color=color, fontsize=8.8, va="center", ha="left", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", shrinkA=1, shrinkB=1, lw=.65, color=color),
                    bbox=dict(facecolor="white", edgecolor="none", pad=.5))
    ax.set_xlim(frame.index[0], stop + delta * 5.5)
    months = pd.date_range(frame.index[0], stop, freq="MS")
    ax.set_xticks(months[::2] if len(months) > 6 else months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m월"))
    ax.tick_params(axis="both", length=0, labelsize=9, pad=5, colors="#455565")
    for name in ("left", "right", "top"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color("#BBC4CB")


def plot():
    options = ("Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "AppleGothic")
    fonts = {entry.name for entry in font_manager.fontManager.ttflist}
    selected = next((name for name in options if name in fonts), None)
    if selected is None:
        raise RuntimeError("The Korean labels require a Korean-capable matplotlib font.")
    plt.rcParams.update({"font.family": selected, "axes.unicode_minus": False, "text.color": "#203244"})
    ytd, h2 = accounts()
    fig, axes = plt.subplots(1, 2, figsize=(13.1, 5.8), dpi=180)
    fig.subplots_adjust(left=.057, right=.984, top=.72, bottom=.19, wspace=.14)
    fig.text(.057, .935, "FINVIZ 이평선과 시장 전환 규칙의 모의 성과", fontsize=19, fontweight="bold")
    fig.text(.057, .876, "미국 정규장 2026.01.02~11.18  ·  원화 기준  ·  편도 비용 0.10%  ·  우측 숫자는 기간 수익률",
             fontsize=10, color="#455565")
    pane(axes[0], ytd, "연초 이후 자산 지수  ·  전년 말 = 100", ["2026-04-01", "2026-07-01", "2026-10-01"])
    pane(axes[1], h2, "7월 이후 자산 지수  ·  6월 말 = 100", ["2026-10-01"])
    handles = [plt.Line2D([], [], color=color, linestyle=style, marker=marker, markersize=4, linewidth=2)
               for _, color, style, marker in SERIES.values()]
    fig.legend(handles, [v[0] for v in SERIES.values()], ncol=4, loc="lower left", frameon=False,
               bbox_to_anchor=(.05, .767), fontsize=10, handlelength=2.8, columnspacing=2.4)
    fig.text(.057, .083, "상반기 고정 선발은 7월부터, 분기별 선발은 4월부터 직전 구간의 결과를 사용했습니다.",
             fontsize=9.2, color="#455565")
    fig.text(.057, .04, "연중 사후 최고는 11월 18일까지의 결과를 모두 본 뒤 고른 조합입니다.",
             fontsize=9.2, color="#695574")
    fig.savefig(HERE / "comparison.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot()
