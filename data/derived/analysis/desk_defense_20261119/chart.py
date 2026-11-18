"""Plot the forward comparison from saved ledgers."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

HERE = Path(__file__).resolve().parent


def draw():
    font = next((f for f in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR")
                 if f in font_manager.get_font_names()), None)
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    grid = pd.read_csv(HERE / "candidates.csv", index_col=0)
    daily = pd.read_csv(HERE / "selected_daily.csv", parse_dates=["date"]).pivot(
        index="date", columns="strategy", values="krw")
    chosen = grid.index[grid.global_selection][0]
    keys = ["base", chosen, "half"]
    colors = ["#304a60", "#bf531e", "#25836e"]
    labels = ["기존 7종목", "상반기 수익 기준 선택", "기존 목표의 절반"] if font else [
        "Weekly desk", "H1 return selection", "Half exposure"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), layout="constrained")
    for i, ax in enumerate(axes):
        values = daily[keys] * 100
        if i:
            values = values.loc["2026-06-30":] / daily.loc["2026-06-30", keys]
        else:
            values.loc[pd.Timestamp("2025-12-31")] = 100
            values = values.sort_index()
            ax.axvline(pd.Timestamp("2026-06-30"), color="#adb4b8", linestyle=":")
        for key, color, label in zip(keys, colors, labels):
            ax.plot(values.index, values[key], color=color, lw=1.9, label=label)
        titles = ["연초 원금 100 기준", "6월 30일 종가 100 기준"] if font else [
            "Starting capital = 100", "June 30 close = 100"]
        ax.set_title(titles[i], loc="left")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1 if i else 2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        ax.set_xlabel("2026년 월" if font else "Month in 2026")
        ax.grid(axis="y", alpha=.24)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("원화 모형 계좌 · 11월 17일 종가까지" if font else "KRW model equity through November 17",
                 weight="bold")
    fig.savefig(HERE / "comparison.png", dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    draw()
