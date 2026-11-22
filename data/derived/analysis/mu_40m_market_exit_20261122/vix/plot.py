"""9월·10월의 회복 신호에서 VIX·VVIX와 직전 63거래일 기준을 비교한다."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from volatility import HERE, load_volatility


def figure():
    data = load_volatility()
    plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "svg.fonttype": "none",
                         "svg.hashsalt": "gbrc-vix-20261122", "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 5.2), sharex=True)
    fig.subplots_adjust(left=.085, right=.97, top=.82, bottom=.11, hspace=.31)
    selected = data.loc["2026-09-01":"2026-10-30"]
    for ax, col in zip(axes, ["VIX", "VVIX"]):
        values = selected[col]
        line = data[col].shift().rolling(63).quantile(.8).reindex(selected.index)
        ax.plot(values.index, values, color="#124b62", linewidth=1.8, label="종가")
        ax.plot(line.index, line, color="#bc5a29", linestyle="--", linewidth=1.6,
                label="직전 63거래일의 80백분위")
        ax.fill_between(values.index, line, values, where=values >= line, interpolate=True,
                        color="#d88e64", alpha=.21)
        ax.set_ylabel(col, rotation=0, labelpad=23, color="#173443", fontweight="bold")
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#bdcbcf")
        ax.grid(axis="y", color="#e0e6e6", linewidth=.7)
        ax.tick_params(axis="both", length=0, labelcolor="#34494f")
        for date in ("2026-09-17", "2026-10-28"):
            day = pd.Timestamp(date)
            ax.axvline(day, color="#849497", linestyle=":", linewidth=1)
            ax.scatter(day, values.at[day], color="#124b62", s=32, zorder=4)
        ax.margins(x=.012, y=.14)
    for date, align in [("2026-09-17", "center"), ("2026-10-28", "right")]:
        axes[0].text(pd.Timestamp(date), 1.015, date[5:].replace("-", "/")+"  SOXX 회복",
                     transform=axes[0].get_xaxis_transform(), ha=align, va="bottom", fontsize=9)
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.suptitle("SOXX 회복 첫날, 평소보다 변동성이 높았나", x=.085, ha="left", fontsize=17,
                 color="#173443", fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper left",
               bbox_to_anchor=(.077,.917), ncol=2)
    return fig


if __name__ == "__main__":
    figure().savefig(HERE / "relative.svg", metadata={"Date": None})
