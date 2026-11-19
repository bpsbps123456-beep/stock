"""Render the five representative net KRW account curves from the frozen daily results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPLIT = pd.Timestamp('2026-06-30')


def draw():
    selected = json.loads((HERE / 'selection.json').read_text(encoding='utf-8'))
    studies = [('selected_daily', 'results', ['base', 'switch']),
               ('robustness_daily', 'robustness', ['D2M3E2-m-em-c0']),
               ('conditional_daily', 'conditional_results', ['3F2M2E_cap2']),
               ('completion_daily', 'completion_results', ['D5F1E1-d-em-c2'])]
    returns = []
    for daily, summary, keys in studies:
        panel = pd.read_csv(HERE / f'{daily}.csv', parse_dates=['date'])
        totals = pd.read_csv(HERE / f'{summary}.csv')
        if daily == 'completion_daily':
            panel = panel[panel.vacancies.eq('fill')]
            totals = totals[totals.vacancies.eq('fill')]
        curve = panel.pivot(index='date', columns='strategy', values='krw10').sort_index().loc[:, keys]
        assert len(curve) == selected['sessions'] and curve.index[-1] == pd.Timestamp(selected['dates'][1])
        np.testing.assert_allclose(curve.iloc[-1] - 1, totals.set_index('strategy').loc[keys, 'full_return_10bp'], atol=1e-9)
        returns.append(curve)
    nav = pd.concat(returns, axis=1)
    font = next((f for f in ['Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic']
                 if any(face.name == f for face in font_manager.fontManager.ttflist)), None)
    ko = font is not None
    if ko:
        plt.rcParams['font.family'] = font
    plt.rcParams['axes.unicode_minus'] = False
    def lang(a, b):
        return a if ko else b

    labels = [
        ('switch', lang('이탈 시 월간 FINVIZ로 교체', 'Switch to monthly FINVIZ'),
         lang('회복 시 기존 데스크로 복귀', 'Resume the original desk on recovery'), '#86501f', (0, (5, 2)), 2.3),
        ('D2M3E2-m-em-c0', lang('기존 2 · 이평 3 · 실적 2', 'Desk 2 · MA 3 · earnings 2'),
         lang('섹터 상한 없음 · 이평과 실적은 매월 교체', 'No sector cap · monthly MA and earnings'), '#007c69', '-', 2.4),
        ('3F2M2E_cap2', lang('이탈 시 세 전략으로 교체', 'Switch to three strategies'),
         lang('이탈 중 FINVIZ 3 · 이평 2 · 실적 2 / 섹터당 2종목', 'FINVIZ 3 · MA 2 · earnings 2 / 2 per defensive sector'),
         '#8e609d', (0, (2, 2)), 2.1),
        ('base', lang('기존 데스크', 'Original desk'), lang('주간 완충 규칙 7종목', '7 weekly buffered positions'), '#21374a', '-', 2.3),
        ('D5F1E1-d-em-c2', lang('기존 5 · FINVIZ 1 · 실적 1', 'Desk 5 · FINVIZ 1 · earnings 1'),
         lang('빈자리 충원 · 섹터당 2종목 · FINVIZ 매일 교체', 'Fill vacancies · 2 per sector · daily FINVIZ'), '#427bb2', '-', 1.8),
    ]
    fig = plt.figure(figsize=(11.8, 5.9), facecolor='white')
    ax = fig.add_axes([.077, .185, .590, .615]); summary = fig.add_axes([.700, .155, .279, .658])
    summary.set_axis_off()
    fig.text(.077, .918, lang('7자리 혼합 전략의 연중 누적 수익률', 'Year-to-date returns of seven-position strategies'),
             fontsize=20, weight='bold', color='#152d3a')
    end = nav.index[-1]
    fig.text(.077, .859, lang(f'2026년 1월 2일~{end.month}월 {end.day}일  |  원화 기준 · 매수와 매도 각각 거래금액의 0.10% 반영',
                             'Jan 2 – Nov 18, 2026  |  KRW returns, after 0.10% per side'), fontsize=10.5, color='#445767')
    start = pd.Timestamp('2025-12-31')
    ix = [start, *nav.index]
    ax.axvspan(SPLIT, end, color='#f1f5f4', zorder=0)
    ax.axvline(SPLIT, color='#9ba8ad', linestyle=(0, (3, 3)), linewidth=1)
    ax.axhline(0, color='#59666d', linewidth=.75)
    summary.text(0, .96, lang('전략', 'Strategy'), fontsize=10, color='#4f5c62', transform=summary.transAxes)
    summary.text(1, .96, lang('11월 18일', 'Nov 18'), ha='right', fontsize=10, color='#4f5c62', transform=summary.transAxes)
    summary.plot([0, 1], [.928]*2, lw=1, color='#cad2d5', transform=summary.transAxes)
    for i, (key, name, desc, color, style, width) in enumerate(labels):
        values = nav[key]
        if values.isna().any():
            raise ValueError(f'Incomplete account curve: {key}')
        y = np.r_[0., (values.to_numpy() - 1) * 100]
        ax.plot(ix, y, color=color, linestyle=style, linewidth=width, solid_capstyle='round')
        ax.scatter(end, y[-1], color=color, s=17, zorder=5, clip_on=False)
        y0 = .838 - i*.178
        summary.plot([0, .055], [y0]*2, color=color, ls=style, lw=width, transform=summary.transAxes)
        summary.text(.082, y0, name, va='center', fontsize=10, weight='bold', color='#1c3545', transform=summary.transAxes)
        summary.text(1, y0, f'{y[-1]:+.1f}%', ha='right', va='center', fontsize=12, weight='bold', color=color, transform=summary.transAxes)
        summary.text(.082, y0-.062, desc, va='center', fontsize=8.3, color='#536572', transform=summary.transAxes)
    ax.set_xlim(start, end)
    ax.margins(y=.08)
    ax.text(SPLIT + pd.Timedelta(days=4), .97, lang('7월 이후', 'July onward'), transform=ax.get_xaxis_transform(),
            ha='left', va='top', color='#4a595b', fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.set_xlabel(lang('월', 'Month'), color='#566670', fontsize=9)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis='y', alpha=.5, linewidth=.7, color='#c8d1d4')
    ax.tick_params(axis='both', colors='#475962', length=0, pad=7, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.077, .069, lang('이탈 신호는 SOXX 종가가 100일선 아래로 내려간 경우입니다. 조건부 전략은 다음 미국 시가에 교체하고 회복 시 복귀합니다.',
                             'A SOXX close below its 100-day average triggers a switch at the next US open. Recovery restores the original desk.'),
             color='#526571', fontsize=9)
    fig.savefig(HERE / 'comparison.png', dpi=155, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    draw()
