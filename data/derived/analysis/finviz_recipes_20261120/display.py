"""Render the research tables and a sensitivity figure from the committed account ledgers."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SCHEDULE_NAMES = {'daily':'매일', 'weekly1':'매주 첫날', 'weekly3':'매주 셋째 날', 'biweekly':'격주 첫날',
                  'monthly1':'월 첫날', 'monthly6':'월 6번째 날', 'monthly11':'월 11번째 날',
                  'monthly1_11':'월 1·11번째 날', 'exit':'이탈 당일', 'exit5':'이탈 당일·5거래일 간격'}
BASE = 'original/monthly1'


def pct(v):
    return f'{v:+.2%}'.replace('-', '−')


def markdown_table(headings, rows):
    return ['| '+' | '.join(headings)+' |', '|---'+'|---:'*(len(headings)-1)+'|',
            *['| '+' | '.join(row)+' |' for row in rows]]


def plot(daily, sw, pick, labels):
    font = next((s for s in ('Malgun Gothic','Noto Sans CJK KR','NanumGothic')
                 if any(f.name == s for f in font_manager.fontManager.ttflist)), None)
    if font:
        plt.rcParams['font.family'] = font
    plt.rcParams['axes.unicode_minus'] = False
    names = list(dict.fromkeys([BASE,pick['train_pick'],pick['hindsight']]))
    colors = ['#475562','#007967','#9e4d28']
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.1), gridspec_kw={'height_ratios':[1.35,1]})
    for name, color in zip(names, colors):
        row = daily.xs(name).copy()
        date = pd.to_datetime(row.index)
        axes[0].plot(date, (row.krw10-1)*100, color=color, linewidth=2,
                     label=labels(name) if font else name)
    axes[0].axvline(pd.Timestamp('2026-07-01'), color='#999999', linestyle='--', linewidth=.9)
    axes[0].axvline(pd.Timestamp('2026-10-01'), color='#999999', linestyle='--', linewidth=.9)
    axes[0].set_xlim(pd.Timestamp('2026-01-01'),pd.Timestamp('2026-11-19'))
    axes[0].set_title('2026년 1월 2일~11월 18일  ·  매수·매도 각각 0.10% 비용 반영' if font
                      else 'Jan 2 – Nov 18, 2026  |  0.10% per side', fontsize=10, color='#505d64')
    axes[0].set_ylabel('연초 이후 수익률 (%)' if font else 'YTD return (%)')
    axes[0].legend(frameon=False, loc='upper left', fontsize=9)
    schedules = list(SCHEDULE_NAMES)
    chosen = list(dict.fromkeys(k.split('/')[0] for k in names))
    for name, color in zip(chosen, colors):
        heights = [sw.loc[f'{name}/{s}', 'year_ret']*100 for s in schedules]
        axes[1].plot(range(len(schedules)), heights, '.-', markersize=8, linewidth=1.5, color=color,
                     label=labels(f'{name}/monthly1').split(' · ')[0] if font else name)
    axes[1].set_xticks(range(len(schedules)), [SCHEDULE_NAMES[s] if font else s for s in schedules],
                       rotation=20, ha='right', fontsize=8)
    axes[1].set_ylabel('11월 18일 누적 (%)' if font else 'Nov 18 return (%)')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('FINVIZ 검색식과 교체 시점에 따른 원화 수익률' if font else 'FINVIZ rules and execution dates', fontsize=15)
    fig.tight_layout()
    fig.savefig(HERE/'comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def render():
    pick = json.loads((HERE/'selection.json').read_text())
    meta = json.loads((HERE/'screens.json').read_text(encoding='utf-8'))
    sw = pd.read_csv(HERE/'switch_results.csv',index_col='key')
    al = pd.read_csv(HERE/'always_results.csv',index_col='key')
    selected = pd.read_csv(HERE/'selected_results.csv',index_col='key')
    daily = pd.read_csv(HERE/'selected_daily.csv',index_col=['key','date'])
    eps = pd.read_csv(HERE/'episodes.csv')
    mark = pd.read_csv(HERE/'forward_mark.csv',index_col='key')
    def label(key):
        if key in ('base','exit'):
            return {'base':'기존 데스크','exit':'이탈 시 달러 현금'}[key]
        name, schedule = key.split('/')
        return meta[name]['name']+' · '+SCHEDULE_NAMES[schedule]
    fwd, top, oct_pick = (pick[x] for x in ('train_pick','hindsight','october_pick'))
    keys = list(dict.fromkeys(['base','exit',BASE,fwd,top,pick['superior_hindsight']]))
    text = ['# FINVIZ 검색식과 교체 날짜를 바꾸면 올해 수익이 늘었나', '',
            '미국 거래일 2026년 1월 2일~11월 18일에 같은 기존 데스크를 운용하고, SOXX가 100일선 아래에 있는 동안 사용할 선별 규칙과 교체 일정만 바꿨다. 주 비교는 원화 가격 수익과 매수·매도 각각 0.10% 비용을 적용한다. 월 첫 거래일 원본 FINVIZ의 기존 연구 222일 원장을 다시 맞췄다.', '',
            f'**다음 모의 비교 후보는 매월 첫 거래일에 갱신하는 재무 강화식이다.** 원본에서 {meta[top.split("/")[0]]["changes"]} 두 조건을 적용했다. 월말 미국 종가 기준 상대거래량 상위 7개로 다음 달 후보 바스켓을 구성했다. SOXX 종가가 100일선 아래로 내려가면 다음 정규장 시가에 그 바스켓으로 전환하고, 선을 회복하면 다음 시가에 기존 데스크로 복귀했다. [FINVIZ에서 후보 검색식 열기]({meta[top.split("/")[0]]["url"]})', '',
            f'전체 기록을 본 후의 연중 1위는 **{label(top)}**로 {pct(sw.at[top,"year_ret"])}였다. 원본 월초보다 연중 수익·7월 이후 수익·7월 이후 최대낙폭이 모두 좋아진 조합은 {pick["superior"]}/{pick["candidates"]}개였다. 서로 같은 48개 이탈 거래일을 여러 규칙이 공유한다.', '',
            f'상반기 자료로 선택한 규칙은 **{label(fwd)}**였다. 연중 누적 {pct(sw.at[fwd,"year_ret"])}, 7월 이후 {pct(sw.at[fwd,"h2_ret"])}였다. 기존 월초 검색식은 연중 {pct(sw.at[BASE,"year_ret"])}, 7월 이후 {pct(sw.at[BASE,"h2_ret"])}였다. 7~9월 수익과 최대낙폭 검사 후 10월부터 유지할 후보는 {label(oct_pick)}로 정했다. 이 규칙은 10월 이후 기존보다 손실이 컸다.', '',
            '## 같은 시장 이탈 신호에서의 성과', '']
    cols = ['year_gross_ret','year_ret','h1_ret','h2_ret','h2_dd','year_dd','year_stress_ret']
    text.extend(markdown_table(['방법','연중 비용 0','연중 0.10%','상반기','7월 이후','7월 이후 최대낙폭','연중 최대낙폭','연중 0.25%'],
                               [[label(k),*[pct(selected.at[k,c]) for c in cols]] for k in keys]))
    beaten = sw[sw[['year_ret','h2_ret','h2_dd']].gt(sw.loc[BASE,['year_ret','h2_ret','h2_dd']].astype(float)).all(axis=1)]
    text += ['', '세 지표를 모두 개선한 규칙과 이탈 중 투자 비중은 다음과 같다.', '']
    text.extend(markdown_table(['검색식 · 교체일','연중','7월 이후','7월 이후 최대낙폭','이탈 중 평균 종목 수','평균 주식 비중'],
              [[label(k),pct(v.year_ret),pct(v.h2_ret),pct(v.h2_dd),f'{v.off_held:.2f}',f'{v.off_exposure:.2%}']
               for k,v in beaten.sort_values('year_ret',ascending=False).iterrows()]))
    periods = eps[eps.key.eq(BASE)]
    sector_info = []
    for key in [BASE,top]:
        account = daily.xs(key)
        off = pd.concat([account.loc[t.exit:t.last_off] for t in periods.itertuples()])
        sector_info.append(f'{label(key)}은 평균 {off.sectors.mean():.2f}개 섹터, 한 섹터에 가장 많이 투자한 날의 비중은 {off.max_sector_weight.max():.2%}')
    text += ['', '이탈 48일을 기준으로 '+', '.join(sector_info)+'였다. 후보가 7개에 모자라면 해당 금액은 달러로 남았다.']
    cap = pd.read_csv(HERE/'sector_check.csv')
    text += ['', '## 한 섹터에 최대 두 종목만 담으면', '',
             '사후 분산 점검에서는 매월 상대거래량 순으로 내려가며 이미 두 종목을 채운 섹터를 건너뛰었다. 조건에 맞는 후보가 7개보다 적으면 빈자리는 현금이다. 섹터 수와 최대 섹터 비중은 이탈 48거래일 기준이다.', '']
    text.extend(markdown_table(['월초 검색식','섹터당 종목 수','연중','7월 이후','연중 최대낙폭','평균 종목 수','평균 섹터 수','최대 섹터 비중'],
                [[meta[row.recipe]['name'],'제한 없음' if row.riskoff_sector_cap == 0 else '최대 2개',
                  pct(row.year_ret),pct(row.h2_ret),pct(row.year_dd),f'{row.off_mean_names:.2f}',
                  f'{row.off_mean_sectors:.2f}',f'{row.off_max_sector_weight:.2%}'] for row in cap.itertuples()]))
    text += ['', '두 검색식 모두 섹터 집중도는 낮아졌지만 7월 이후 손실과 연중 최대낙폭이 커졌다. 재무 강화식에 섹터 제한을 더하면 이탈 중 평균 주식 비중도 91.95%에서 82.11%로 낮아졌다. [섹터별 종목 수 제한 대조](sector_check.csv)']
    independent = pd.read_csv(HERE/'claude/results.csv',index_col='path')
    independent_summary = json.loads((HERE/'claude/summary.json').read_text(encoding='utf-8'))
    independent_names = {'R0':'원본 월초 상위 7개', 'R1':'월초·16일 이후 첫 거래일마다 상위 7개',
                         'R2':'같은 월 2회, 보유 종목이 합격 명단 14위 이내면 유지',
                         'RF':'월초, 최근 분기 매출 성장·영업흑자 추가'}
    text += ['', '## Claude가 미리 정한 두 가설', '',
             '같은 222거래일의 기존 월초 원장을 독립 재현한 뒤, 교체를 줄이는 보유 규칙과 실적 조건을 각각 비교했다. 비용은 편도 0.10%다.', '']
    text.extend(markdown_table(['선별·교체 규칙','연중','7월 이후','연중 최대낙폭'],
                               [[name,*[pct(independent.at[k,c]) for c in
                                 ('full_return_10bp','h2_return_10bp','full_mdd_10bp')]]
                                for k,name in independent_names.items()]))
    text += ['', f'두 후보 모두 사전에 정한 채택 기준을 충족하지 못했다. 보유 기준을 완화한 월 2회 방식은 연중 최대낙폭을 조금 줄였지만 원본 월초보다 7월 이후 손실이 컸다. 같은 합격 명단에서 무작위로 뽑은 월초 바스켓 300개의 연중 중앙값은 {pct(independent_summary["placebo_percentiles"]["R0"]["placebo_median"])}였고, 상대거래량으로 정렬한 원본 월초는 {independent_summary["placebo_percentiles"]["R0"]["percentile"]:.0f}번째 백분위였다. [Claude의 사전 가설](claude/HYPOTHESIS.md) · [독립 결과와 월 1·2회 무작위 대조 600개](claude/REPORT.md)']
    text += ['', '## 검색식 변경과 교체 시점의 효과', '',
             '아래 네 비교는 원래 월초 규칙, 새 검색식만 적용, 날짜만 변경, 두 가지를 함께 변경한 경우다. 같은 비교가 중복되면 한 행만 표시한다.', '']
    ablations = list(dict.fromkeys(k for final in [fwd,top] for k in
                      [BASE,final.split('/')[0]+'/monthly1','original/'+final.split('/')[1],final]))
    text.extend(markdown_table(['검색식 · 교체일','연중 수익','7월 이후','연중 최대낙폭','이탈 중 주식 비중','연중 매매대금/자산'],
             [[label(k),pct(sw.at[k,'year_ret']),pct(sw.at[k,'h2_ret']),pct(sw.at[k,'year_dd']),
               f'{sw.at[k,"off_exposure"]:.1%}',f'{sw.at[k,"year_turn"]:.1f}배'] for k in ablations]))
    designs = list(dict.fromkeys(['original',fwd.split('/')[0],top.split('/')[0]]))
    text += ['', '각 날짜는 미국 정규장 거래일 순서다. 아래는 교체 시점만 바꿨을 때의 연중 비용 후 수익이다.', '']
    text.extend(markdown_table(['점검 일정',*[meta[n]['name'] for n in designs]],
             [[sname,*[pct(sw.at[f'{n}/{s}','year_ret']) for n in designs]]
              for s,sname in SCHEDULE_NAMES.items()]))
    text += ['', '같은 상대거래량 내림차순에서 하한만 1배→1.5배·2배로 올리면 순위가 바뀌지 않는다. 상위 7개가 달랐던 110일·193일은 강화된 조건이 7개를 채우지 못한 날이었다. 하한을 아예 없애도 상위 7개는 원본이 7개를 채우지 못한 5일에만 달라졌다. 이 비교는 종목 발굴뿐 아니라 현금 보유 효과를 측정한다.']
    diag = pd.read_csv(HERE/'sensitivity.csv',index_col=['recipe','timing'])
    options = {'fresh1':'월 첫 시가, 직전 종가로 선발', 'stale2':'첫 시가 명단을 고정해 둘째 시가에 체결',
               'fresh2':'둘째 시가, 직전 종가로 새로 선발', 'fresh3':'셋째 시가, 직전 종가로 새로 선발'}
    text += ['', '## 첫 거래일 명단과 실제 체결을 하루 옮겨 본 결과', '',
             '연중 결과를 본 후 두 검색식의 발표·체결 시차에 대한 민감도를 확인했다. SOXX 시장 전환 시점은 모든 행에서 동일하다. 이 검사는 후보를 다시 고르는 데 사용하지 않았다.', '']
    text.extend(markdown_table(['선발 및 체결','원본 연중','재무 강화 연중','원본 7월 이후','재무 강화 7월 이후'],
        [[desc,*[pct(diag.at[(n,t),f]) for f in ('year_ret','h2_ret') for n in ('original','balance')]]
         for t,desc in options.items()]))
    text += ['', '재무 강화식과 매월 같은 수의 투자 자리를 쓰게 한 원본 검색식은 연중 +133.02%, 7월 이후 −33.50%였다. 이탈 중 평균 주식 비중은 해당 대조 90.21%, 재무 강화식 91.95%였다. 현금 비중을 비슷하게 맞춰도 이번 표본에서는 재무 강화식이 앞섰다. [체결 지연·현금 비중 대조](sensitivity.csv)']
    text += ['', '![조건별 연중 원화 수익과 교체 시점별 결과](comparison.png)', '',
             '## 앞 구간으로 선택하고 뒤 구간으로 확인', '',
             f'상반기 SOXX 이탈은 {pick["h1_off"]}일이었다. 별도 상시 FINVIZ 계좌에서 8개 일정의 상반기 성과를 같은 일정의 원본 검색식과 비교했다. 평균 투자 비중과 낙폭 기준은 사전 설계를 따랐다. 7~9월 두 기준 통과 여부는 **{"통과" if pick["q3_pass"] else "미통과"}**였다. 이 판단에 10월 이후 성과를 쓰지 않았다.', '']
    text.extend(markdown_table(['조건부 계좌','7~9월 수익','7~9월 최대낙폭','10월~11월 18일 수익','10월 이후 최대낙폭'],
              [[label(k),*[pct(sw.at[k,c]) for c in ['q3_ret','q3_dd','oct_ret','oct_dd']]]
               for k in dict.fromkeys([BASE,fwd,top])]))
    text += ['', '## 다섯 번의 시장 이탈', '',
             '각 칸은 같은 구간 원본 월초 전략 대비 상대 자산 수익이다. 예컨대 +2%는 원본의 같은 구간 말 자산보다 2% 많았다는 뜻이다. 이탈 직전 종가부터 복귀일 종가까지이며, 마지막 구간은 연구 종료일 종가까지다.', '']
    keys2 = list(dict.fromkeys([fwd,top]))
    grouped = eps.pivot(index='episode',columns='key',values='relative_return')
    dates = eps[eps.key.eq(BASE)]
    text.extend(markdown_table(['이탈일','마지막 이탈일','이탈 거래일', *[label(k) for k in keys2]],
             [[v.exit,v.last_off,str(v.sessions_off),*[pct(grouped.at[v.episode,k]) for k in keys2]] for v in dates.itertuples()]))
    text += ['', f'사후 1위가 원본 월초를 앞선 이탈은 {int(sw.at[top,"episodes_ahead"])}/5회다. 가장 유리한 한 구간을 제외할 때 나머지의 원본 대비 누적 상대 수익은 {pct(sw.at[top,"worst_excluding_one"])}다. 다섯 번 중 네 번 이상 앞선 조합은 {pick["four_or_more_of_five"]}개였다.', '',
             '## 11월 19일 추가 종가', '',
             '11월 18일 종가 보유 수량을 그대로 유지해, 선발에 쓰지 않은 다음 거래일의 종가와 환율로 평가했다. 이미 마감한 하루이며 19일 신규 주문은 포함하지 않는다.', '']
    text.extend(markdown_table(['전략','추가 하루 수익','동결 보유 연장 시 연초 누적','그날 종목 점검'],
                               [[label(k),pct(mark.at[k,'mark_return']),pct(mark.at[k,'marked_ytd']),
                                 '예정된 보유 유지' if mark.at[k,'scheduled_hold'] else '점검일, 동결 보유만 평가']
                                for k in keys]))
    text += ['', '## 실제 검색식과 재현 범위', '',
             '모두 미국기업·중형주 이상·가격 10달러 등 원래 조건을 출발점으로 한다. 링크는 해당 조건을 FINVIZ에서 여는 용도다. 클릭 시점의 결과 목록과 과거 연구 표본은 달라질 수 있다. 필터별 변경 사항은 아래와 같다.', '']
    for item in meta.values():
        text.append(f'- {item["name"]} — {item["changes"]}. [설정 열기]({item["url"]})')
    text += ['', '월별 대형주 명단과 연초 전 고정 중형주 401개를 사용했다. SEC·13F는 당시 알려진 데이터로 반영했고, FINVIZ의 기관보유율·장중 시간조정 상대거래량은 종가 자료로 근사했다. 후보 부족·종료 종목 자리는 달러 현금이다. 환율, 교체 전날 밤의 가격 변동, 편도 비용을 반영했고 배당·현금 이자·세금은 제외했다. 가격이 비슷하게 움직이는 규칙이 많아 170개 비교를 독립된 증거로 볼 수 없다. [FINVIZ 정의](https://finviz.com/help/screener) · [SEC 데이터](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)', '',
             '상시 FINVIZ 단독은 별도 대조다. 같은 후보를 연중 계속 보유하는 것과 시장 이탈 기간에만 보유하는 결과를 구분했다. '
             f'상시 단독 원본 월초는 연중 {pct(al.at[BASE,"year_ret"])}, 상반기 선택 규칙은 {pct(al.at[fwd,"year_ret"])}, '
             f'조건부 사후 1위의 같은 검색식·같은 일정은 {pct(al.at[top,"year_ret"])}였다. 조건부 전략의 연중 수익에는 기존 계좌로 보낸 174거래일의 수익이 포함된다.', '',
             '저장된 입력으로 이 폴더에서 research.py를 실행한 뒤 forward.py, sensitivity.py, sector_check.py를 실행한다. Claude의 계산은 claude/code/run.py와 claude/code/diagnose.py로 재현할 수 있다. 마지막에 display.py를 실행하면 이 보고서와 그래프가 생성된다. [사전 설계](protocol.md) · [조건부 170개](switch_results.csv) · [상시 136개](always_results.csv) · [상반기 선발 원장](training.csv) · [구간별 상대성과](episodes.csv)']
    (HERE/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    plot(daily,sw,pick,label)


if __name__ == '__main__':
    render()
