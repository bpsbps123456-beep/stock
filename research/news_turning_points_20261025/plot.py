"""Export a static study chart; requires matplotlib (not a runtime dependency)."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
DEFAULT_DATA=HERE.parents[1]/'data/derived/analysis/news_turning_points_20261025'

def draw(data: Path):
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    prices=pd.read_csv(HERE/'prices.csv',header=[0,1],index_col=0,parse_dates=True)
    e=pd.read_csv(data/'event_outcomes.csv',parse_dates=['date','entry'])
    p=pd.read_csv(data/'pivots_8.csv',parse_dates=['date'])
    style={'D':('#c53c36','v','기록적 매도·축소'), 'R':('#158369','^','매수 복귀·매물 인수'),
           'S':('#1877bd','s','극단적 숏 쏠림'), 'T':('#7955b4','D','롱·레버리지 쏠림')}
    fig,axes=plt.subplots(3,1,figsize=(16,12),layout='constrained')
    for ax,year in zip(axes,[2024,2025,2026]):
        close=prices['close']['SOXX']; close=close[close.index.year==year]
        ax.plot(close.index,close,color='#273f53',lw=1.7,label='SOXX 종가')
        lo=float(close.min()); hi=float(close.max()); span=hi-lo
        pp=p[p.date.dt.year==year]
        for kind,marker in [('bottom','o'),('top','x')]:
            q=pp[pp.kind==kind]
            ax.scatter(q.date,q.close,s=28,color='#9ca6ad',marker=marker,zorder=3,
                       label='사후 확정 저점' if kind=='bottom' else '사후 확정 고점')
        for fam,(color,marker,label) in style.items():
            g=e[(e.entry.dt.year==year)&(e.family==fam)&e.keep]
            y=prices['open']['SOXX'].reindex(g.entry).to_numpy()
            ax.scatter(g.entry,y,color=color,marker=marker,s=76,zorder=4,label=label)
            for j,((_,r),yy) in enumerate(zip(g.iterrows(),y)):
                offset=15 if j%2==0 else -23
                ax.annotate(r.id,(r.entry,yy),xytext=(0,offset),textcoords='offset points',
                            ha='center',fontsize=9,color=color)
        ax.set_title(f'{year}년  |  종가 기준 8% 반전으로 확정한 고점·저점',loc='left',fontsize=13,fontweight='bold')
        ax.set_ylabel('SOXX (달러)',fontsize=11)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m월'))
        ax.set_ylim(lo-.16*span,hi+.18*span)
        ax.grid(axis='y',alpha=.15)
        for s in ['top','right']: ax.spines[s].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    # All families get a label even when 2024 has no S event.
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=10)
    fig.suptitle('헤지펀드 뉴스는 SOXX 저점·고점에 얼마나 가까웠나?',fontsize=20,fontweight='bold')
    axes[-1].set_xlabel('색 표식은 기사 다음 미국 거래일의 시가. N번호는 전체 사건표와 연결. 회색 극값은 사후 정답이며 실시간 신호가 아님.\n2026년은 10월 23일까지. 10월 끝 저점 미확정. 출처: 당시 보도·Yahoo Finance, 직접 계산.',fontsize=10)
    for ext in ['png','svg']:
        fig.savefig(data/f'comparison_2024_2026.{ext}',dpi=150,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    draw(ap.parse_args().data_dir)
