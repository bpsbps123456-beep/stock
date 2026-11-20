"""Value selected November 18 holdings for one additional session without rescreening."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/'seven_sleeves_20261120'), str(HERE.parents[3])]
from market import load_inputs
from baselines import market_gate
from extension import get_forward, hold_return
from timing import review_days


def run():
    src = load_inputs()
    next_day = get_forward(src)
    holdings = json.loads((HERE/'end_weights.json').read_text())
    start = pd.read_csv(HERE/'selected_results.csv',index_col='key')
    date = '2026-11-19'
    daily_soxx = src.book.cls.SOXX.dropna().loc[:src.signals[-1]]
    tail = pd.read_csv(HERE.parent/'seven_sleeves_20261120/tail_prices.csv',index_col='symbol')
    closes = np.r_[daily_soxx.iloc[-99:], tail.at['SOXX','close']]
    allowed = np.r_[market_gate(src), closes[-1] >= closes.mean()]
    is_thursday = pd.Timestamp(date).dayofweek == 3
    assert is_thursday and len(closes)==100
    rows = []
    for key, holdings_at_close in holdings.items():
        w = np.zeros(len(src.symbols))
        for symbol, part in holdings_at_close.items():
            if symbol != 'CASH':
                w[src.loc[symbol]] = part
        np.testing.assert_allclose(w.sum()+holdings_at_close['CASH'], 1., atol=1e-12)
        change = hold_return(w, src, next_day)
        screen_due = '/' in key and len(src.days) in review_days([*src.days,date],allowed,key.split('/')[1])
        follows_rule = key=='base' or (allowed[-1]==allowed[-2] and (allowed[-1] or not screen_due))
        rows.append(dict(key=key, holdings_date=src.days[-1], mark_date=date,
                         mark_return=change, marked_ytd=(1+start.at[key,'year_ret'])*(1+change)-1,
                         scheduled_hold=bool(follows_rule)))
    pd.DataFrame(rows).to_csv(HERE/'forward_mark.csv',index=False,float_format='%.11g')


if __name__ == '__main__':
    run()
