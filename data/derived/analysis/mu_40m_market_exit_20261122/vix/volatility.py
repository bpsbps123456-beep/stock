"""Cboe 지수 종가와 CFE 표준 VX 최근월·차근월 정산가를 미국 주식 거래일에 맞춘다."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reentry"))
from inputs import LAST, study

START = "2025-06-02"
INDICES = ("VIX", "VIX9D", "VIX3M", "VVIX")
INDEX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
FUTURE_URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{}.csv"


def csv_from(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def index_close(symbol):
    source = csv_from(INDEX_URL.format(symbol))
    dates = pd.to_datetime(source.DATE, format="%m/%d/%Y")
    values = source[symbol if symbol == "VVIX" else "CLOSE"]
    return pd.Series(values.to_numpy(dtype=float), index=dates, name=symbol)


def monthly_expiry(month, sessions):
    """다음 달 제3 금요일(휴장 시 직전 영업일)의 30일 전, 휴장이면 다시 직전 영업일."""
    def preceding(day):
        return sessions[sessions.searchsorted(day, side="right") - 1]

    first = (month + 1).start_time
    third_friday = first + pd.Timedelta(days=(4 - first.dayofweek) % 7 + 14)
    return preceding(preceding(third_friday) - pd.Timedelta(days=30))


def settlement(expiry):
    raw = csv_from(FUTURE_URL.format(expiry.date()))
    # 주간물·미니·TAS를 제외하고 파일 안의 표준 월물 코드까지 확인한다.
    code = "FGHJKMNQUVXZ"[expiry.month - 1]
    label = f"{code} ({expiry.strftime('%b %Y')})"
    if not raw.Futures.eq(label).all():
        raise ValueError(f"{expiry.date()} 표준 월물 코드 불일치")
    values = raw.set_index(pd.to_datetime(raw["Trade Date"])).Settle
    if not values.index.is_unique:
        raise ValueError(f"{label} 정산일 중복")
    return values.rename(expiry)


def load_volatility():
    path = HERE / "market_vol.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date", "vx1_expiry", "vx2_expiry"]).set_index("date")
    sessions = study.market_calendar().schedule.index
    dates = sessions[(sessions >= START) & (sessions <= LAST)]
    months = pd.period_range(dates[0], dates[-1].to_period("M") + 2, freq="M")
    expiries = pd.DatetimeIndex([monthly_expiry(m, sessions) for m in months])
    with ThreadPoolExecutor(max_workers=4) as pool:
        indices = pd.concat(list(pool.map(index_close, INDICES)), axis=1)
        settles = pd.concat(list(pool.map(settlement, expiries)), axis=1)
    daily = indices.reindex(dates).copy()
    settles = settles.reindex(index=dates, columns=expiries)
    near = expiries.searchsorted(dates, side="right")  # 오전 만기일 오후에는 다음 계약이 최근월
    for rank in (1, 2):
        idx = near + rank - 1
        daily[f"vx{rank}_expiry"] = expiries[idx]
        daily[f"vx{rank}"] = settles.to_numpy()[np.arange(len(dates)), idx]
    if daily.isna().any().any() or daily[[*INDICES, "vx1", "vx2"]].le(0).any().any():
        raise ValueError(f"지수 또는 근월물 정산가 누락·오류\n{daily.loc[daily.isna().any(axis=1)]}")
    daily.rename_axis("date").to_csv(path, float_format="%.6f")
    return daily


def flags(data):
    vix = data.VIX
    out = pd.DataFrame(index=data.index)
    for threshold in (20, 25, 30):
        out[f"level{threshold}"] = vix >= threshold
    for threshold in (10, 20):
        out[f"jump{threshold}"] = vix >= vix.shift() * (1 + threshold / 100) - 1e-10
    out["rise5"] = vix >= 1.25 * vix.shift(5) - 1e-10
    out["rank80"] = vix >= vix.shift().rolling(63).quantile(.8)
    out["nine_day"] = data.VIX9D > vix
    out["three_month"] = vix > data.VIX3M
    out["futures"] = data.vx1 > data.vx2
    out["vvix_rank"] = data.VVIX >= data.VVIX.shift().rolling(63).quantile(.8)
    out["two_of_three"] = out[["level25", "jump10", "three_month"]].sum(axis=1) >= 2
    edges = np.select([vix >= 25, vix <= 20], [1., 0.], default=np.nan)
    out["band25_20"] = pd.Series(edges, index=data.index).ffill().fillna(0).eq(1)
    return out


if __name__ == "__main__":
    data = load_volatility()
    print(data.tail(1).to_string())
