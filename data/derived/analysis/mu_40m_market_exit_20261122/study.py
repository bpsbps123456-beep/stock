"""MU 40분봉 진입에 붙이는 반도체 시장 대피 규칙. 저장소 루트에서 py <이 파일>.

기간·규칙을 고정한 연구다. 결과를 정하기 전의 설계:
2025-10~12는 준비, 2026 1~5월 선택, 6~11/20 확인, 8/21~11/20 최근 분기.
MU와 SOXL 각각에 같은 40분봉 비교용 진입을 둔다. SOXX 시간봉 단독, 일봉
단독, 동시 약세 대피·동시 강세 해제를 현재 진입 신호와 비교한다. 차트용
새 진입식이 확정되기 전에 쓰는 대조이며 지표의 실제 체결 기록과 구별한다.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "hatta_bar" / "code"))
from desk.config import market_calendar
from fetch_alpaca import URL, keys

SYMBOLS = ("MU", "SOXL", "SOXX")
MODES = ("base", "hour", "day", "both")
WINDOWS = {
    "train": ("2026-01-02", "2026-05-29"),
    "forward": ("2026-06-01", "2026-11-20"),
    "recent": ("2026-08-21", "2026-11-20"),
    "since_oct15": ("2026-10-15", "2026-11-20"),
}
BAR = pd.Timedelta(minutes=10)


def download():
    cache = HERE / "cache" / "sip_10m_20251001_20261120.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    params = dict(symbols=",".join(SYMBOLS), timeframe="10Min", adjustment="split",
                  feed="sip", start="2025-10-01T00:00:00Z", end="2026-11-21T00:00:00Z",
                  limit=10000, sort="asc")
    rows = []
    with requests.Session() as session:
        session.headers.update(keys())
        while True:
            response = session.get(URL, params=params, timeout=45)
            response.raise_for_status()
            page = response.json()
            for symbol, bars in page["bars"].items():
                rows.append(pd.DataFrame(bars).assign(symbol=symbol))
            token = page.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
    data = pd.concat(rows, ignore_index=True)
    data["t"] = pd.to_datetime(data["t"], utc=True)
    data = data.set_index(["symbol", "t"])[["o", "c"]].sort_index()
    if not data.index.is_unique or (data <= 0).any().any():
        raise ValueError("SIP 봉 키·가격 오류")
    cache.parent.mkdir(exist_ok=True)
    data.to_parquet(cache)
    return data


def session_grid():
    schedule = market_calendar().schedule.loc["2025-10-01":"2026-11-20"]
    parts = []
    for row in schedule.itertuples():
        times = pd.date_range(row.open, row.close, freq=BAR, inclusive="left")
        parts.append(pd.DataFrame({"day": row.Index, "slot": np.arange(len(times)),
                                   "session_close": row.close}, index=times))
    return pd.concat(parts)


def confirmed_band(data, minutes):
    """EMA 10·20 양쪽을 모두 넘어야 방향 전환. 단축장 잔여 봉도 마감 때 확정."""
    bucket = data.slot // (minutes // 10) if minutes else np.zeros(len(data), dtype=int)
    grouped = data.groupby([data.day, bucket], sort=False)
    # 실제 입력이 빠진 10분칸이 하나라도 있으면 해당 상위 봉으로 신호를 갱신하지 않는다.
    close = grouped.c.last()
    valid = grouped.c.count().eq(grouped.size())
    end = grouped["end"].last()
    close = close[valid].set_axis(pd.DatetimeIndex(end[valid]))
    fast = close.ewm(span=10, adjust=False, min_periods=20).mean()
    slow = close.ewm(span=20, adjust=False, min_periods=20).mean()
    high = (close > fast) & (close > slow)
    low = (close < fast) & (close < slow)
    return pd.Series(np.select([high, low], [1., 0.], default=np.nan),
                     index=close.index).ffill()


def at_times(series, index):
    return series.reindex(index, method="ffill")


def coupled_gate(hour, day):
    both_up = hour.eq(1) & day.eq(1)
    both_down = hour.eq(0) & day.eq(0)
    return pd.Series(np.select([both_up, both_down], [1., 0.], default=np.nan),
                     index=hour.index).ffill().fillna(1)


def quote_grid(asset):
    opens = asset.o.dropna()
    last = asset[asset.end == asset.session_close].set_index("end").c
    if last.isna().any():
        raise ValueError("종가 평가에 필요한 마지막 10분봉 누락")
    marks = pd.concat([opens, last]).sort_index()
    return pd.DataFrame({"mark": marks, "is_open": marks.index.isin(opens.index)})


def with_premarket(tape, quotes):
    """같은 정규장 신호를 다음날 뉴욕 07시 이후 첫 관측 시가에도 실행하는 비교."""
    local = quotes.tz_convert("America/New_York").between_time("07:00", "09:30", inclusive="left")
    first = local.groupby(local.index.normalize()).head(1).tz_convert("UTC")
    first = first[first.index.normalize().isin(tape.index.normalize())]
    extra = first.o.to_frame("mark").assign(is_open=True)
    return pd.concat([tape, extra]).sort_index()


def prepare(inputs, symbol):
    asset = inputs[symbol]
    tape = quote_grid(asset)
    market = inputs["SOXX"]
    # 신호 시각의 합집합에서 상태를 정한다. 수면 중 발생한 재진입과 무효화도 순서대로 처리한다.
    entry = confirmed_band(asset, 40)
    hourly = confirmed_band(market, 60)
    daily = confirmed_band(market, None)
    timeline = entry.index.union(hourly.index).union(daily.index).sort_values()
    signals = pd.DataFrame({name: at_times(s, timeline) for name, s in
                            [("base", entry), ("hour", hourly), ("day", daily)]}).dropna()
    signals["both"] = coupled_gate(signals.hour, signals.day)
    # 1차 성적 확인 후의 진단 두 가지. 앞 구간 선택·이동 선택의 후보에는 넣지 않는다.
    own_hour = at_times(confirmed_band(asset, 60), signals.index)
    own_day = at_times(confirmed_band(asset, None), signals.index)
    signals["self_both"] = coupled_gate(own_hour, own_day)
    fresh = signals.base.eq(1) & signals.base.shift().eq(0)
    wanted = {"base": signals.base}
    for mode in (*MODES[1:], "self_both"):
        # 해제는 다음 신규 진입을 허용한다. 해제 자체로 오래된 신호를 되살리지 않는다.
        flat = signals.base.eq(0) | signals[mode].eq(0)
        order = pd.Series(np.select([flat, fresh], [0., 1.], default=np.nan),
                          index=signals.index)
        wanted[mode] = order.ffill().fillna(0)
    target = pd.DataFrame(wanted)
    return tape, target


def fills(tape, target, awake=True, delay=0):
    idx = tape.index
    kst = idx.tz_convert("Asia/Seoul")
    eligible = tape.is_open & ((kst.hour >= 8) if awake else True)
    cutoff = idx[eligible] - delay * BAR
    state = target.reindex(cutoff, method="ffill").fillna(0)
    state.index = idx[eligible]
    # 이미 체결된 보유만 유지한다. 수면 중에만 잠시 켜진 진입 신호는 보유가 아니다.
    entry_only = np.select([state.base.eq(0), state.both.eq(1)], [0., 1.], default=np.nan)
    state["entry_only"] = pd.Series(entry_only, index=state.index).ffill().fillna(0)
    return state.reindex(idx).ffill().fillna(0).astype(int)


def pnl(tape, positions, cost):
    # 시가 갭은 직전 보유에 귀속. 새 보유는 체결가 이후부터 적용한다.
    logret = np.log(tape.mark).diff().fillna(0)
    turnover = positions.diff().abs().fillna(positions)
    return positions.shift(fill_value=0).mul(logret, axis=0) + turnover.mul(np.log1p(-cost), axis=0)


def metrics(returns):
    curve = np.exp(returns.cumsum())
    return (100 * np.expm1(returns.sum()),
            100 * (curve / curve.cummax().clip(lower=1) - 1).min())


def matched_reference(tape, positions, returns, start, end):
    """진단용. 같은 창 평균 보유칸 비율만큼 기준 전략 일일 순수익을 고정 배분."""
    prev = positions.shift(fill_value=0).loc[start:end]
    ratio = prev.sum() / prev.base.sum() if prev.base.sum() else prev.sum() * 0.
    base_daily = np.expm1(returns.base.loc[start:end].groupby(
        tape.loc[start:end].index.normalize()).sum())
    return 100 * np.expm1(np.log1p(base_daily.to_numpy()[:, None] * ratio.to_numpy()).sum(axis=0))


def pick(returns):
    total, drawdown = metrics(returns.loc[:, list(MODES)])
    eligible = drawdown >= drawdown["base"] - 1e-10
    return total[eligible].idxmax()


def missed_runs(tape, positions, mode):
    """실제 두 모형의 보유 차이. 미완 회차 제외. 비교 단가는 두 실제 체결 시각의 시가."""
    exposed = positions.base.eq(1) & positions[mode].eq(0)
    starts = tape.index[exposed & ~exposed.shift(fill_value=False)]
    ends = tape.index[~exposed & exposed.shift(fill_value=False)]
    pairs = pd.DataFrame({"start": starts[:len(ends)], "end": ends})
    pairs["asset_return_pct"] = 100 * (
        tape.mark.loc[ends].to_numpy() / tape.mark.loc[pairs.start].to_numpy() - 1)
    return pairs


def walk_forward(tape, positions, returns):
    # 각 월 첫 개장에 지난 세 달의 선택을 적용한다. 월경계 포지션 차이의 비용도 다시 계산한다.
    choice = pd.Series("base", index=tape.index)
    rows = []
    for month in pd.period_range("2026-06", "2026-11", freq="M"):
        start = month.start_time.strftime("%Y-%m-%d")
        left = (month - 3).start_time.strftime("%Y-%m-%d")
        right = (month - 1).end_time.strftime("%Y-%m-%d")
        winner = pick(returns.loc[left:right])
        choice.loc[start:month.end_time.strftime("%Y-%m-%d")] = winner
        rows.append({"month": str(month), "mode": winner})
    selected = pd.Series(0, index=tape.index)
    for mode in MODES:
        use = choice.eq(mode)
        selected.loc[use] = positions.loc[use, mode]
    return selected, rows


def main():
    grid = session_grid()
    data = download()
    inputs = {}
    coverage = []
    for symbol in SYMBOLS:
        frame = grid.join(data.loc[symbol])
        frame["end"] = frame.index + BAR
        missing = int(frame.c.isna().sum())
        if not frame.o.notna().equals(frame.c.notna()) or missing > .005 * len(frame):
            raise ValueError(f"{symbol} 정규장 SIP 누락 {missing}/{len(frame)}")
        inputs[symbol] = frame
        coverage.append({"symbol": symbol, "expected": len(frame), "missing": missing,
                         "sessions": frame.day.nunique()})
    scores, lost, choices = [], [], []
    for symbol in ("MU", "SOXL"):
        regular, target = prepare(inputs, symbol)
        for label, awake, delay, cost in [("main", True, 0, .001), ("cost25", True, 0, .0025),
                                           ("delay10m", True, 1, .001), ("all_regular", False, 0, .001),
                                           ("premarket7", True, 0, .001)]:
            tape = regular
            if label == "premarket7":
                tape = with_premarket(regular, data.loc[symbol])
                cost = pd.Series(np.where(tape.index.isin(regular.index), cost, .003), index=tape.index)
            positions = fills(tape, target, awake, delay)
            returns = pnl(tape, positions, cost)
            if not positions.drop(columns="base").le(positions.base, axis=0).all().all():
                raise AssertionError("대피 모형이 원래 신호 밖에서 보유")
            if label == "main":
                selected, months = walk_forward(tape, positions, returns)
                positions["rolling3"] = selected
                returns = pnl(tape, positions, cost)
                for row in months:
                    choices.append(dict(symbol=symbol, **row))
                first = pick(returns.loc[slice(*WINDOWS["train"])])
                choices.append(dict(symbol=symbol, month="train_to_forward", mode=first))
                for mode in MODES[1:]:
                    lost.append(missed_runs(tape, positions, mode).assign(symbol=symbol, mode=mode))
            for name, (start, end) in WINDOWS.items():
                r = returns.loc[start:end]
                totals, drawdowns = metrics(r)
                trades = positions.diff().abs().loc[start:end].sum()
                reference = matched_reference(tape, positions, returns, start, end)
                exposure = positions.shift(fill_value=0).loc[start:end].mean() * 100
                for i, mode in enumerate(returns):
                    scores.append(dict(symbol=symbol, execution=label, window=name, mode=mode,
                                       return_pct=totals[mode], drawdown_pct=drawdowns[mode],
                                       switches=int(trades[mode]), exposure_pct=exposure[mode],
                                       exposure_control_pct=reference[i]))
    for name, result in [("scores", pd.DataFrame(scores)), ("choices", pd.DataFrame(choices)),
                         ("missed", pd.concat(lost)), ("coverage", pd.DataFrame(coverage))]:
        result.to_csv(HERE / f"{name}.csv", index=False, float_format="%.6f")
    print(pd.DataFrame(scores).query("execution == 'main' and window in ['forward', 'recent']")
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
