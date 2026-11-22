"""SOXX 킬스위치 대비 VIX의 추가 효과. 저장소 루트에서 이 파일을 실행한다."""
from typing import NamedTuple

import numpy as np
import pandas as pd

from volatility import HERE, flags, load_volatility
import study_reentry as prior
from inputs import ORIGINAL, replay_panel

ANCHOR = "soxx"
ACTIONS = ("block", "clear2", "confirm_exit", "panic_exit", "both_block", "vix_only")
COST = prior.COST
WINDOWS = prior.WINDOWS


class Orders(NamedTuple):
    allow_new: np.ndarray
    force_exit: np.ndarray


def orders(name, sector, stress, yesterday):
    off = np.zeros(len(sector), dtype=bool)
    if name == ANCHOR:
        return Orders(sector, off)
    if name == "original":
        return Orders(~off, off)
    if name == "soxx_exit":
        return Orders(sector, ~sector)
    action, signal = name.split("__")
    alert = stress[signal].to_numpy()
    if action == "block":
        return Orders(sector & ~alert, off)
    if action == "clear2":
        return Orders(sector & ~alert & ~yesterday[signal].to_numpy(), off)
    if action == "confirm_exit":
        return Orders(sector, ~sector & alert)
    if action == "panic_exit":
        return Orders(sector & ~alert, alert)
    if action == "both_block":
        return Orders(sector | ~alert, off)
    if action == "vix_only":
        return Orders(~alert, off)
    raise ValueError(name)


def switched(first, second, mask):
    return Orders(np.where(mask, second.allow_new, first.allow_new),
                  np.where(mask, second.force_exit, first.force_exit))


class Experiment:
    def __init__(self):
        panel = replay_panel()
        close = panel.pivot(columns="symbol", values="close")
        self.dates, self.symbols = close.index, close.columns
        self.close = close.to_numpy()
        self.open = panel.pivot(columns="symbol", values="open").to_numpy()
        self.original = panel.pivot(columns="symbol", values="original_at_open").to_numpy(bool)
        self.fresh = self.original & ~self.previous(self.original)
        self.groups = {"original9": self.symbols.isin(ORIGINAL), "extra10": ~self.symbols.isin(ORIGINAL)}
        market = prior.market_at_closes(self.dates)
        self.sector = market.both.shift(1, fill_value=0).to_numpy(bool)
        self.vol = load_volatility()
        if (len(self.vol) != len(self.vol.index.unique()) or not self.vol.index.is_monotonic_increasing
                or not self.dates.isin(self.vol.index).all() or self.vol.isna().any().any()
                or not ((self.vol.index < self.vol.vx1_expiry)
                        & (self.vol.vx1_expiry < self.vol.vx2_expiry)).all()):
            raise ValueError("Cboe 공통 거래일·만기 입력 오류")
        self.warnings = flags(self.vol)
        self.live, self.earlier = self.shifted_flags(0)
        self.names = [f"{action}__{flag}" for action in ACTIONS for flag in self.live]
        self.candidates = [ANCHOR] + [n for n in self.names if n.split("__")[0] in ACTIONS[:4]]
        self.controls = [ANCHOR, "original", "soxx_exit"]
        self.static = {n: orders(n, self.sector, self.live, self.earlier) for n in self.controls + self.names}
        self.paths = {n: self.hold(rule) for n, rule in self.static.items()}
        self.legs = {n: self.pnl(path) for n, path in self.paths.items()}
        for new, old in [(ANCHOR, "fresh1"), ("original", "base"), ("soxx_exit", "exit_both")]:
            expected = prior.positions(self.original, market, [old] * len(self.dates))
            if not np.array_equal(self.paths[new], expected):
                raise ValueError(f"{old} 기준 보유 불일치")

    @staticmethod
    def previous(held):
        return np.vstack((np.zeros(held.shape[1], dtype=bool), held[:-1]))

    def shifted_flags(self, extra_lag):
        # VIX 종가는 주식 정규장 종가 이후에도 바뀐다. 미국 거래일 t 값은 t+1 개장부터 사용한다.
        frames = [self.warnings.shift(n + extra_lag).loc[self.dates] for n in (1, 2)]
        if any(frame.isna().any().any() for frame in frames):
            raise ValueError("다음 시가에 쓸 VIX 정보 누락")
        return tuple(frame.astype(bool) for frame in frames)

    def hold(self, rule):
        held = np.zeros(self.original.shape[1], dtype=bool)
        path = np.empty_like(self.original)
        for i, (can_enter, must_exit) in enumerate(zip(*rule)):
            held = self.original[i] & ~must_exit & (held | (can_enter & self.fresh[i]))
            path[i] = held
        return path

    def pnl(self, held, cost=COST):
        legs = prior.return_legs(held, self.open, self.close)
        if cost != COST:
            legs[:, 0] -= (cost - COST) * (held != self.previous(held))
        return legs

    def period(self, left, right):
        return prior.window(self.dates, left, right)

    def pick(self, period):
        sample = self.groups["original9"]
        base_ret, base_dd = prior.score(self.legs[ANCHOR], period)
        best, best_gain = ANCHOR, 0.
        for name in self.candidates[1:]:
            diff = (self.paths[name][:, sample] != self.paths[ANCHOR][:, sample]).any(axis=1)
            starts = diff & ~np.r_[False, diff[:-1]]
            if (starts & period).sum() < 3:
                continue
            ret, dd = prior.score(self.legs[name], period)
            gain = round(np.median((ret - base_ret)[sample]), 8)
            if np.median((dd - base_dd)[sample]) >= -1e-8 and gain > best_gain:
                best, best_gain = name, gain
        return best

    def choose(self):
        left, right = WINDOWS["train"]
        self.winner = self.pick(self.period(left, right))
        forward = self.period(*WINDOWS["forward"])
        self.active_rules = {"fixed": switched(self.static[ANCHOR], self.static[self.winner], forward)}
        rule = self.static[ANCHOR]
        self.months = []
        for month in pd.period_range("2026-06", "2026-11", freq="M"):
            earlier, last = (month-3).start_time, (month-1).end_time
            name = self.pick(self.period(earlier, last))
            mask = self.period(month.start_time, month.end_time)
            rule = switched(rule, self.static[name], mask)
            self.months.append((month, name))
        self.active_rules["rolling"] = rule
        for name, order in self.active_rules.items():
            self.paths[name] = self.hold(order)
            self.legs[name] = self.pnl(self.paths[name])
        decisions = [dict(applied="fixed", left=left, right=right, rule=self.winner)]
        decisions += [dict(applied=str(m), left=(m-3).start_time.date(), right=(m-1).end_time.date(), rule=n)
                      for m, n in self.months]
        return pd.DataFrame(decisions)

    def scores(self):
        rows = []
        base = self.paths[ANCHOR]
        daily = np.expm1(self.legs[ANCHOR].sum(axis=1))
        for label, edges in WINDOWS.items():
            window = self.period(*edges)
            base_ret, base_dd = prior.score(self.legs[ANCHOR], window)
            base_exposure = base[window].sum(axis=0) + self.previous(base)[window].sum(axis=0)
            for name, values in self.legs.items():
                if label == "train" and name in self.active_rules:
                    continue
                ret, dd = prior.score(values, window)
                pos = self.paths[name]
                prev = self.previous(pos)
                total = pos[window].sum(axis=0) + prev[window].sum(axis=0)
                # 같은 평균 노출을 갖는 기준의 일일 수익. 진단 전용이며 재조정 추가 비용을 빼지 않는다.
                ratio = np.divide(total, base_exposure, where=base_exposure > 0,
                                  out=np.full(len(self.symbols), np.nan))
                ratio[(base_exposure == 0) & (total == 0)] = 0.
                # 더 자주 보유한 확장 대조는 현금 조절만으로 같은 노출을 만들 수 없다.
                ratio[ratio > 1 + 1e-10] = np.nan
                matched = 100 * np.expm1(np.log1p(daily[window] * ratio).sum(axis=0))
                for i, symbol in enumerate(self.symbols):
                    rows.append(dict(window=label, rule=name, symbol=symbol,
                                     cohort="original9" if symbol in ORIGINAL else "extra10",
                                     return_pct=ret[i], drawdown_pct=dd[i], gain_pp=ret[i]-base_ret[i],
                                     dd_gain_pp=dd[i]-base_dd[i], gain_vs_matched_pp=ret[i]-matched[i],
                                     entries=int((pos & ~prev)[window, i].sum()), held_days=int(pos[window, i].sum())))
        return pd.DataFrame(rows)

    def intervals(self):
        rows = []
        base = self.paths[ANCHOR]
        for rule, pos in self.paths.items():
            if rule in self.controls:
                continue
            for col, symbol in enumerate(self.symbols):
                for skipped in (True, False):
                    diff = (base[:, col] & ~pos[:, col]) if skipped else (pos[:, col] & ~base[:, col])
                    starts = np.flatnonzero(diff & ~np.r_[False, diff[:-1]])
                    ends = np.flatnonzero(~diff & np.r_[False, diff[:-1]])
                    for start in starts:
                        if self.dates[start] < pd.Timestamp(WINDOWS["train"][0]):
                            continue
                        j = ends.searchsorted(start)
                        closed = j < len(ends)
                        end = ends[j] if closed else len(pos) - 1
                        price = self.open[end, col] if closed else self.close[end, col]
                        action = "extra" if not skipped else ("block" if self.fresh[start, col] else "exit")
                        ratio = price / self.open[start, col]
                        full_trade = (action == "block" and closed and not base[end, col])
                        rows.append(dict(rule=rule, symbol=symbol, entry=self.dates[start].date(),
                                         end=self.dates[end].date(), signal_close=self.dates[start-1].date(),
                                         closed=closed, action=action,
                                         price_return_pct=100*(ratio-1),
                                         baseline_trade_net_pct=100*(ratio*np.exp(-2*COST)-1) if full_trade else np.nan))
        return pd.DataFrame(rows)

    def policy_before_open(self, name, live, earlier):
        base = self.static[ANCHOR]
        if name == "rolling":
            rule = base
            for month, choice in self.months:
                mask = self.period(month.start_time, month.end_time)
                rule = switched(rule, orders(choice, self.sector, live, earlier), mask)
            return rule
        choice = self.winner if name == "fixed" else name
        return switched(base, orders(choice, self.sector, live, earlier), self.period(*WINDOWS["forward"]))

    def diagnostics(self):
        examples = ["fixed", "rolling", "block__level25", "block__three_month",
                    "block__futures", "confirm_exit__three_month"]
        rows, shifts = [], []
        forward = self.period(*WINDOWS["forward"])
        for lag, cost in ((0, COST), (0, 0.), (0, .0025), (1, COST)):
            live, earlier = self.shifted_flags(lag)
            ref = self.pnl(self.paths[ANCHOR], cost)
            for name in examples:
                hold = self.hold(self.policy_before_open(name, live, earlier))
                values = self.pnl(hold, cost)
                for w in ("forward", "quarter"):
                    period = self.period(*WINDOWS[w])
                    ret, dd = prior.score(values, period)
                    br, bd = prior.score(ref, period)
                    for group, sample in self.groups.items():
                        rows.append(dict(rule=name, window=w, cohort=group, extra_lag=lag, cost=cost,
                                         gain_pp=np.median((ret-br)[sample]), dd_gain_pp=np.median((dd-bd)[sample])))
        # 실제·위약 모두 6월 시작 시 기준 보유를 승계한다. SOXX와 종목 가격은 날짜 그대로다.
        base_ret, _ = prior.score(self.legs[ANCHOR], forward)
        for offset in range(forward.sum()):
            live, earlier = self.live.copy(), self.earlier.copy()
            for f in (live, earlier):
                f.loc[forward] = np.roll(f.loc[forward].to_numpy(), offset, axis=0)
            for name in [n for n in examples if n != "rolling"]:
                held = self.hold(self.policy_before_open(name, live, earlier))
                ret, _ = prior.score(self.pnl(held), forward)
                for group, mask in self.groups.items():
                    shifts.append(dict(rule=name, offset=offset, cohort=group, gain_pp=np.median((ret-base_ret)[mask])))
        return pd.DataFrame(rows), pd.DataFrame(shifts)

    def common_start(self):
        """6월 직전 보유를 모두 같게 맞춘 부가 대조. 집단 합산 위험은 기간 첫날 같은 금액으로 시작한다."""
        after = self.period(*WINDOWS["forward"])
        records = []
        for name in [*self.controls, *self.names, "fixed", "rolling"]:
            if name in self.active_rules:
                rule = self.active_rules[name]
            else:
                rule = switched(self.static[ANCHOR], self.static[name], after)
            legs = self.pnl(self.hold(rule))
            for window in ("forward", "quarter", "oct15"):
                mask = self.period(*WINDOWS[window])
                ret, dd = prior.score(legs, mask)
                ref_ret, ref_dd = prior.score(self.legs[ANCHOR], mask)
                for group, cols in self.groups.items():
                    selected = legs[mask][:, :, cols].reshape(-1, cols.sum())
                    equity = np.r_[1., np.exp(selected.cumsum(axis=0)).mean(axis=1)]
                    closes = equity[::2]
                    daily = closes[1:] / closes[:-1] - 1
                    tail = np.sort(daily)[:int(np.ceil(.05*len(daily)))]
                    records.append(dict(rule=name, window=window, cohort=group, ret=np.median(ret[cols]),
                                        dd=np.median(dd[cols]), paired_gain=np.median((ret-ref_ret)[cols]),
                                        paired_dd_gain=np.median((dd-ref_dd)[cols]),
                                        combined_return=100*(equity[-1]-1),
                                        combined_dd=100*(equity/np.maximum.accumulate(equity)-1).min(),
                                        worst_day_pct=100*daily.min(), worst5pct_avg=100*tail.mean()))
        return pd.DataFrame(records)

    def overlap(self):
        rows = []
        for window, edges in WINDOWS.items():
            period = self.period(*edges)
            for name in self.live:
                active = self.live[name].to_numpy() & period
                dates = active & self.sector
                fresh = self.fresh[dates]
                rows.append(dict(window=window, signal=name, sessions=int(period.sum()),
                                 alert=int(active.sum()), already_blocked=int((active & ~self.sector).sum()),
                                 soxx_open=int(dates.sum()), affected_fresh_entries=int(fresh.sum()),
                                 entry_dates=int(fresh.any(axis=1).sum())))
        return pd.DataFrame(rows)

    def nearby_thresholds(self):
        """사후 민감도. 상대 경보의 창과 백분위만 움직이고 전방·이동 선택에 되먹이지 않는다."""
        after = self.period(*WINDOWS["forward"])
        mu = self.symbols.get_loc("MU")
        records = []
        for symbol, action in (("VIX", "block"), ("VVIX", "clear2")):
            series = self.vol[symbol]
            for length in (42, 63, 84):
                for percentile in (.75, .80, .85):
                    bound = series.shift().rolling(length).quantile(percentile)
                    flags_at_close = (series >= bound).to_frame("rank")
                    first, second = [flags_at_close.shift(n).loc[self.dates].astype(bool) for n in (1, 2)]
                    rule = orders(action+"__rank", self.sector, first, second)
                    held = self.hold(switched(self.static[ANCHOR], rule, after))
                    values = self.pnl(held)
                    for w in ("forward", "quarter"):
                        mask = self.period(*WINDOWS[w])
                        returns, dd = prior.score(values, mask)
                        ref, ref_dd = prior.score(self.legs[ANCHOR], mask)
                        for group, cols in self.groups.items():
                            records.append(dict(index=symbol, lookback=length, percentile=percentile,
                                                action=action, window=w, cohort=group,
                                                kept_mu_sep18=held[self.dates.get_loc("2026-09-18"), mu],
                                                blocked_mu_oct29=not held[self.dates.get_loc("2026-10-29"), mu],
                                                paired_gain=np.median((returns-ref)[cols]),
                                                paired_dd_gain=np.median((dd-ref_dd)[cols])))
        return pd.DataFrame(records)


def save(name, table):
    table.to_csv(HERE / name, index=False, float_format="%.6f")


def main():
    exp = Experiment()
    decisions = exp.choose()
    scores = exp.scores()
    summary = scores.groupby(["window", "rule", "cohort"], sort=False).agg(
        ret=("return_pct", "median"), dd=("drawdown_pct", "median"),
        paired_gain=("gain_pp", "median"), paired_dd_gain=("dd_gain_pp", "median"),
        matched_gain=("gain_vs_matched_pp", "median"), better=("gain_pp", lambda s: (s > 1e-8).sum()),
        worse=("gain_pp", lambda s: (s < -1e-8).sum()), held_days=("held_days", "median")).reset_index()
    sensitivity, phase = exp.diagnostics()
    for name, table in [("choices.csv", decisions), ("scores.csv", scores), ("summary.csv", summary),
                        ("intervals.csv", exp.intervals()), ("sensitivity.csv", sensitivity),
                        ("phase_shift.csv", phase), ("common_start.csv", exp.common_start()),
                        ("signal_overlap.csv", exp.overlap()), ("nearby.csv", exp.nearby_thresholds())]:
        save(name, table)
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
