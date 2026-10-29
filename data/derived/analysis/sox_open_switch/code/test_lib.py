"""엔진 체결 규칙 점검(합성 자료). 사용: py test_lib.py"""
import datetime as dt
import numpy as np
import pandas as pd
import lib as L


def fake(days=(dt.date(2026, 7, 1), dt.date(2026, 1, 5))):
    D = L.Data.__new__(L.Data)
    D.days = list(days)
    D.pos = {d: k for k, d in enumerate(D.days)}
    D.A, D.has = {}, {}
    for s in L.SYMS:
        base = np.full((len(days), L.NMIN), 100.0)
        D.A[s] = {f: base.copy() for f in L.FIELDS}
        D.A[s]["Volume"][:] = 1000
        D.has[s] = np.ones_like(base, bool)
    D.daily = pd.DataFrame()
    return D


ZERO = {s: {k: 0.0 for k in ["pre", "open5", "early", "late"]} for s in ["SOXL", "SOXS"]}


class Once:
    """09:31 봉 마감에 한 번 주문."""
    def __init__(self, order, at="09:31"):
        self.order, self.at = order, L.idx(at)

    def on_bar(self, ctx, i):
        return self.order if i == self.at else None


def setp(D, sym, d, i, o=None, h=None, l=None, c=None):
    A = D.A[sym]
    for k, v in (("Open", o), ("High", h), ("Low", l), ("Close", c)):
        if v is not None:
            A[k][d, i] = v


def test_cutoff():
    assert L.cutoff_idx(dt.date(2026, 10, 28)) == L.idx("11:00")
    assert L.cutoff_idx(dt.date(2026, 11, 2)) == L.idx("10:00")
    assert L.cutoff_idx(dt.date(2026, 1, 5)) == L.idx("10:00")
    assert L.cutoff_idx(dt.date(2026, 7, 1), "std") == L.idx("10:00")


def test_delay_fill_and_cutoff_exit():
    D = fake()
    i = L.idx("09:31")
    setp(D, "SOXL", 0, i + 1, c=101.0)                  # 다음 봉 종가 체결
    D.A["SOXL"]["Close"][0, L.idx("10:59")] = 103.02    # 마감 1분 전 봉 종가로 청산
    E = L.Engine(D, costs=ZERO)
    r, t = E.run_day(Once(L.Order(1)), 0)
    assert t[0][2] == "09:32" and t[0][3] == "10:59" and t[0][7] == "cutoff", t
    assert abs(r - (103.02 / 101 - 1)) < 1e-12


def test_stop_first_and_gap_fill():
    D = fake()
    i = L.idx("09:31")
    setp(D, "SOXS", 0, i + 2, o=98.0, h=102.0, l=97.0)  # 같은 봉 목표·손절 → 손절, 시가가 손절선 아래면 시가
    E = L.Engine(D, costs=ZERO)
    r, t = E.run_day(Once(L.Order(-1, stop=0.01, target=0.01)), 0)
    assert t[0][7] == "stop" and abs(r - (98.0 / 100 - 1)) < 1e-12, t


def test_target_touch():
    D = fake()
    i = L.idx("09:31")
    setp(D, "SOXL", 0, i + 1, h=105.0)                  # 체결 봉 고가는 세지 않는다
    setp(D, "SOXL", 0, i + 3, h=101.0)                  # 1.0002 관통 부족
    setp(D, "SOXL", 0, i + 4, h=101.03)
    E = L.Engine(D, costs=ZERO)
    r, t = E.run_day(Once(L.Order(1, target=0.01)), 0)
    assert t[0][7] == "target" and t[0][3] == L.hm(i + 4) and abs(r - 0.01) < 1e-12, t


def test_costs_and_switch():
    D = fake()
    i = L.idx("09:31")
    costs = {s: {k: 0.001 for k in ["pre", "open5", "early", "late"]} for s in ["SOXL", "SOXS"]}

    class Sw:
        def on_bar(self, ctx, j):
            if j == i:
                return L.Order(1)
            if j == i + 5:
                return L.Order(-1)
    E = L.Engine(D, costs=costs)
    r, t = E.run_day(Sw(), 0)
    assert len(t) == 2 and t[0][1] == "SOXL" and t[1][1] == "SOXS"
    assert abs(t[0][6] - (0.999 / 1.001 - 1)) < 1e-12


def test_keep_after_stop_does_not_reopen():
    D = fake()
    i = L.idx("09:31")
    setp(D, "SOXL", 0, i + 2, l=98.0)

    class K:
        def on_bar(self, ctx, j):
            if j == i:
                return L.Order(1, stop=0.01)
            if j == i + 1:
                return L.Order(1, stop=0.03, keep=True)   # 체결은 i+2, 그 봉에서 이미 손절
    E = L.Engine(D, costs=ZERO)
    r, t = E.run_day(K(), 0)
    assert len(t) == 1 and t[0][7] == "stop", t


def test_no_future_in_ctx():
    D = fake()
    seen = []

    class S:
        def on_bar(self, ctx, j):
            seen.append(len(ctx.c("SOXL")) == j + 1)
    L.Engine(D, costs=ZERO).run_day(S(), 0)
    assert all(seen) and len(seen) > 0


def test_locked_period():
    D = fake()
    try:
        L.Engine(D, costs=ZERO).run(lambda: Once(L.Order(1)), period="conf")
        assert False
    except PermissionError:
        pass


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
