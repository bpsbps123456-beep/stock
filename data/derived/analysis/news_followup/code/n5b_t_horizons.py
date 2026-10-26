"""N5b — 사후 질문(2026-10-26): '레버리지 최고(T) 뉴스 때 사도 일정 기간 지나면 좋은 거 아니야?'

설명용이다(판정 없음). 계산 전에 정한 것
  - 묶음: n5 의 G4(family T), 진입 2024-01-02~2026-10-23. 같은 날 여러 은행 노트는 한 날짜로 센다(날짜 기준).
  - 보유: 20·40·60·120거래일(r_h = C[i+h−1]/O[i] − 1)과 2026-10-23 종가까지(now).
  - 보유 중 최저: 진입 뒤 그 기간 안 최저 종가 / O[i] − 1(0 이하).
  - 비교: 아무 날(같은 창, 기간이 끝난 날만), 같은 해(Y)·같은 해 + 낙폭 구간(A) 대조, 대조 차이의 날짜 묶음 부트스트랩·원형 이동(n5.stats).

py code/n5b_t_horizons.py → results/n5b_t_dates.csv · n5b_t_horizons.csv · n5b_tables.txt
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent.parent
RES = HERE / "results"
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("n5_run", HERE / "code" / "n5_run.py")
n5 = importlib.util.module_from_spec(_spec)
sys.modules["n5_run"] = n5
_spec.loader.exec_module(n5)

HS = (20, 40, 60, 120)
SEED = 20261028


def main():
    px = n5.load_prices()
    sess = px.index
    O = px["open"]["SOXX"].to_numpy(float)
    C = px["close"]["SOXX"].to_numpy(float)
    N = len(sess)
    sess_px, _, _, day, _, i0, D = n5.build_daily(px)
    assert sess_px.equals(sess)
    run_min = np.minimum.accumulate(C[::-1])[::-1]          # i 이후(끝까지) 최저 종가
    full = {}
    for h in HS:
        r = np.full(N, np.nan)
        r[:N - h + 1] = C[h - 1:] / O[:N - h + 1] - 1
        m = np.full(N, np.nan)
        for i in range(N - h + 1):
            m[i] = min(0.0, C[i:i + h].min() / O[i] - 1)
        full[f"r{h}"], full[f"low{h}"] = r, m
    full["now"] = C[-1] / O - 1
    full["low_now"] = np.minimum(0.0, run_min / O - 1)
    arr = {k: v[i0:i0 + D] for k, v in full.items()}

    sig = pd.read_csv(RES / "n5_signals.csv", encoding="utf-8-sig", parse_dates=["entry"])
    t = sig[sig.G4].copy()
    dates = (t.groupby("entry").agg(w=("w", "first"), banks=("bank", lambda x: "·".join(sorted(set(x)))),
                                   scope=("scope", "first"), n_notes=("group_id", "size")).reset_index())
    dates["semis_related"] = ~dates.scope.str.contains("헬스케어", na=False)
    for k, v in arr.items():
        dates[k] = v[dates.w.to_numpy(int)]
    dates["entry_open"] = O[i0 + dates.w.to_numpy(int)]
    dates.to_csv(RES / "n5b_t_dates.csv", index=False, encoding="utf-8-sig")

    def pbound(a, b):
        p = np.flatnonzero((day.index >= a) & (day.index <= b))
        return int(p[0]), int(p[-1])

    pb = pbound("2024-01-02", "2026-10-23")
    rng = np.random.default_rng(SEED)
    rows = []
    for label, pos in (("T 날짜 전부", dates.w.to_numpy(int)),
                       ("T 날짜(헬스케어 1건 제외)", dates[dates.semis_related].w.to_numpy(int))):
        for h in HS:
            y = arr[f"r{h}"]
            any_ok = ~np.isnan(y)
            base = dict(any_n=int(any_ok.sum()), any_win=float((y[any_ok] > 0).mean()), any_mean=float(y[any_ok].mean()),
                        any_low=float(np.nanmean(arr[f"low{h}"])))
            for cname, keys in (("Y", ["year"]), ("A", ["year", "b"])):
                cm, cu = n5.control(day, y, keys)
                r = n5.stats(np.sort(pos), y, cm, cu, pb, rng, h)
                lows = arr[f"low{h}"][pos]
                lows = lows[~np.isnan(y[pos])]
                r.update(group=label, h=h, control=cname, low_mean=float(lows.mean()) if len(lows) else np.nan,
                         low_worst=float(lows.min()) if len(lows) else np.nan, **base)
                rows.append(r)
    hz = pd.DataFrame(rows)
    front = ["group", "h", "control", "n", "clusters", "win", "mean", "median", "worst", "best", "low_mean", "low_worst",
             "any_n", "any_win", "any_mean", "any_low", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_two"]
    hz = hz[front + [c for c in hz.columns if c not in front]]
    hz.to_csv(RES / "n5b_t_horizons.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = dates[["entry", "banks", "n_notes", "semis_related", "entry_open", "r20", "r40", "r60", "r120", "now", "low_now"]].copy()
    show["entry"] = show.entry.dt.strftime("%Y-%m-%d")
    L = [f"N5b T 보유 기간별  창 2024-01-02~2026-10-23  마지막 종가 {sess[-1].date()} {C[-1]:.2f}",
         "== 날짜별 ==",
         show.round(4).to_string(index=False), "",
         "== 기간별 요약(대조 A 행이 주; Y 는 같은 해 모든 날) ==",
         hz[front].round(4).to_string(index=False)]
    txt = "\n".join(L)
    (RES / "n5b_tables.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
