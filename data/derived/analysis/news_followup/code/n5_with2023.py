"""N5 2023 포함 확장 — BRIEF2.md 정의 그대로, 평가 창만 2023-01-03 부터로 넓힌다(사전 등록 뒤 표본 확장, 2026-10-28).

py code/n5_with2023.py   → results/n5_with2023_{signals,cells,g5,audit,audit_hand}.csv · n5_with2023_verdict.json · n5_with2023_tables.txt

왜 따로 있나
  n5_run.py 는 클라우드 세션(Yahoo 차단)에서 Codex 10/25 스냅샷(2023-12-01~)만으로 돌려 2023 노트 51개를 뺐다.
  로컬 cache/n1_raw.parquet(n1_fetch.py, yfinance 분할 반영·배당 미포함 = Codex 와 같은 기준)에 2022-06~ SOXX 시가·종가와
  ^TNX 종가가 있어 그 51개를 넣는다. n5_run.py 의 주 결과(2024~2026)는 그대로 두고 이 파일은 새 결과 파일만 쓴다.

바꾼 것(그 외는 n5_run.py 함수를 그대로 부른다)
  가격   n1_raw.parquet 2022-06-01~2026-10-23(앞 20일 낙폭 구간·5일 M 계산용 여유). 겹치는 날은 Codex 스냅샷과 대조한다.
  창     진입 2023-01-03~2026-10-23. 기간 줄에 '2023~2026'(확장 합산)·'2023' 추가.
  판정   등록 판정은 2024~2026(재현). 같은 규칙을 2023~2026 에 적용한 줄은 '사후 확장'으로만 표시한다.
  G5     n4h5_soxx_events 의 2023 확인 3건을 넣는다.
한계(좌측 절단)
  전수 노트의 첫 진입이 2023-02-07 이라, 2023 초 신호의 '앞 20·30일 D·T 노트'와 G5 2023-01-10 의 'R 사전'은 2022 노트가 없어 거짓 쪽으로 치우친다.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
HERE = pathlib.Path(__file__).resolve().parent.parent
RES = HERE / "results"
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("n5_run", HERE / "code" / "n5_run.py")
n5 = importlib.util.module_from_spec(_spec)
sys.modules["n5_run"] = n5
_spec.loader.exec_module(n5)
h14 = n5.h14

W0, W1 = pd.Timestamp("2023-01-03"), pd.Timestamp("2026-10-23")
n5.W0 = W0                                   # build_daily·load_signals 가 모듈 전역 창을 읽는다
PERIODS = [("2023~2026", "2023-01-03", "2026-10-23")] + n5.PERIODS + [("2023", "2023-01-03", "2023-12-29")]
OUT = "n5_with2023"


def load_prices() -> pd.DataFrame:
    raw = pd.read_parquet(HERE / "cache" / "n1_raw.parquet")
    raw = raw[raw.sym.isin(["SOXX", "^TNX"]) & (raw.date >= "2022-06-01") & (raw.date <= W1)]
    o = raw.pivot(index="date", columns="sym", values="O")
    c = raw.pivot(index="date", columns="sym", values="C")
    px = pd.concat({"open": o[["SOXX"]], "close": c[["SOXX", "^TNX"]]}, axis=1).sort_index()
    px.index = pd.DatetimeIndex(px.index).astype("datetime64[ns]")
    assert px.index.equals(n5.xnys("2022-06-01", W1)), "XNYS 세션과 다름"
    assert px.notna().all().all() and (px > 0).all().all()
    return px


def price_audit(px: pd.DataFrame) -> list[dict]:
    """겹치는 날(2023-12-01~2026-10-23) Codex 스냅샷과 같은 축인지."""
    cx = pd.read_csv(n5.CODEX / "prices.csv", header=[0, 1], index_col=0)
    cx.index = pd.to_datetime(cx.index, errors="coerce")
    cx = cx[cx.index.notna()].dropna().sort_index()
    out = []
    for fld, sym in (("open", "SOXX"), ("close", "SOXX"), ("close", "^TNX")):
        a = px[fld][sym].reindex(cx.index)
        b = cx[fld][sym]
        rel = (a / b - 1).abs()
        out.append(dict(check=f"가격 대조 {fld} {sym}(Codex 스냅샷 겹침)", n=int(rel.notna().sum()),
                        nan_mismatch=int(a.isna().sum()), max_abs_diff=float(rel.max())))
    return out


def main():
    px = load_prices()
    sess, O, C, day, arr, i0, D = n5.build_daily(px)
    assert day.index[0] == W0 and day.index[-1] == W1
    ctl = {}
    for k in ("r5", "r10", "r20", "o21", "mae20"):
        for name, keys in (("Y", ["year"]), ("A", ["year", "b"]), ("AM", ["year", "b", "M"])):
            ctl[(k, name)] = n5.control(day, arr[k], keys)
    sig, win = n5.load_signals(sess, i0)
    assert len(win) == len(sig) == 245 and (win.entry.dt.year == 2023).sum() == 51
    wpos = win.w.to_numpy(int)
    for k in ("r5", "r20", "o21", "mae20"):
        win[k] = arr[k][wpos]
    win["M"] = day.M.to_numpy()[wpos]
    win["bucket"] = day.b.to_numpy()[wpos]
    for k in ("r5", "r20", "mae20"):
        win[f"A_{k}"] = ctl[(k, "A")][0][wpos]
        win[f"Y_{k}"] = ctl[(k, "Y")][0][wpos]
    win["AM_r5"] = ctl[("r5", "AM")][0][wpos]
    win["AM_r20"] = ctl[("r20", "AM")][0][wpos]

    # ------------------------------------------- 점검 1: 가격 축 2: n4h14 재현(2023 포함 245개) 3: 손 계산(2023 5건)
    aud = price_audit(px)
    for h in (5, 10, 20):
        for lab, new, old in ((f"r{h}", arr[f"r{h}"][wpos], win[f"SOXX_r{h}"].to_numpy(float)),
                              (f"대조 A{h}", ctl[(f"r{h}", "A")][0][wpos], win[f"SOXX_A{h}"].to_numpy(float))):
            both = ~np.isnan(new) & ~np.isnan(old)
            aud.append(dict(check=f"{lab} 재현(2023~2026 신호, n4h14)", n=int(both.sum()),
                            nan_mismatch=int((np.isnan(new) != np.isnan(old)).sum()),
                            max_abs_diff=float(np.max(np.abs(new[both] - old[both])))))
    hand = []
    rs = np.random.default_rng(20261028)
    w23 = win[(win.entry.dt.year == 2023) & win.r20.notna()]
    for gid in rs.choice(w23.group_id.to_numpy(), 5, replace=False):
        x = win[win.group_id == gid].iloc[0]
        i = int(x.i)
        o, c19, cmin, o20 = O[i], C[i + 19], C[i:i + 20].min(), O[i + 20]
        hand.append(dict(group_id=gid, entry=x.entry.date().isoformat(), exit_close_date=sess[i + 19].date().isoformat(),
                         open_entry=o, close_20th=c19, min_close=cmin, open_21st=o20,
                         r20_hand=c19 / o - 1, r20_code=x.r20, mae20_hand=min(0, cmin / o - 1), mae20_code=x.mae20,
                         o21_hand=o20 / o - 1, o21_code=x.o21))
    hand = pd.DataFrame(hand)
    for a, b in (("r20_hand", "r20_code"), ("mae20_hand", "mae20_code"), ("o21_hand", "o21_code")):
        aud.append(dict(check=f"손 계산 {a[:-5]} 2023 5건", n=5, nan_mismatch=0,
                        max_abs_diff=float(np.nanmax(np.abs(hand[a] - hand[b])))))

    # ------------------------------------------- 묶음(n5_run.py 와 같은 정의)
    f, lb, sell = win.family, win.lookback_months, win.sell_note.astype(bool)
    A_GROUPS = {
        "G1 기록적 매도(D, 1년+)": (f == "D") & (lb >= 12),
        "G2 매수 복귀·숏 청산(R)": f == "R",
        "G3 극단 매도(5년+)": sell & (lb >= 60),
        "G3b 매도 1~5년": sell & (lb >= 12) & (lb < 60),
        "G3c 매도 1년 미만": sell & (lb < 12),
        "G4 레버리지 쏠림(T)": f == "T",
        "참고 S 숏 쏠림": f == "S",
    }
    C_GROUPS = {
        "C1 (D·S) & 주가·금리 동반 하락": (f.isin(["D", "S"]) & win.M, -1, ("r5", "r20"), "AM"),
        "C1o (D·S) & 동반 하락 아님": (f.isin(["D", "S"]) & ~win.M, 0, ("r5", "r20"), "AM"),
        "C2 R & 앞 20일 D": ((f == "R") & win.prior_D20, +1, ("r20",), "A"),
        "C2o R & 앞 20일 D 없음": ((f == "R") & ~win.prior_D20, 0, ("r20",), "A"),
        "C3 D & 앞 30일 T": ((f == "D") & win.prior_T30, -1, ("r20",), "A"),
        "C3o D & 앞 30일 T 없음": ((f == "D") & ~win.prior_T30, 0, ("r20",), "A"),
    }
    for name, m in A_GROUPS.items():
        win[name.split()[0]] = m.to_numpy()
    for name, (m, *_r) in C_GROUPS.items():
        win[name.split()[0]] = m.to_numpy()
    ep_pos = np.array([int(np.flatnonzero(day.index == pd.Timestamp(d))[0]) for d in n5.OVERLAP4])
    win.sort_values(["entry", "group_id"]).drop(columns=[c for c in win.columns if c.startswith("SOXX_")] + ["entry_n4"]).to_csv(
        RES / f"{OUT}_signals.csv", index=False, encoding="utf-8-sig")

    def pbound(a, b):
        p = np.flatnonzero((day.index >= a) & (day.index <= b))
        return int(p[0]), int(p[-1])

    pbs = {p: pbound(a, b) for p, a, b in PERIODS}
    rng = np.random.default_rng(n5.SEED)
    rows = []
    for gname, m in A_GROUPS.items():
        g = win[m.to_numpy()]
        for vname in ("all", "first"):
            pos_all = n5.variant_pos(g, vname)
            for pname, *_ in PERIODS:
                p0, p1 = pbs[pname]
                pos = pos_all[(pos_all >= p0) & (pos_all <= p1)]
                for k, cname in (("r20", "A"), ("r20", "Y"), ("o21", "A"), ("mae20", "A"), ("mae20", "Y")):
                    cm, cu = ctl[(k, cname)]
                    r = n5.stats(pos, arr[k], cm, cu, (p0, p1), rng, 20)
                    r.update(part="A", group=gname, variant=vname, period=pname, outcome=k, control=cname, direction=0,
                             verdict="", verdict_kind="")
                    rows.append(r)
    for gname, (m, d, outs, cname) in C_GROUPS.items():
        g = win[m.to_numpy()]
        vlist = ("all", "first", "new", "new_episode") if gname.startswith("C1") else ("all", "first", "new")
        for vname in vlist:
            pos_all = n5.variant_pos(g, vname, ep_pos)
            for pname, *_ in PERIODS:
                p0, p1 = pbs[pname]
                pos = pos_all[(pos_all >= p0) & (pos_all <= p1)]
                for k in outs:
                    for cn in ((cname, "A") if cname == "AM" else (cname,)):
                        cm, cu = ctl[(k, cn)]
                        r = n5.stats(pos, arr[k], cm, cu, (p0, p1), rng, 5 if k == "r5" else 20)
                        judged = d != 0 and cn == cname and pname in ("2024~2026", "2023~2026")
                        r.update(part="C", group=gname, variant=vname, period=pname, outcome=k, control=cn, direction=d,
                                 verdict=n5.verdict(r, d) if judged else "",
                                 verdict_kind=("등록(재현)" if pname == "2024~2026" else "사후 확장") if judged else "")
                        rows.append(r)
    cells = pd.DataFrame(rows)
    front = ["part", "group", "variant", "period", "outcome", "control", "direction", "verdict", "verdict_kind", "n", "clusters"]
    cells = cells[front + [c for c in cells.columns if c not in front]]
    cells.to_csv(RES / f"{OUT}_cells.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------- 점검 4: 2024~2026 칸이 n5_cells.csv 와 같은가(부트스트랩 구간은 난수 순서가 달라 제외)
    old = pd.read_csv(RES / "n5_cells.csv", encoding="utf-8-sig")
    key = ["part", "group", "variant", "period", "outcome", "control"]
    mg = cells[cells.period != "2023~2026"].merge(old, on=key, suffixes=("", "_old"), how="inner")
    assert len(mg) == len(old), (len(mg), len(old))
    for col in ("n", "mean", "ctl_mean", "ex_mean", "p_two"):
        a, b = mg[col].to_numpy(float), mg[f"{col}_old"].to_numpy(float)
        diff = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9))
        aud.append(dict(check=f"n5_cells 2024~2026 {col} 재현(다른 칸 수)", n=len(mg), nan_mismatch=int(diff.sum()),
                        max_abs_diff=float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0))
    vchg = mg[(mg.verdict.fillna("") != mg.verdict_old.fillna("")) & (mg.verdict_old.fillna("") != "")]
    aud.append(dict(check="등록 판정 14칸 재현(다른 칸 수)", n=int((mg.verdict_old.fillna("") != "").sum()),
                    nan_mismatch=len(vchg), max_abs_diff=0.0))
    aud = pd.DataFrame(aud)
    aud.to_csv(RES / f"{OUT}_audit.csv", index=False, encoding="utf-8-sig")
    hand.to_csv(RES / f"{OUT}_audit_hand.csv", index=False, encoding="utf-8-sig")
    hard = aud[~aud.check.str.startswith("n5_cells 2024~2026")]
    assert (hard.max_abs_diff < 1e-9).all() and (hard.nan_mismatch == 0).all(), hard

    # ------------------------------------------- G5(2023 확인 3건 포함)
    ev = pd.read_csv(RES / "n4h5_soxx_events.csv", encoding="utf-8-sig")
    ev = ev[(ev.det == "ZZA3") & (ev.type == 1) & (ev.conf_date >= "2023-01-01")].copy()
    ev["c"] = sess.get_indexer(pd.to_datetime(ev.conf_date))
    assert (ev.c >= 0).all()
    ev["entry"] = [sess[c + 1].date().isoformat() if c + 1 < len(sess) else None for c in ev.c]
    ev = ev[ev.entry.notna()].copy()
    ev["w"] = ev.c + 1 - i0
    for k in ("r20", "o21", "mae20"):
        ev[f"{k}_buy"] = arr[k][ev.w.to_numpy(int)]
        ev[f"A_{k}"] = ctl[(k, "A")][0][ev.w.to_numpy(int)]
    g5 = ev[["conf_date", "entry", "pivot_date", "R_all_pre", "R_pre_ids", "r20_buy", "A_r20", "o21_buy", "mae20_buy", "A_mae20", "y1"]]
    g5.to_csv(RES / f"{OUT}_g5.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------- 판정·출력
    reg = cells[cells.verdict != ""]
    ver = {f"{r.verdict_kind}|{r.group}|{r.variant}|{r.outcome}": r.verdict for r in reg.itertuples()}
    tests = dict(judged_cells=len(ver),
                 verdict_counts={kind: reg[reg.verdict_kind == kind].verdict.value_counts().to_dict() for kind in reg.verdict_kind.unique()},
                 cells_total=len(cells), cells_new_period=int((cells.period.isin(["2023~2026", "2023"])).sum()),
                 bootstrap=dict(B=n5.B, seed=n5.SEED), null=dict(kind="공통 원형 이동", kmin=h14.KMIN),
                 window=[str(W0.date()), str(W1.date()), D], signals_in_window=len(win),
                 signals_2023=int((win.entry.dt.year == 2023).sum()), runtime_s=round(time.time() - T0, 1))
    (RES / f"{OUT}_verdict.json").write_text(json.dumps(dict(verdicts=ver, tests=tests), ensure_ascii=False, indent=1), encoding="utf-8")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 40)
    L = []
    P = L.append
    P(f"N5 2023 포함 확장 {time.strftime('%Y-%m-%d %H:%M')}  창 {W0.date()}~{W1.date()} D={D}  신호 {len(win)}(2023 {tests['signals_2023']})")
    P("== 점검 ==")
    P(aud.to_string(index=False))
    P(hand.round(5).to_string(index=False))
    P("")
    cols = ["group", "period", "n", "clusters", "win", "mean", "median", "worst", "ctl_win", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_two"]
    focus = ["2023~2026", "2024~2026", "2026", "2023"]
    a = cells[(cells.part == "A") & (cells.outcome == "r20") & (cells.control == "A") & (cells.variant == "all") & cells.period.isin(focus)]
    P("== A. r20 대조 A, all ==")
    P(a[cols].round(4).to_string(index=False))
    P("")
    a = cells[(cells.part == "A") & (cells.outcome == "r20") & (cells.control == "Y") & (cells.variant == "all") & cells.period.isin(focus)]
    P("== A. r20 같은 해 모든 날(Y), all ==")
    P(a[["group", "period", "n", "win", "mean", "ctl_win", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_two"]].round(4).to_string(index=False))
    P("")
    a = cells[(cells.part == "A") & (cells.outcome == "mae20") & (cells.variant == "all") & cells.period.isin(focus)]
    P("== A. mae20, all ==")
    P(a[["group", "period", "control", "n", "mean", "median", "worst", "deep10", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_two"]].round(4).to_string(index=False))
    P("")
    a = cells[(cells.part == "A") & (cells.outcome == "o21") & (cells.variant == "all") & cells.period.isin(focus)]
    P("== A. o21 대조 A, all ==")
    P(a[["group", "period", "n", "win", "mean", "ctl_mean", "ex_mean", "ex_lo", "ex_hi", "p_two"]].round(4).to_string(index=False))
    P("")
    P("== 아무 날(Y) 기준: 기간별 r20·mae20 ==")
    for pname, a0, b0 in PERIODS:
        m = (day.index >= a0) & (day.index <= b0)
        v, mm = arr["r20"][m], arr["mae20"][m]
        ok = ~np.isnan(v)
        P(f"{pname}: 날 {int(ok.sum())} 수익 비율 {np.mean(v[ok] > 0):.3f} 평균 {np.mean(v[ok]):.4f} 중앙 {np.median(v[ok]):.4f} "
          f"mae20 평균 {np.nanmean(mm):.4f} −10% 비율 {np.mean(mm[~np.isnan(mm)] <= -0.10):.3f}")
    P("")
    P("== G5 상승 반전 확인(ZZA3) 다음 날 시가 매수(2023 포함) ==")
    P(g5.round(4).to_string(index=False))
    for lab, gg in (("2023~2026", g5), ("2024~2026", g5[g5.conf_date >= "2024-01-01"])):
        for flag, g in gg.groupby("R_all_pre"):
            v = g.r20_buy.dropna()
            P(f"{lab} R 사전 {flag}: n={len(v)} 승 {int((v > 0).sum())} 평균 {v.mean():.4f} 중앙 {v.median():.4f} 최악 {v.min():.4f} "
              f"대조A {g.A_r20[v.index].mean():.4f} mae20 평균 {g.mae20_buy[v.index].mean():.4f}")
    P("")
    P("== C. 결합 해석(등록 판정 2024~2026 재현 + 2023~2026 사후 확장) ==")
    cc = ["group", "variant", "period", "outcome", "control", "verdict", "verdict_kind", "n", "clusters", "win", "mean", "ctl_mean",
          "ex_mean", "ex_lo", "ex_hi", "p_hi", "p_lo"]
    P(cells[(cells.part == "C") & cells.period.isin(focus)][cc].round(4).to_string(index=False))
    P("")
    P("== 2023 신호 한 줄씩(묶음 표지, r20·mae20) ==")
    w = win[win.entry.dt.year == 2023].sort_values("entry")
    marks = [c for c in ("G1", "G2", "G3", "G4", "C1", "C2", "C3") if c in w]
    for x in w.itertuples():
        tag = ",".join(c for c in marks if getattr(x, c))
        P(f"{x.entry.date()} {x.group_id} {x.bank} {x.family} lb={x.lookback_months} [{tag}] r20 {x.r20:+.3f} mae {x.mae20:+.3f} A {x.A_r20:+.3f}")
    P("")
    P(json.dumps(tests, ensure_ascii=False))
    txt = "\n".join(L)
    (RES / f"{OUT}_tables.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
