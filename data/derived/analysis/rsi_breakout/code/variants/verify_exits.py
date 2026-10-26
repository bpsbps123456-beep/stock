"""감사(2026-10-10): variants/exits.py 결과를 반박해 본다. exits.py·harness.py는 고치지 않는다.

1) 미래값 탐지: 결정일 T 이후 행(cls·K·r160·sig70은 T행부터, 시가는 T+1행부터)을 마구 바꿔도
   T일까지의 보유 목록이 그대로인지 14개 규칙 모두 확인한다.
2) 독립 재계산: 원본 1d/1h parquet에서 160분봉 RSI 70 돌파·20일선·5일 수익을 따로 계산하고
   7자리 계좌를 직접 굴려 '종가 < 20일선' 변형의 날짜별 보유·구간 수익·거래를 harness 결과와 맞춘다.
   3개 날짜에서 보유 종목마다 매수 신호(전날 160분봉 RSI 돌파)와 매도(전날 종가 < 20일선)를 원본 값으로 보인다.
3) 강건성: 앞 1위와 이웃(감사용 사후 10·15·25·30일선), 뒤 구간 종목 기여·하루 쏠림, 빌더 주장 확인.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/rsi_breakout/code/variants/verify_exits.py
"""
import os, sys, copy, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from harness import Harness, PERIODS                                   # noqa: E402
import exits as E                                                       # noqa: E402

CACHE = os.path.join(os.path.dirname(HERE), "..", "cache")
J = json.load(open(os.path.join(os.path.dirname(HERE), "..", "results", "variants", "exits.json"), encoding="utf-8"))
JB = {r["name"]: r for r in J}


def entry_of(H):
    sig = {s: H.sig70[s].to_numpy(dtype=bool) for s in H.sig70.columns}
    return lambda s, k: bool(sig[s][k - 1]) if s in sig else False


def run_prefix(H, rule, T):
    """run_one과 같은 매수·우선순위로 k0..T까지 pick만 돌려 날짜별 보유를 돌려준다."""
    pick = H.entry_exit_picker(entry_of(H), rule, priority=lambda k: H.ret(5).iloc[k - 1])
    k0 = next(i for i, d in enumerate(H.days) if d >= "2026-01-02")
    held, out = [], {}
    for k in range(k0, T + 1):
        held = pick(k, held)
        out[k] = tuple(sorted(held))
    return out


def scrambled(H, T, seed):
    """T행부터(시가는 T+1행부터) 값을 무작위로 바꾼 얕은 복사본."""
    rng = np.random.default_rng(seed)
    H2 = copy.copy(H)
    H2._cache = {}
    H2.P = copy.copy(H.P)

    def scr(df, kind, start):
        x = df.copy()
        n = len(x) - start
        if n <= 0:
            return x
        if kind == "price":
            x.iloc[start:] = x.iloc[start:].to_numpy() * rng.uniform(0.5, 1.6, size=(n, x.shape[1]))
        elif kind == "osc":
            x.iloc[start:] = rng.uniform(0, 100, size=(n, x.shape[1]))
        else:
            x.iloc[start:] = rng.random(size=(n, x.shape[1])) < 0.3
        return x
    H2.cls = scr(H.cls, "price", T)
    H2.opn = scr(H.opn, "price", T + 1)
    H2.k = scr(H.k, "osc", T)
    H2.P.k = H2.k
    H2.r160 = scr(H.r160, "osc", T)
    H2.sig70 = scr(H.sig70.astype(bool), "bool", T)
    return H2


# ---------------------------------------------------------------- 독립 재계산(원본 parquet)
def wilder_rsi(x, n=14):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    d = np.diff(x)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    if len(d) < n:
        return out
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, len(d) + 1):
        if i > n:
            au = (au * (n - 1) + up[i - 1]) / n
            ad = (ad * (n - 1) + dn[i - 1]) / n
        out[i] = 100.0 if ad == 0 else (0.0 if au == 0 else 100 - 100 / (1 + au / ad))
    return out


def raw_tables(days, syms):
    d = pd.read_parquet(os.path.join(CACHE, "1d.parquet"))
    d["day"] = d.ts.dt.strftime("%Y-%m-%d")
    O = d.pivot_table(index="day", columns="sym", values="Open", aggfunc="last").reindex(days)[syms]
    C = d.pivot_table(index="day", columns="sym", values="Close", aggfunc="last").reindex(days)[syms]
    h = pd.read_parquet(os.path.join(CACHE, "1h.parquet"))
    h = h[h.sym.isin(syms)]
    mins = h.ts.dt.hour * 60 + h.ts.dt.minute
    h = h[(mins >= 570) & (mins < 960)].copy()
    mins = h.ts.dt.hour * 60 + h.ts.dt.minute
    h["day"] = h.ts.dt.strftime("%Y-%m-%d")
    h["b"] = np.select([mins < 750, mins < 870], [0, 1], 2)                # 9:30·10:30·11:30 | 12:30·13:30 | 14:30·15:30
    sig, r160 = {}, {}
    for s, g in h.sort_values("ts").groupby("sym"):
        bars = g.groupby(["day", "b"]).Close.last()
        r = wilder_rsi(bars.to_numpy())
        prev = np.r_[np.nan, r[:-1]]
        hit = (prev <= 70) & (r > 70)
        dd = bars.index.get_level_values(0)
        sig[s] = pd.Series(hit, index=dd).groupby(level=0).any().reindex(days, fill_value=False)
        r160[s] = pd.Series(r, index=dd).groupby(level=0).last().reindex(days)
    S = pd.DataFrame(sig).reindex(columns=syms, fill_value=False).fillna(False).astype(bool)
    return O, C, S, pd.DataFrame(r160)


def indep_sim(H, O, C, S, n_ma=20):
    days, syms = H.days, list(O.columns)
    M = C.rolling(n_ma, min_periods=n_ma).mean()
    R5 = C / C.shift(5) - 1
    o, c, m, sg, r5 = O.to_numpy(), C.to_numpy(), M.to_numpy(), S.to_numpy(), R5.to_numpy()
    col = {s: i for i, s in enumerate(syms)}
    k0 = next(i for i, d in enumerate(days) if d >= "2026-01-02")
    held, rows, trades, k_in = [], [], [], {}
    for k in range(k0, len(days) - 1):
        keep, sold = [], []
        for s in held:
            j = col[s]
            (sold if c[k - 1, j] < m[k - 1, j] else keep).append(s)
        elig = H.eligible(days[k])
        new = [s for s in syms if s not in keep and s in elig and np.isfinite(o[k, col[s]]) and sg[k - 1, col[s]]]
        new.sort(key=lambda s: -(r5[k - 1, col[s]] if np.isfinite(r5[k - 1, col[s]]) else -1e9))
        add = new[:max(0, 7 - len(keep))]
        for s in sold:
            trades.append(dict(sym=s, k_in=k_in.pop(s), k_out=k))
        for s in add:
            if s in k_in:                                      # 판 날 다시 산 경우는 sold에서 이미 닫혔다
                trades.append(dict(sym=s, k_in=k_in.pop(s), k_out=k))
            k_in[s] = k
        held = keep + add
        w = 1 / max(7, len(held))
        r = 0.0
        for s in held:
            a, b = o[k, col[s]], o[k + 1, col[s]]
            r += w * (b / a - 1 if np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0)
        rows.append(dict(day=days[k], ret=r, n=len(held), held=",".join(sorted(held)), sold=",".join(sold),
                         add=",".join(add)))
    last = len(days) - 1
    for s, ki in k_in.items():
        trades.append(dict(sym=s, k_in=ki, k_out=last, open=True))
    tr = pd.DataFrame(trades)
    tr["ret"] = [o[r.k_out, col[r.sym]] / o[r.k_in, col[r.sym]] - 1 for r in tr.itertuples()]
    tr["entry"] = [days[i] for i in tr.k_in]
    tr["exit"] = [days[i] for i in tr.k_out]
    return pd.DataFrame(rows), tr, M


def period_tot(df, a, b):
    x = df[(df.day >= a) & (df.day <= b)]
    return float((1 + x.ret).prod() - 1)


def period_contrib(H, df, a, b):
    O = H.opn
    x = df[(df.day >= a) & (df.day <= b)]
    val, c, daily = 1.0, {}, []
    for row in x.itertuples():
        held = row.held.split(",") if row.held else []
        k = H.days.index(row.day)
        w = 1 / max(7, len(held))
        r = 0.0
        for s in held:
            rr = O[s].iloc[k + 1] / O[s].iloc[k] - 1
            rr = rr if np.isfinite(rr) else 0.0
            c[s] = c.get(s, 0.0) + val * w * rr
            r += w * rr
        daily.append((row.day, r))
        val *= 1 + r
    return val - 1, sorted(c.items(), key=lambda kv: -kv[1]), daily


def main():
    H = Harness()
    rules = E.build_rules(H)
    print("== 0) exits.json 재현(run_one 재실행) ==")
    for n, p, f in rules:
        r = E.run_one(H, n, p, f)
        diff = max(abs(r[lab]["수익"] - JB[n][lab]["수익"]) for lab, _, _ in PERIODS)
        print(f"  {n[:24]:24s} 최대 차이 {diff:.2e}  앞 {r[PERIODS[0][0]]['수익']:+.4f} 뒤 {r[PERIODS[1][0]]['수익']:+.4f}")

    print("\n== 1) 미래값 탐지(행 T 이후 무작위화, T까지 보유 비교) ==")
    Ts = [H.days.index(d) for d in ("2026-02-17", "2026-06-15", "2026-09-21")]
    base = {n: run_prefix(H, f, max(Ts)) for n, p, f in rules}
    for T in Ts:
        H2 = scrambled(H, T, seed=T)
        rules2 = E.build_rules(H2)
        bad = []
        for (n, p, f), (_, _, f2) in zip(rules, rules2):
            got = run_prefix(H2, f2, T)
            mism = [H.days[k] for k in got if got[k] != base[n][k]]
            if mism:
                bad.append((n, mism[:3]))
        print(f"  T={H.days[T]}: 불일치 {len(bad)}개 규칙 {bad}")
    # 양성 대조: 같은 날(k행) 종가를 쓰는 고의 미래값 규칙이면 탐지기가 잡는지
    leak = lambda s, k, p: H.cls[s].iloc[k] < H.ma(20)[s].iloc[k]
    T = Ts[1]
    b0 = run_prefix(H, leak, T)
    H2 = scrambled(H, T, seed=T)
    leak2 = lambda s, k, p: H2.cls[s].iloc[k] < H2.ma(20)[s].iloc[k]
    g2 = run_prefix(H2, leak2, T)
    print(f"  양성 대조(고의 k행 사용): T일 보유 다름 = {g2[T] != b0[T]}")

    print("\n== 2) 독립 재계산: '종가 < 20일선' ==")
    syms = list(H.syms)
    O, C, S, R160 = raw_tables(H.days, syms)
    print(f"  원본 시가·종가 vs harness 최대 상대차: 시가 {np.nanmax(np.abs(O.to_numpy() / H.opn[syms].to_numpy() - 1)):.2e}, "
          f"종가 {np.nanmax(np.abs(C.to_numpy() / H.cls[syms].to_numpy() - 1)):.2e}")
    hs = H.sig70.reindex(columns=syms, fill_value=False).astype(bool)
    k0 = next(i for i, d in enumerate(H.days) if d >= "2025-12-01")
    a, b = S.iloc[k0:].to_numpy(), hs.iloc[k0:].to_numpy()
    print(f"  160분 RSI 70 돌파(2025-12~): 독립 {a.sum()}건, harness {b.sum()}건, 불일치 칸 {(a != b).sum()}")
    df_i, tr_i, M = indep_sim(H, O, C, S, 20)
    rule20 = next(f for n, p, f in rules if n == "종가 < 20일선")
    df_h = H.simulate(H.entry_exit_picker(entry_of(H), rule20, priority=lambda k: H.ret(5).iloc[k - 1]))
    m = df_h.merge(df_i, on="day", suffixes=("_h", "_i"))
    print(f"  날짜 {len(m)}일 중 보유 다른 날 {(m.held_h != m.held_i).sum()}일, 일수익 최대 차 {np.abs(m.ret_h - m.ret_i).max():.2e}")
    for lab, a_, b_ in PERIODS:
        print(f"  {lab}: 독립 {period_tot(df_i, a_, b_):+.4f} / json {JB['종가 < 20일선'][lab]['수익']:+.4f}")
    for lab, a_, b_ in PERIODS[:2]:
        x = tr_i[(tr_i.entry >= a_) & (tr_i.entry <= b_)]
        print(f"  거래 {lab}: 독립 {len(x)}건 평균 {x.ret.mean():+.4f} 보유 {(x.k_out - x.k_in).mean():.1f}일 / json "
              f"{JB['종가 < 20일선']['거래'][lab]['건수']}건 {JB['종가 < 20일선']['거래'][lab]['평균수익']:+.4f}")

    print("\n  -- 날짜별 점검(원본 값) --")
    col = {s: i for i, s in enumerate(syms)}
    for day in ("2026-03-23", "2026-07-15", "2026-10-06"):
        k = H.days.index(day)
        row = df_i[df_i.day == day].iloc[0]
        print(f"  [{day}] 보유 {row['held']} | 이날 시가 매도 {row['sold'] or '-'} | 이날 시가 매수 {row['add'] or '-'}")
        for s in row["held"].split(","):
            t = tr_i[(tr_i.sym == s) & (tr_i.k_in <= k) & (tr_i.k_out > k)].iloc[0]
            ki = int(t.k_in)
            seg = C[s].iloc[ki - 1:k].to_numpy() >= M[s].iloc[ki - 1:k].to_numpy()
            hold_ok = bool(np.all(seg[1:])) if len(seg) > 1 else True
            print(f"    {s:5s} 산 날 {H.days[ki]} 시가 {O[s].iloc[ki]:.2f} | 전날 {H.days[ki - 1]} 160분 RSI 돌파 {bool(S[s].iloc[ki - 1])}"
                  f"(마지막봉 RSI {R160[s].iloc[ki - 1]:.1f}) | 산 뒤~{H.days[k - 1]} 종가≥20일선 유지 {hold_ok} "
                  f"| 전날 종가 {C[s].iloc[k - 1]:.2f} 20일선 {M[s].iloc[k - 1]:.2f}")
        for s in filter(None, row["sold"].split(",")):
            print(f"    매도 {s}: {H.days[k - 1]} 종가 {C[s].iloc[k - 1]:.2f} < 20일선 {M[s].iloc[k - 1]:.2f} "
                  f"→ {day} 시가 {O[s].iloc[k]:.2f} 체결")
    # 거래 3건 원본 시가로 수익 재계산
    print("\n  -- 거래 3건 --")
    for sym in ("MRVL", "ARM", "INTC"):
        x = tr_i[tr_i.sym == sym].head(2)
        for t in x.itertuples():
            ko = int(t.k_out)
            print(f"    {sym} {t.entry} 시가 {O[sym].iloc[t.k_in]:.2f} → {t.exit} 시가 {O[sym].iloc[ko]:.2f} = {t.ret:+.4f}"
                  f" | 매도 근거 {H.days[ko - 1]} 종가 {C[sym].iloc[ko - 1]:.2f} vs 20일선 {M[sym].iloc[ko - 1]:.2f}"
                  f"{' (끝까지 보유)' if getattr(t, 'open', False) is True else ''}")
    # 20일선 NaN(상장 직후·빈칸) 때문에 매도가 막힌 보유일
    nan_days = 0
    for r in df_i.itertuples():
        k = H.days.index(r.day)
        for s in r.held.split(","):
            if s and not np.isfinite(M[s].iloc[k - 1]):
                nan_days += 1
    print(f"  20일선 NaN이라 매도 판단 불가였던 보유·일: {nan_days}")

    print("\n== 3) 강건성 ==")
    print("  [감사 사후] 20일선 이웃 — 선택에 쓰지 않음")
    for n_ma in (10, 15, 20, 25, 30):
        d_, t_, _ = indep_sim(H, O, C, S, n_ma)
        cells = " ".join(f"{lab[:5]} {period_tot(d_, a_, b_):+.3f}" for lab, a_, b_ in PERIODS)
        print(f"    {n_ma}일선: {cells} | 평균보유 {d_.n.mean():.2f}")
    for n in ("종가 < 20일선", "고점 대비 -12% 종가", "고점 대비 -16% 종가", "20거래일 보유", "고점 -12% 또는 20일선 아래",
              "기준: K>95(사용자)"):
        f = next(f for nn, p, f in rules if nn == n)
        df = H.simulate(H.entry_exit_picker(entry_of(H), f, priority=lambda k: H.ret(5).iloc[k - 1]))
        tot, top, daily = period_contrib(H, df, "2026-06-01", "2026-10-09")
        gain = sum(v for _, v in top if v > 0)
        dr = sorted(daily, key=lambda x: -x[1])
        ex2 = float(np.prod([1 + r for d, r in daily if d not in {dr[0][0], dr[1][0]}]) - 1)
        print(f"  {n}: 뒤 {tot:+.4f} | 상위3 {[(s, round(v, 3)) for s, v in top[:3]]} 합 {sum(v for _, v in top[:3]):+.3f}"
              f" (뒤 수익 대비 {sum(v for _, v in top[:3]) / tot if tot > 0 else float('nan'):.0%}) | 하위3 "
              f"{[(s, round(v, 3)) for s, v in top[-3:]]} | 최고 2일 {[(d, round(r, 3)) for d, r in dr[:2]]} 빼면 {ex2:+.4f}")
        if n in ("고점 대비 -12% 종가", "기준: K>95(사용자)"):
            oc = df[df.day >= "2026-10-01"]
            sep = df[df.day >= "2026-09-01"]
            hs_ = [set(h.split(",")) for h in sep.held]
            new_sep = sum(len(hs_[i] - hs_[i - 1]) for i in range(1, len(hs_)))
            run = []
            for i in range(1, len(df)):
                a0 = set(df.held.iloc[i - 1].split(",")) if df.held.iloc[i - 1] else set()
                for s in set(df.held.iloc[i].split(",")) - a0 - {""}:
                    run.append((s, df.day.iloc[i]))
            last_in = {s: d for s, d in run if s in oc.held.iloc[-1].split(",")}
            print(f"     10월 마지막 보유의 마지막 매수일 {last_in}")
            print(f"     10월 첫날 보유 {oc.held.iloc[0]} / 마지막 {oc.held.iloc[-1]} | 9/2~ 새로 들어온 종목 수 {new_sep}"
                  f" | 10월 행 수 {len(oc)}")
    # 사용자 규칙에 PEPG가 들어간 적 있나
    fb = next(f for nn, p, f in rules if nn.startswith("기준"))
    dfb = H.simulate(H.entry_exit_picker(entry_of(H), fb, priority=lambda k: H.ret(5).iloc[k - 1]))
    print(f"  사용자 규칙 7자리 계좌에 PEPG 보유일: {sum('PEPG' in h.split(',') for h in dfb.held)}")


if __name__ == "__main__":
    main()
