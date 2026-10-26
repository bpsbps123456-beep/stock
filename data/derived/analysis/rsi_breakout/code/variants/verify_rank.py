"""rank 계열 감사(2026-10-10): 고른 변형 'rank rs60 주간+완충12'를 원자료(cache/1d.parquet)에서 따로 다시 계산한다.

1) 독립 재계산: 1d.parquet 시가·종가만으로 rs60(k-1행)·ISO 주 첫 거래일·완충 12·7자리·시가→다음 시가를 새로 짜서
   하네스 simulate 결과와 날마다 대조한다. rs60은 k일마다 C.iloc[:k](k-1행까지)만 잘라서 계산 → 미래 자료가 들어갈 수 없다.
2) 표본 날짜 3개: 들고 있는 종목의 k-1일 순위·값, 판 종목의 순위(>12 또는 대상 밖), 판 시가 대조.
3) 견고성: 뒤 구간 종목별·날짜별 기여, 이웃 값(완충·기간·주중 재조정 위치·신호 하루 더 늦추기).
4) 다른 점수의 시점 점검: r160 날짜 정렬(1시간봉), hi(252) 그날 포함 여부, near252 주간 독립 재계산.
"""
import json, os, sys, time
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from harness import Harness, PERIODS                                   # noqa: E402
from rsi_breakout import CACHE, RES, START, SLOTS, rsi                 # noqa: E402

FRONT, BACK = "앞 1/2~5/29", "뒤 6/1~10/9"


def iso(d):
    return date.fromisoformat(d).isocalendar()[:2]


def load_raw():
    raw = pd.read_parquet(os.path.join(CACHE, "1d.parquet"))
    raw["d"] = pd.to_datetime(raw.ts).dt.strftime("%Y-%m-%d")
    qdays = sorted(raw.loc[raw.sym == "QQQ", "d"])
    O = raw.pivot(index="d", columns="sym", values="Open").reindex(qdays)
    C = raw.pivot(index="d", columns="sym", values="Close").reindex(qdays)
    return qdays, O, C


def period_tot(days, rets, a, b):
    r = [x for d, x in zip(days, rets) if a <= d <= b]
    return float(np.prod([1 + x for x in r]) - 1) if r else float("nan")


def indep_sim(days, O, C, elig_fn, score_at, buffer=12, slots=SLOTS, rebal=None, lag=1):
    """완전히 따로 짠 7자리 계좌. score_at(Ctrunc) → 종목별 점수(마지막 행 기준). Ctrunc = C.iloc[:k-lag+1]."""
    k0 = next(i for i, d in enumerate(days) if d >= START)
    held, out = [], []
    for k in range(k0, len(days) - 1):
        day = days[k]
        is_rebal = (not held) or (rebal(k, k0) if rebal else iso(day) != iso(days[k - 1]))
        if not is_rebal:
            held = [s for s in held if np.isfinite(O[s].iloc[k])]
        else:
            el = sorted(s for s in elig_fn(day) if s in O.columns and np.isfinite(O[s].iloc[k]))
            sc = score_at(C.iloc[:k - lag + 1])
            sc = sc.reindex(el).dropna().sort_values(ascending=False)
            ranked = list(sc.index)
            keep = [s for s in held if s in ranked[:buffer]] if buffer else []
            held = keep + [s for s in ranked if s not in keep][:slots - len(keep)]
        w = 1 / max(slots, len(held)) if held else 0
        r = 0.0
        for s in held:
            a, b = O[s].iloc[k], O[s].iloc[k + 1]
            r += w * (b / a - 1) if np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0
        out.append((day, r, tuple(sorted(held))))
    return out


def rs_n(n):
    def f(Ct):
        if len(Ct) <= n:
            return pd.Series(dtype=float)
        last, base = Ct.iloc[-1], Ct.iloc[-1 - n]
        q = last["QQQ"] / base["QQQ"]
        return (last / base) / q - 1
    return f


def main():
    t0 = time.time()
    H = Harness()
    print(f"Harness {time.time() - t0:.0f}초")
    days, O, C = load_raw()
    assert days == list(H.days), "거래일 불일치"
    # 하네스 표와 원자료 일치
    dO = (H.opn - O[H.syms]).abs().max().max()
    dC = (H.cls - C[H.syms]).abs().max().max()
    print(f"[원자료 대조] 시가 최대 차 {dO:.2e}, 종가 최대 차 {dC:.2e}")

    # ---------------- 1) 고른 변형 하네스 재실행
    df = H.simulate(H.rank_picker(lambda k: H.rs(60).iloc[k - 1], slots=SLOTS, buffer=12, weekly=True))
    ev = H.evaluate("rank rs60 주간+완충12", df)
    js = {r["name"]: r for r in json.load(open(os.path.join(RES, "variants", "rank.json"), encoding="utf-8"))["results"]}
    jb = js["rank rs60 주간+완충12"]
    print("\n[하네스 재실행 vs rank.json]")
    for lab, _, _ in PERIODS:
        print(f"  {lab}: 재실행 {ev[lab]['수익']:+.4f} / json {jb[lab]['수익']:+.4f} · 낙폭 {ev[lab]['낙폭']:.4f}/{jb[lab]['낙폭']:.4f}")

    # ---------------- 1b) 독립 재계산
    ind = indep_sim(days, O, C, H.eligible, rs_n(60))
    idays = [d for d, _, _ in ind]
    irets = [r for _, r, _ in ind]
    ih = [",".join(h) for _, _, h in ind]
    same_hold = sum(a == b for a, b in zip(ih, df.held))
    maxdiff = float(np.max(np.abs(np.array(irets) - df.ret.to_numpy())))
    print(f"\n[독립 재계산] 거래일 {len(ind)} · 보유 일치 {same_hold}/{len(ind)} · 일간 수익 최대 차 {maxdiff:.2e}")
    for lab, a, b in PERIODS:
        print(f"  {lab}: 독립 {period_tot(idays, irets, a, b):+.4f}")
    first_bad = next((d for d, a, b in zip(idays, ih, df.held) if a != b), None)
    if first_bad:
        print("  첫 불일치", first_bad)

    # ---------------- 2) 표본 날짜
    pos = {d: i for i, d in enumerate(days)}
    held_by = dict(zip(df.day, df.held))
    chg = [d for i, d in enumerate(df.day) if i > 0 and df.held.iloc[i] != df.held.iloc[i - 1]]
    print(f"\n[보유가 바뀐 날] {len(chg)}일: {chg}")
    midweek_changes = [d for d in chg if iso(d) == iso(days[pos[d] - 1])]
    print(f"  주중(첫 거래일 아님)에 바뀐 날: {midweek_changes or '없음'}")
    samples = [d for d in chg if d >= "2026-06-01"]
    pick3 = [samples[0], next(d for d in samples if d >= "2026-07-13"), samples[-1]] if len(samples) >= 3 else samples
    for d in pick3:
        k = pos[d]
        prev = set(held_by[days[k - 1]].split(","))
        cur = set(held_by[d].split(","))
        el = sorted(s for s in H.eligible(d) if np.isfinite(O[s].iloc[k]))
        c1, c61 = C.iloc[k - 1], C.iloc[k - 61]
        rs = ((c1 / c61) / (c1["QQQ"] / c61["QQQ"]) - 1).reindex(el).dropna().sort_values(ascending=False)
        rank = {s: i + 1 for i, s in enumerate(rs.index)}
        print(f"\n  [{d}] 결정 근거 행 {days[k - 1]}(k-1), 60행 전 {days[k - 61]} · 대상 {len(el)}종목 · QQQ 60일 {c1['QQQ'] / c61['QQQ'] - 1:+.3f}")
        print(f"    상위 12: {[(s, round(float(v), 3)) for s, v in rs.head(12).items()]}")
        for s in sorted(cur):
            tag = "유지" if s in prev else "새로 삼"
            print(f"    보유 {s:6s} {tag:4s} 순위 {rank.get(s)} rs60 {rs.get(s, np.nan):+.3f} "
                  f"(종가 {c61[s]:.2f}→{c1[s]:.2f}) 시가 {O[s].iloc[k]:.2f}")
        for s in sorted(prev - cur):
            # 판 종목: 마지막 수익 행이 k-1(시가 k-1→시가 k)인지
            last_row = df[df.held.str.split(",").apply(lambda h: s in h)].day.max()
            print(f"    판 {s:6s} 순위 {rank.get(s, '대상 밖')} rs60 {rs.get(s, np.nan):+.3f} · 마지막 보유 행 {last_row}"
                  f"(=k-1 {days[k - 1]}) · 판 시가 {O[s].iloc[k]:.2f}")
        rr = sum((O[s].iloc[k + 1] / O[s].iloc[k] - 1) for s in cur) / 7
        print(f"    그날 수익(시가 {d}→{days[k + 1]}) 손계산 {rr:+.5f} / 하네스 {df.set_index('day').ret[d]:+.5f}")

    # ---------------- 3) 견고성
    print("\n[뒤 구간 기여 — 종목별 Σ(가치×비중×수익)]")
    back = df[(df.day >= "2026-06-01") & (df.day <= "2026-10-09")].reset_index(drop=True)
    val, con, daily = 1.0, {}, []
    for _, row in back.iterrows():
        k = pos[row.day]
        hs = row.held.split(",") if row.held else []
        for s in hs:
            r = O[s].iloc[k + 1] / O[s].iloc[k] - 1
            con[s] = con.get(s, 0) + val * r / max(SLOTS, len(hs))
        val *= 1 + row.ret
    tot = val - 1
    srt = sorted(con.items(), key=lambda t: -t[1])
    print(f"  뒤 합계 {tot:+.4f} (기여 합 {sum(con.values()):+.4f})")
    print(f"  상위 5 {[(s, round(v, 4)) for s, v in srt[:5]]}")
    print(f"  하위 5 {[(s, round(v, 4)) for s, v in srt[-5:]]}")
    pos_sum = sum(v for v in con.values() if v > 0)
    print(f"  상위3 ÷ 플러스 기여 합 {sum(v for _, v in srt[:3]) / pos_sum:.2f} · 종목 수 {len(con)}")
    # 종목 하나씩 빼기(그 자리 현금)
    def back_wo(excl):
        v = 1.0
        for _, row in back.iterrows():
            k = pos[row.day]
            hs = row.held.split(",")
            r = sum((O[s].iloc[k + 1] / O[s].iloc[k] - 1) / max(SLOTS, len(hs)) for s in hs if s not in excl)
            v *= 1 + r
        return v - 1
    for s, _ in srt[:2] + srt[-2:]:
        print(f"  {s} 빼면(현금) 뒤 {back_wo({s}):+.4f}")
    br = back.ret.to_numpy()
    o = np.argsort(br)
    def prod(x):
        return float(np.prod(1 + x) - 1)
    print(f"  최고 2일 {[(back.day[i], round(br[i], 4)) for i in o[-2:]]} 빼면 {prod(np.delete(br, o[-2:])):+.4f}")
    print(f"  최악 2일 {[(back.day[i], round(br[i], 4)) for i in o[:2]]} 빼면 {prod(np.delete(br, o[:2])):+.4f}")
    print(f"  월별 {[(m, round(v, 4)) for m, v in ev['월별'].items() if m >= '2026-06']}")
    print(f"  7월 제외 뒤(6·8·9·10월 곱) {np.prod([1 + ev['월별'][m] for m in ('2026-06', '2026-08', '2026-09', '2026-10')]) - 1:+.4f}")

    print("\n[이웃 값 — 감사용 민감도, 고르기에 쓰지 않음]")
    rows = []
    def run(label, **kw):
        out = indep_sim(days, O, C, H.eligible, **kw)
        dd, rr = [d for d, _, _ in out], [r for _, r, _ in out]
        f, b = period_tot(dd, rr, *PERIODS[0][1:]), period_tot(dd, rr, *PERIODS[1][1:])
        s, oc = period_tot(dd, rr, *PERIODS[2][1:]), period_tot(dd, rr, *PERIODS[3][1:])
        rows.append((label, f, b, s, oc))
        print(f"  {label:28s} 앞 {f:+.3f} 뒤 {b:+.3f} 9/1~ {s:+.3f} 10월 {oc:+.3f}")
    for buf in (7, 9, 12, 15, 20):
        run(f"rs60 주간 완충{buf}", score_at=rs_n(60), buffer=buf)
    for n in (40, 50, 70, 80, 120):
        run(f"rs{n} 주간 완충12", score_at=rs_n(n), buffer=12)
    for off in range(5):
        run(f"rs60 5거래일마다(시작+{off}) 완충12", score_at=rs_n(60), buffer=12,
            rebal=lambda k, k0, off=off: (k - k0 - off) % 5 == 0)
    run("rs60 주간 완충12 신호 하루 늦게(k-2)", score_at=rs_n(60), buffer=12, lag=2)
    nb = [r for r in rows if r[0] != "rs60 주간 완충12"]
    print(f"  이웃 {len(nb)}개 중 뒤 플러스 {sum(r[2] > 0 for r in nb)} · 뒤 QQQ(+0.021) 초과 {sum(r[2] > 0.021 for r in nb)} · "
          f"뒤 범위 {min(r[2] for r in nb):+.3f}~{max(r[2] for r in nb):+.3f}")

    # ---------------- 4) 다른 점수의 시점 점검
    print("\n[r160 날짜 정렬] 1시간봉 → 그날 마지막 봉(15:30 시작, 16:00 끝)")
    h1 = pd.read_parquet(os.path.join(CACHE, "1h.parquet"))
    for s in ("PLTR", "MU"):
        x = h1[h1.sym == s].set_index("ts")
        m = x.index.hour * 60 + x.index.minute
        x = x[(m >= 570) & (m < 960)]
        m = x.index.hour * 60 + x.index.minute
        g = np.where(m < 750, 0, np.where(m < 870, 1, 2))
        dkey = x.index.strftime("%Y-%m-%d")
        b = x.groupby([dkey, g]).Close.last()
        r = pd.Series(rsi(b.to_numpy()), index=b.index)
        last = r.groupby(level=0).last()
        for d in ("2026-10-07", "2026-10-08"):
            lastbar = x[dkey == d].index.max()
            print(f"  {s} {d}: 독립 {last[d]:.3f} 하네스 {H.r160[s][d]:.3f} · 그날 마지막 1시간봉 시작 {lastbar.strftime('%H:%M %Z')}")
        diff = (last.reindex(H.days) - H.r160[s]).abs().max()
        print(f"  {s} 전 기간 최대 차 {diff:.2e}")

    print("\n[hi(252) 그날 포함·앞날만] ")
    lg = pd.read_parquet(os.path.join(CACHE, "1d_long.parquet"))
    lg["d"] = pd.to_datetime(lg.ts).dt.strftime("%Y-%m-%d")
    for s, d in (("GTLB", "2026-10-02"), ("NTAP", "2026-09-30"), ("MU", "2026-07-15")):
        y = lg[(lg.sym == s) & (lg.d <= d)].sort_values("d").tail(252)
        print(f"  {s} {d}: 독립 {y.High.max():.3f}(마지막 행 {y.d.iloc[-1]}) 하네스 {H.hi(252)[s][d]:.3f}")
    near = H.cls / H.hi(252)
    print(f"  종가/hi252 > 1.0001 칸 {int((near > 1.0001).sum().sum())}")

    def near_at(Ct):
        d = Ct.index[-1]
        hi = H.hi(252).loc[d]                      # 행 d(=k-1)까지 고가 최고
        return Ct.iloc[-1] / hi
    out = indep_sim(days, O, C, H.eligible, score_at=near_at, buffer=12)
    dd, rr = [d for d, _, _ in out], [r for _, r, _ in out]
    jn = js["rank near252 주간+완충12"]
    print("\n[near252 주간+완충12 독립 재계산]")
    for lab, a, b in PERIODS:
        print(f"  {lab}: 독립 {period_tot(dd, rr, a, b):+.4f} / json {jn[lab]['수익']:+.4f}")

    # 일부러 미래를 쓴 판(시점 위반 크기 감 잡기)
    cheat = H.simulate(H.rank_picker(lambda k: H.rs(60).iloc[k], slots=SLOTS, buffer=12, weekly=True))
    ec = H.evaluate("cheat", cheat)
    print(f"\n[대조: k행(그날 종가)을 쓰면] 앞 {ec[FRONT]['수익']:+.3f} 뒤 {ec[BACK]['수익']:+.3f} — 보고값과 멀면 보고값엔 이 누수가 없다")
    print(f"완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
