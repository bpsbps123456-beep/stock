"""'stops' 계열 감사(2026-10-10). deskx_stops.py 결과를 반박하려는 검증. desk/·deskx.py는 고치지 않는다.

1) 기본 Params()가 desk.strategies.run_model을 그대로 재현하는지.
2) trail20_cash(앞 기준 최선)를 원본 가격 CSV·TIME PDF로 따로 짠 모의로 다시 돌려 경로·손절 목록 대조.
3) 손절 2건 + 재조정 1건을 원본 자료로 직접 계산.
4) 기간 수익·월별 재확인, 기본 대비 차이를 종목·날짜로 분해, 9월 경로 분기 원인.
5) refill 경로 감사(손절 뒤 실제로 채웠는지), 메커니즘 민감도(감사용, 선택에 쓰지 않음).

실행: PYTHONIOENCODING=utf-8 py verify_deskx_stops.py   (이 폴더에서)
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deskx import Engine, Params, START, END, PERIODS, MONTHS, FROZEN
from desk import config
from desk.strategies import run_model, iso_week, us_weights, project_caps
from gbrc.etf.trades import load_fund_snapshots
from gbrc.etf.market import yahoo_ticker

STOPS_JSON = os.path.join(HERE, "../results/deskx/stops.json")
REPORTED = {"trail20_cash": dict(front=1.955, back=-0.067, full=1.758, mdd=-0.254, sepoct=0.159, jul=-0.174),
            "trail20_refill": dict(front=1.941, back=0.032, full=2.036, mdd=-0.223, sepoct=0.159, jul=-0.144)}


def hdr(t):
    print("\n" + "=" * 8, t, "=" * 8)


# ---------------------------------------------------------------- 독립 모의(원본 CSV + PDF)
class Raw:
    def __init__(self, sector):
        rd = lambda n: pd.read_csv(os.path.join(FROZEN, n), index_col=0)
        adj, opn, cls = rd("prices_adj_close.csv"), rd("prices_open.csv"), rd("prices_close.csv")
        self.days = [d[:10] for d in adj["QQQ"].dropna().index]
        fix = lambda df: df.set_axis([d[:10] for d in df.index]).reindex(self.days)
        self.adj, self.opn, self.cls = fix(adj), fix(opn), fix(cls)
        snaps = load_fund_snapshots("456600", "9999-12-31")
        self.pdfs = sorted(snaps)
        self.w = {d: us_weights(s) for d, s in snaps.items()}
        self.sector = sector

    def a(self, frame, key):
        tk = yahoo_ticker(key)
        df = getattr(self, frame)
        return df[tk].to_numpy(float) if tk in df.columns else None

    def pdf_for(self, day):
        before = [p for p in self.pdfs if p < day]          # 엄격히 이전 날짜
        return before[-1] if before else None

    def rank(self, k, day):
        pdf = self.pdf_for(day)
        rows = []
        if pdf is None:
            return pdf, rows
        for key, wt in sorted(self.w[pdf].items()):
            px = self.a("adj", key)
            if wt <= 0 or px is None or k - 121 < 0:
                continue
            p1, p0 = px[k - 1], px[k - 121]
            if np.isfinite(p1) and np.isfinite(p0) and p0 > 0:
                rows.append((key, p1 / p0 - 1, wt))
        rows.sort(key=lambda r: -r[1])
        return pdf, rows

    def choose(self, k, day, current, exclude=()):
        pdf, rows = self.rank(k, day)
        rows = [r for r in rows if r[0] not in exclude]
        if not rows:
            return pdf, rows, None, None
        order = [r[0] for r in rows]
        keep = [x for x in order[:12] if x in current][:7]
        for x in order:
            if len(keep) >= 7:
                break
            if x not in keep:
                keep.append(x)
        wm = {r[0]: r[2] for r in rows}
        tot = sum(wm[x] for x in keep)
        return pdf, rows, keep, project_caps({x: wm[x] / tot for x in keep}, self.sector)

    def peak(self, key, entry_k, k):
        """산 시가와 산 날부터 k-1일까지 종가 중 최고값을 매번 원본 배열에서 새로 잰다."""
        o, c = self.a("opn", key), self.a("cls", key)
        seg = c[entry_k:k]
        return np.nanmax(np.concatenate([[o[entry_k]], seg[np.isfinite(seg)]]))

    def run(self, trail=None, keep_buffer_after_stop=False, stop_on_reb=False, refill=None):
        """refill: None(현금) | 'engine'(deskx 방식 그대로: 순위 빈자리를 오늘 판 종목이 차지하면 못 채움)
                | 'fixed'(이번 주에 판 종목을 모두 순위에서 빼고 다음 종목으로 채움)."""
        k0 = self.days.index(next(d for d in self.days if d >= START))
        val, w, sel, entry, stops, hold = [1.0], {}, set(), {}, [], []
        started = False
        week_stopped = set()
        self.rebuy_same_week = []
        for k in range(k0, len(self.days)):
            day = self.days[k]
            if k > k0:
                r = {}
                for key in w:
                    o = self.a("opn", key)
                    r[key] = o[k] / o[k - 1] - 1 if np.isfinite(o[k]) and np.isfinite(o[k - 1]) and o[k - 1] > 0 else 0.0
                g = sum(w[x] * r[x] for x in w)
                tot = sum(w[x] * (1 + r[x]) for x in w) + 1 - sum(w.values())
                w = {x: w[x] * (1 + r[x]) / tot for x in w}
                val.append(val[-1] * (1 + g))
            reb = (not started) or iso_week(day) != iso_week(self.days[k - 1])
            if trail is not None and (not reb or stop_on_reb):
                sold = []
                for key in sorted(w):
                    c = self.a("cls", key)
                    if np.isfinite(c[k - 1]) and c[k - 1] <= self.peak(key, entry[key], k) * (1 - trail):
                        stops.append((day, yahoo_ticker(key)))
                        sold.append(key)
                for key in sold:
                    w.pop(key)
                    entry.pop(key)
                    if not keep_buffer_after_stop:
                        sel.discard(key)
                week_stopped |= set(sold)
                if sold and refill and not reb:
                    excl = week_stopped if refill == "fixed" else ()
                    _, _, keep, tgt = self.choose(k, day, sel, exclude=excl)
                    for key in keep or []:
                        if key not in w and key not in sold and len(w) < 7:
                            add = min(tgt.get(key, 0.0), 1 - sum(w.values()))
                            if add > 1e-9:
                                if key in week_stopped:
                                    self.rebuy_same_week.append((day, yahoo_ticker(key)))
                                w[key] = add
                                sel.add(key)
                                entry[key] = k
            if reb:
                _, _, keep, tgt = self.choose(k, day, sel)
                if keep is None:
                    hold.append(dict(w))
                    continue
                started = True
                week_stopped = set()
                for key in tgt:
                    if key not in w:
                        entry[key] = k
                for key in list(entry):
                    if key not in tgt:
                        entry.pop(key)
                w, sel = dict(tgt), set(keep)
            hold.append(dict(w))
        return pd.Series(val, index=self.days[k0:]), stops, hold


def periods(v):
    r = (v.shift(-1) / v - 1).dropna()
    out = {}
    for lab, a, b in PERIODS:
        x = r[(r.index >= a) & (r.index <= b)]
        cur = (1 + x).cumprod()
        out[lab] = (float(cur.iloc[-1] - 1), float((cur / cur.cummax() - 1).min()))
    out["월별"] = {m: float((1 + r[r.index.str.startswith(m)]).prod() - 1) for m in MONTHS}
    return out


def contrib(E, rows_a, rows_b, a, b):
    """같은 날 시가→다음 시가 수익을 종목별로: (w_a - w_b) × r. 단순 합(복리 무시)."""
    out = {}
    days = [r["day"] for r in rows_a]
    for i in range(len(days) - 1):
        d = days[i]
        if not (a <= d <= b):
            continue
        k = E.days.index(d)
        for key in set(rows_a[i]["w"]) | set(rows_b[i]["w"]):
            o = E.col("opn", key)
            r = o[k + 1] / o[k] - 1 if o is not None and np.isfinite(o[k + 1]) and np.isfinite(o[k]) and o[k] > 0 else 0.0
            dlt = (rows_a[i]["w"].get(key, 0.0) - rows_b[i]["w"].get(key, 0.0)) * r
            out[E.ticker[key]] = out.get(E.ticker[key], 0.0) + dlt
    return dict(sorted(out.items(), key=lambda t: -abs(t[1])))


def main():
    E = Engine()
    raw = Raw(E.sector)

    hdr("1. 기본 재현")
    v0, rows0 = E.run(Params())
    ref = run_model(E.book, start=START)
    print("deskx 기본 vs desk run_model 최대 차:", float(np.max(np.abs(v0.to_numpy() - np.asarray(ref.value)))))
    tk = "SNDK"
    a = raw.opn[tk].to_numpy(float); b = E.book.opn[tk].to_numpy(float)
    print("엔진이 읽은 SNDK 시가 = 동결 CSV:", bool(np.allclose(a, b, equal_nan=True)))
    vb, _, _ = raw.run(trail=None)
    print("독립 모의(손절 없음) vs run_model 최대 차:", float(np.max(np.abs(vb.to_numpy() - np.asarray(ref.value)))))

    hdr("2. trail20_cash 독립 재계산")
    v20, rows20 = E.run(Params(trail=0.20))
    vi, stops_i, hold_i = raw.run(trail=0.20)
    print("경로 최대 차(엔진 vs 독립):", float(np.max(np.abs(v20.to_numpy() - vi.to_numpy()))))
    js = json.load(open(STOPS_JSON, encoding="utf-8"))
    rep = {e["name"]: e for e in js["results"]}
    st_rep = sorted((s["day"], s["ticker"]) for s in rep["trail20_cash"]["stops"])
    print("손절 목록 일치:", st_rep == sorted(stops_i), f"(보고 {len(st_rep)}회, 독립 {len(stops_i)}회)")
    for nm in ("trail20_cash", "trail20_refill"):
        v, rows = (v20, rows20) if nm == "trail20_cash" else E.run(Params(trail=0.20, trail_refill=True))
        p = periods(v)
        got = dict(front=p[PERIODS[0][0]][0], back=p[PERIODS[1][0]][0], sepoct=p[PERIODS[2][0]][0],
                   full=p[PERIODS[3][0]][0], mdd=p["전체"][1], jul=p["월별"]["2026-07"])
        ok = all(abs(got[x] - REPORTED[nm][x]) < 0.0015 for x in got)
        print(nm, {x: round(y, 4) for x, y in got.items()}, "보고값과 일치" if ok else "불일치!")
    pi = periods(vi)
    print("독립 경로 기간:", {lab: round(pi[lab][0], 4) for lab, _, _ in PERIODS}, "전체MDD", round(pi["전체"][1], 4))

    hdr("3. 손절 판정 직접 계산(원본 CSV)")
    def show_stop(day, tkr):
        k = raw.days.index(day)
        js_ = E.days.index(day) - E.days.index(START)
        key = next(x for x in rows20[js_ - 1]["w"] if E.ticker[x] == tkr)
        # rows는 '그날 시가 거래 뒤' 보유. 연속 보유가 시작된 날 = 진입일(손절은 보유에서 빼므로 다시 산 날이 새 진입일)
        i = js_ - 1
        while i > 0 and key in rows20[i - 1]["w"]:
            i -= 1
        ek = raw.days.index(rows20[i]["day"])
        c = raw.cls[tkr].to_numpy(float); o = raw.opn[tkr].to_numpy(float)
        pk = raw.peak(key, ek, k)
        pk_day = raw.days[ek + int(np.nanargmax(c[ek:k]))] if np.nanmax(c[ek:k]) >= o[ek] else raw.days[ek] + "(시가)"
        prev_pk = raw.peak(key, ek, k - 1)
        print(f"{day} {tkr}: 진입 {raw.days[ek]} 시가 {o[ek]:.2f}, 최고 {pk:.2f}({pk_day}), "
              f"{raw.days[k-1]} 종가 {c[k-1]:.2f} = 최고의 {c[k-1]/pk:.3f} (문턱 0.800) → {raw.days[k]} 시가 {o[k]:.2f}에 매도; "
              f"전날 {raw.days[k-2]} 종가/최고 {c[k-2]/prev_pk:.3f}"
              + ("  ※전날은 재조정일(손절 판정 안 함)" if iso_week(raw.days[k - 1]) != iso_week(raw.days[k - 2]) else ""))
    for d, t in (("2026-07-07", "SNDK"), ("2026-07-30", "ARM"), ("2026-02-11", "SNDK")):
        show_stop(d, t)

    print("\n재조정 2026-07-13(trail20_cash):")
    k = raw.days.index("2026-07-13")
    sel_before = {x for x in rows20[E.days.index("2026-07-10") - E.days.index(START)]["w"]}
    pdf, rows, _, _ = raw.choose(k, "2026-07-13", set())
    print(" 사용 PDF:", pdf, "(세션 2026-07-13보다 앞)", "| 가격 기준일:", raw.days[k - 1], "종가")
    print(" 7/10 보유(손절 뒤):", sorted(E.ticker[x] for x in sel_before))
    print(" 순위 1~12:", [(E.ticker[r[0]], round(r[1], 3)) for r in rows[:12]])
    # 엔진의 selected 집합은 7/6 재조정 선택에서 손절 종목을 뺀 것
    sel_06 = set(rows20[E.days.index("2026-07-06") - E.days.index(START)]["w"])
    stopped = {x for x in sel_06 if x not in sel_before}
    _, _, keep, tgt = raw.choose(k, "2026-07-13", sel_06 - stopped)
    eng = rows20[E.days.index("2026-07-13") - E.days.index(START)]["w"]
    print(" 직접 계산 선택:", sorted(E.ticker[x] for x in keep), "| 엔진 보유:", sorted(E.ticker[x] for x in eng))
    print(" 비중 일치:", all(abs(tgt.get(x, 0) - eng.get(x, 0)) < 1e-12 for x in set(tgt) | set(eng)))

    hdr("4. 견고성: 월별 차이·종목 분해")
    res = {e["name"]: e for e in js["results"]}
    base = res["기본 #774"]
    print("월(6~10) 기본 대비 차이:")
    for nm, e in res.items():
        if nm == "기본 #774":
            continue
        d = [e["월별"][m] - base["월별"][m] for m in MONTHS[5:]]
        print(f"  {nm:<16}", " ".join(f"{x:+.3f}" for x in d), f"| 6~10월 중 이긴 달 {sum(x > 1e-9 for x in d)}/5")
    vr, rowsr = E.run(Params(trail=0.20, trail_refill=True))
    for nm, rr in (("trail20_cash", rows20), ("trail20_refill", rowsr)):
        for lab, a, b in (("7월", "2026-07-01", "2026-07-31"), ("뒤 6/1~10/9", "2026-06-01", "2026-10-09"),
                          ("앞 1/2~5/29", "2026-01-02", "2026-05-29")):
            c = contrib(E, rr, rows0, a, b)
            tot = sum(c.values())
            print(f"  {nm} {lab}: 차이 합 {tot:+.3f}, 큰 종목 " + ", ".join(f"{t} {x:+.3f}" for t, x in list(c.items())[:5]))
    # 날짜 집중도: 뒤 구간 일별 수익 차 상위
    r0 = (v0.shift(-1) / v0 - 1).dropna()
    for nm, v in (("trail20_cash", v20), ("trail20_refill", vr)):
        dr = ((v.shift(-1) / v - 1).dropna() - r0)
        back = dr[(dr.index >= "2026-06-01")]
        top = back.abs().sort_values(ascending=False).index[:5]
        print(f"  {nm} 뒤 구간 일별 차 상위 5일:", ", ".join(f"{d} {back[d]:+.3f}" for d in top),
              f"| 5일 합 {back[top].sum():+.3f} / 전체 합 {back.sum():+.3f}")

    print("\n9월 경로: 9월 손절 수 =", sum(s[0].startswith("2026-09") for s in stops_i))
    same = sum(set(rows20[i]["w"]) == set(rows0[i]["w"]) for i, r in enumerate(rows0) if r["day"].startswith("2026-09"))
    nsep = sum(r["day"].startswith("2026-09") for r in rows0)
    i = next(i for i, r in enumerate(rows0) if r["day"] == "2026-09-01")
    print(f"  9월 {nsep}일 중 기본과 보유 종목 집합이 같은 날 {same}일")
    print("  9/1 기본:", sorted(E.ticker[x] for x in rows0[i]["w"]), "| trail20_cash:", sorted(E.ticker[x] for x in rows20[i]["w"]))

    hdr("5. refill 감사")
    for t in (0.10, 0.15, 0.20):
        v, rows = E.run(Params(trail=t, trail_refill=True))
        days = [r["day"] for r in rows]
        stop_days = filled = blocked = 0
        for i in range(1, len(rows)):
            if iso_week(days[i]) != iso_week(days[i - 1]):
                continue
            out = set(rows[i - 1]["w"]) - set(rows[i]["w"])
            inn = set(rows[i]["w"]) - set(rows[i - 1]["w"])
            if out:
                stop_days += 1
                filled += bool(inn)
                # 채움이 없을 때: 그날 순위에서 보유 밖 첫 후보가 방금 판 종목인지
                if not inn:
                    k = E.days.index(days[i])
                    sel = set(rows[i]["w"])
                    keep, _, ranked = E.target(Params(trail=t, trail_refill=True), k, days[i], sel)
                    blocked += bool(set(keep) & out)
        print(f"  trail{int(t*100)}_refill: 손절한 날 {stop_days}일, 바로 채운 날 {filled}일, 못 채운 날 {stop_days - filled}일"
              f"(그중 판 종목이 순위 빈자리를 차지해 막힌 날 {blocked}일)")

    print("\n  독립 모의로 deskx refill 재현 + 고친 refill(이번 주 판 종목을 순위에서 빼고 채움):")
    fixed = {}
    for t in (0.10, 0.15, 0.20):
        ve, _ = E.run(Params(trail=t, trail_refill=True))
        vr_e, st_e, _ = raw.run(trail=t, refill="engine")
        same_week = list(raw.rebuy_same_week)
        vf, st_f, _ = raw.run(trail=t, refill="fixed")
        p = periods(vf)
        fixed[t] = p
        print(f"    trail{int(t*100)}: deskx vs 독립(engine 방식) 최대 차 {float(np.max(np.abs(ve.to_numpy() - vr_e.to_numpy()))):.1e}; "
              f"deskx가 같은 주에 손절한 종목을 다시 산 경우 {len(same_week)}건 {same_week[:4]}")
        print(f"      고친 refill: 앞 {p[PERIODS[0][0]][0]:.3f} 뒤 {p[PERIODS[1][0]][0]:.3f} 9/1~ {p[PERIODS[2][0]][0]:.3f} "
              f"전체 {p['전체'][0]:.3f} MDD {p['전체'][1]:.3f} 뒤MDD {p[PERIODS[1][0]][1]:.3f} 손절 {len(st_f)}회 | 월 6~10 "
              + " ".join(f"{p['월별'][m]:+.3f}" for m in MONTHS[5:]))

    vf, st_f, hold_f = raw.run(trail=0.20, refill="fixed")
    rows_f = [{"day": d, "w": h} for d, h in zip(raw.days[E.days.index(START):], hold_f)]
    for lab, a, b in (("7월", "2026-07-01", "2026-07-31"), ("뒤 6/1~10/9", "2026-06-01", "2026-10-09")):
        c = contrib(E, rows_f, rows0, a, b)
        print(f"    고친 trail20_refill {lab}: 차이 합 {sum(c.values()):+.3f}, 큰 종목 " + ", ".join(f"{t} {x:+.3f}" for t, x in list(c.items())[:5]))
    add_cash = sum(bool(set(rows20[i]["w"]) - set(rows20[i - 1]["w"])) for i in range(1, len(rows20))
                   if iso_week(rows20[i]["day"]) == iso_week(rows20[i - 1]["day"]))
    print(f"  trail20_cash: 재조정 아닌 날 새로 산 경우 {add_cash}건(0이면 판 종목은 다음 재조정까지 현금)")

    hdr("5b. 재조정일 사각지대·손절 뒤 수익")
    k0 = E.days.index(START)
    late = 0
    for d, tkr in stops_i:
        k = raw.days.index(d)
        if iso_week(raw.days[k - 1]) != iso_week(raw.days[k - 2]):   # 전날이 재조정일
            js_ = k - k0
            key = next(x for x in rows20[js_ - 1]["w"] if E.ticker[x] == tkr)
            i = js_ - 1
            while i > 0 and key in rows20[i - 1]["w"]:
                i -= 1
            ek = raw.days.index(rows20[i]["day"])
            if ek <= k - 2:
                c = raw.cls[tkr].to_numpy(float)
                if c[k - 2] <= raw.peak(key, ek, k - 1) * 0.8:
                    late += 1
    print(f"  trail20_cash 손절 {len(stops_i)}회 중 재조정 전 종가에 이미 문턱을 깼는데 재조정에서 다시 담았다가 다음 날 판 경우: {late}회")
    st = rep["trail20_cash"]["stops"]
    rr = [s["ret_to_next_reb"] for s in st if s["ret_to_next_reb"] is not None]
    print(f"  손절 뒤 다음 재조정까지 그 종목 수익: 평균 {np.mean(rr):+.3f}, 중앙 {np.median(rr):+.3f}, 손절이 도움(음수) {sum(x < 0 for x in rr)}/{len(rr)}, "
          f"다음 재조정에서 다시 산 경우 {sum(s['rebought_next_reb'] for s in st)}/{len(st)}")

    hdr("6. 메커니즘 민감도(감사용, 고르기에 쓰지 않음)")
    for lab, kw in (("판 종목도 다음 재조정 때 버퍼 유지", dict(keep_buffer_after_stop=True)),
                    ("재조정일에도 손절 판정(판 뒤 버퍼 잃음)", dict(stop_on_reb=True))):
        v, st, _ = raw.run(trail=0.20, **kw)
        p = periods(v)
        print(f"  trail20_cash + {lab}: 앞 {p[PERIODS[0][0]][0]:.3f} 뒤 {p[PERIODS[1][0]][0]:.3f} "
              f"9/1~ {p[PERIODS[2][0]][0]:.3f} 전체 {p['전체'][0]:.3f} MDD {p['전체'][1]:.3f} 7월 {p['월별']['2026-07']:.3f} 손절 {len(st)}회")
    print("  격자 끝 너머(이웃 값 확인): ")
    for t in (0.25, 0.30):
        for rf in (False, True):
            v, _ = E.run(Params(trail=t, trail_refill=rf))
            p = periods(v)
            print(f"    trail{int(t*100)}_{'refill' if rf else 'cash'}: 앞 {p[PERIODS[0][0]][0]:.3f} 뒤 {p[PERIODS[1][0]][0]:.3f} "
                  f"전체 {p['전체'][0]:.3f} MDD {p['전체'][1]:.3f} 7월 {p['월별']['2026-07']:.3f}")
    p0 = periods(v0)
    print(f"    기본: 앞 {p0[PERIODS[0][0]][0]:.3f} 뒤 {p0[PERIODS[1][0]][0]:.3f} 전체 {p0['전체'][0]:.3f} MDD {p0['전체'][1]:.3f}")


if __name__ == "__main__":
    main()
