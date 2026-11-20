"""Claude 갈래 — 이평선 경고선·재매수선 (2026-11-20). 엔진: engine.py, 규약: ../BRIEF.md

계열(합계 724개)
 A 단일 이평선+띠 504: tf{5,15,30,60,120,240,D} × 세션{reg,ext} × {EMA,SMA} × n{5,10,20,50,100,200} × 띠{0,1,2%}
     종가 < 선×(1−띠) 대피, 종가 > 선×(1+띠) 복귀
 B 두 이평선 비대칭 96: tf{15,60,240,D} × 세션 2 × (대피선 n, 복귀선 n) 순서쌍{10,20,50,100} 12개 (EMA)
 C 이평 교차 상태 40: tf{15,60,240,D} × 세션 2 × (빠른,느린){(5,20),(10,50),(20,100),(10,25),(25,60)} — 빠른>느린이면 보유
 D SOXX 이평선 48: tf{60,240,D} × 세션 2 × EMA n{10,20,50,100} × 띠{0,0.5%} — SOXX 종가로 판정, SOXL 체결
 T 닿으면 대피(스탑) 36: tf{60,240,D} × 세션 2 × EMA n{10,20,50} × 띠{0,1%} — 직전 봉 선×(1−띠)에 닿으면 스탑 매도, 복귀는 종가 > 선
결과: ../results/ma_scan.json (전체 표 + 자격 통과 순위 + SEL→CONF 이어짐 통계)
"""
import json, os, time
import numpy as np, pandas as pd
import engine as E

OUT = os.path.join(E.ROOT, "results", "ma_scan.json")
WINS = ("SEL", "CONF", "Y2025", "B1", "B2")


def line(b, kind, n):
    return E.ema(b["Close"], n) if kind == "EMA" else E.sma(b["Close"], n)


def configs():
    c = []
    for tf in (5, 15, 30, 60, 120, 240, "D"):
        for sess in ("reg", "ext"):
            for kind in ("EMA", "SMA"):
                for n in (5, 10, 20, 50, 100, 200):
                    for band in (0.0, 0.01, 0.02):
                        c.append(dict(fam="A", sym="SOXL", tf=tf, sess=sess, kind=kind, n=n, band=band))
    for tf in (15, 60, 240, "D"):
        for sess in ("reg", "ext"):
            for nx in (10, 20, 50, 100):
                for ne in (10, 20, 50, 100):
                    if nx != ne:
                        c.append(dict(fam="B", sym="SOXL", tf=tf, sess=sess, nx=nx, ne=ne))
            for f, s in ((5, 20), (10, 50), (20, 100), (10, 25), (25, 60)):
                c.append(dict(fam="C", sym="SOXL", tf=tf, sess=sess, f=f, s=s))
    for tf in (60, 240, "D"):
        for sess in ("reg", "ext"):
            for n in (10, 20, 50, 100):
                for band in (0.0, 0.005):
                    c.append(dict(fam="D", sym="SOXX", tf=tf, sess=sess, kind="EMA", n=n, band=band))
            for n in (10, 20, 50):
                for band in (0.0, 0.01):
                    c.append(dict(fam="T", sym="SOXL", tf=tf, sess=sess, kind="EMA", n=n, band=band))
    return c


def run(c):
    b = E.bars(c["sym"], c["tf"], c["sess"])
    C = b["Close"]
    f = c["fam"]
    if f in ("A", "D"):
        ln = line(b, c["kind"], c["n"])
        return E.simulate(b, exit_close=C < ln * (1 - c["band"]), entry_close=C > ln * (1 + c["band"]), exec_sess=c["sess"])
    if f == "B":
        return E.simulate(b, exit_close=C < E.ema(C, c["nx"]), entry_close=C > E.ema(C, c["ne"]), exec_sess=c["sess"])
    if f == "C":
        ef, es = E.ema(C, c["f"]), E.ema(C, c["s"])
        return E.simulate(b, exit_close=ef < es, entry_close=ef > es, exec_sess=c["sess"])
    if f == "T":
        ln = line(b, c["kind"], c["n"])
        return E.simulate(b, exit_touch=(ln * (1 - c["band"])).shift(1), entry_close=C > ln, exec_sess=c["sess"])
    raise ValueError(f)


def name(c):
    k = {k: v for k, v in c.items() if k not in ("fam", "sym")}
    return c["fam"] + ":" + c["sym"] + ":" + ",".join(f"{a}={b}" for a, b in k.items())


def main():
    t0 = time.time()
    rows = []
    for c in configs():
        pos, corr, sw = run(c)
        sc = E.score_all(pos, corr, sw, WINS)
        rows.append({"name": name(c), "cfg": c, **{w: sc[w] for w in WINS}})
    df = pd.DataFrame([{"name": r["name"], "fam": r["cfg"]["fam"], "tf": str(r["cfg"]["tf"]), "sess": r["cfg"]["sess"],
                        **{f"{w}_{k}": r[w][k] for w in WINS for k in ("tv%p", "tv_t", "ret%", "fixed%", "mdd%", "fixed_mdd%", "expo%", "switches")}}
                       for r in rows])
    elig = df[(df["SEL_switches"] >= 10) & (df["SEL_expo%"].between(20, 90))].copy()
    elig = elig.sort_values("SEL_tv%p", ascending=False)
    stats = {
        "configs": int(len(df)), "eligible": int(len(elig)),
        "SEL_tv_pos_share%": round(100 * float((elig["SEL_tv%p"] > 0).mean()), 1),
        "CONF_tv_pos_share%": round(100 * float((elig["CONF_tv%p"] > 0).mean()), 1),
        "Y2025_tv_pos_share%": round(100 * float((elig["Y2025_tv%p"] > 0).mean()), 1),
        "rank_corr_SEL_CONF": round(float(elig["SEL_tv%p"].rank().corr(elig["CONF_tv%p"].rank())), 3),
        "rank_corr_SEL_Y2025": round(float(elig["SEL_tv%p"].rank().corr(elig["Y2025_tv%p"].rank())), 3),
        "SEL_tv_median": round(float(elig["SEL_tv%p"].median()), 2), "CONF_tv_median": round(float(elig["CONF_tv%p"].median()), 2),
        "top_decile_SEL_CONF_tv_median": round(float(elig.head(max(1, len(elig) // 10))["CONF_tv%p"].median()), 2),
    }
    by = {}
    for key in ("fam", "tf", "sess"):
        g = elig.groupby(key)[["SEL_tv%p", "CONF_tv%p", "Y2025_tv%p", "SEL_mdd%", "CONF_mdd%", "CONF_expo%"]].median().round(2)
        g["n"] = elig.groupby(key).size()
        by[key] = g.reset_index().to_dict("records")
    json.dump({"stats": stats, "by": by, "ranking": elig.to_dict("records"), "all": rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)
    print(json.dumps(stats, ensure_ascii=False))
    for key in by:
        print(key, pd.DataFrame(by[key]).to_string(index=False))
    cols = ["name", "SEL_tv%p", "SEL_tv_t", "SEL_ret%", "SEL_mdd%", "SEL_expo%", "SEL_switches", "CONF_tv%p", "CONF_ret%", "CONF_fixed%", "CONF_mdd%", "CONF_expo%", "Y2025_tv%p"]
    print(elig[cols].head(25).to_string(index=False))
    print(round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
