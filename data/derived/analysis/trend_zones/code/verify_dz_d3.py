"""dip_by_zone D3 숫자 재계산·부트스트랩 t 안정성·원래 rules.sim 대조 (2026-10-11, 검증용). 결과 results/verify_dz_d3.txt"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dip_by_zone as DZ  # noqa
from rules import sim as rsim, hot_days  # noqa  (dip_by_zone이 경로를 넣어 둠)
from events import bars, prep  # noqa

RES = os.path.join(HERE, "../results")
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def boot_t(a, b, col, n=2000, seed=0):
    """날짜 묶음 부트스트랩 — dip_by_zone과 독립 구현(일별 합·건수로 빠르게)."""
    da = a.groupby("day")[col].agg(["sum", "count"])
    db = b.groupby("day")[col].agg(["sum", "count"])
    days = np.array(sorted(set(da.index) | set(db.index)))
    A = da.reindex(days).fillna(0).to_numpy()
    B = db.reindex(days).fillna(0).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), (n, len(days)))
    sa, ca = A[idx, 0].sum(1), A[idx, 1].sum(1)
    sb, cb = B[idx, 0].sum(1), B[idx, 1].sum(1)
    ok = (ca > 0) & (cb > 0)
    bs = sa[ok] / ca[ok] - sb[ok] / cb[ok]
    diff = a[col].mean() - b[col].mean()
    return diff, diff / bs.std()


def main():
    T = pd.read_csv(os.path.join(RES, "dip_trades.csv"), encoding="utf-8-sig")
    T["day"] = pd.to_datetime(T["day"]).dt.date
    T["b3"] = (T.mae <= -0.03) * 1.0
    P("== D3 재계산 (dip_trades.csv)")
    for scope in ("24시간", "정규장만"):
        for w in T.win.unique():
            x = T[T.win == w]
            if scope == "정규장만":
                x = x[x.ses == "정규"]
            up1, dn1 = x[x.zone == 1], x[x.zone == -1]
            upc, dnc = x[x.zone.isin([1, 2])], x[x.zone.isin([-1, -2])]
            P(f" {w} {scope}: 상방 {len(up1)}건 평균 {up1.ret.mean()*100:+.2f}% -3%↓ {up1.b3.mean()*100:.0f}% | "
              f"하방 {len(dn1)}건 평균 {dn1.ret.mean()*100:+.2f}% -3%↓ {dn1.b3.mean()*100:.0f}% | 상방계 {len(upc)} 하방계 {len(dnc)}")
            for col in ("ret", "b3"):
                ts = []
                for sd in range(10):
                    d, t = boot_t(upc, dnc, col, seed=sd)
                    ts.append(t)
                d0, t0 = DZ.clustered_diff(upc, dnc, col)
                d1, t1 = boot_t(up1, dn1, col)
                P(f"    {col}: 상방계−하방계 {d0*100:+.2f}%p, 원 코드 t {t0:+.2f}, 내 구현 t 10개 시드 {min(ts):+.2f}~{max(ts):+.2f}"
                  f" | 상방−하방(극 제외) {d1*100:+.2f}%p t {t1:+.2f}")
            wa, nw = x[x.warn], x[~x.warn]
            d, t = boot_t(nw, wa, "ret")
            P(f"    SOXX 평상−경고: {nw.ret.mean()*100:+.2f} vs {wa.ret.mean()*100:+.2f} (차 {d*100:+.2f}%p, t {t:+.2f}),"
              f" 경고 날짜 {wa.day.nunique()}일, 평상 날짜 {nw.day.nunique()}일")
    P("\n== 최악 거래 10건(두 창)")
    w10 = T.nsmallest(10, "ret")[["win", "sym", "t", "ses", "ret", "mae", "zone", "warn"]]
    P(w10.to_string())
    P("하방계 최악:", T[T.zone.isin([-1, -2])].groupby("win").ret.min().round(4).to_dict())
    P("수익 -10% 이하 건수 단계별:", T[T.ret <= -0.10].groupby(["win", "zone"]).size().to_dict())
    # 9/9 EMA 예열
    x = T[(T.win.str.startswith("3분")) & (T.day == date(2026, 9, 9))]
    P(f"\n3분 창 9/9 거래 {len(x)}건, 평균 {x.ret.mean()*100:+.2f}% (1분봉 9/9 04:00 시작 → EMA60 예열 61봉뿐)")
    # 원래 rules.sim(hot 7%)로 3분 창 V0 재현
    S = DZ.syms()
    n, tot = 0, 0.0
    for s in S:
        d = prep(bars(DZ.RAW, s, 3), 60, 14)
        hot = hot_days(DZ.RAW, s, d, 0.07)
        for ei, xi, ent, px, why in rsim(d, hot=hot):
            if date(2026, 9, 9) <= d.index[ei].date() <= date(2026, 10, 9):
                n += 1
                tot += px / ent - 1
    c3 = T[T.win.str.startswith("3분")]
    P(f"원래 dip_reason rules.sim(hot 7%) 3분 창: {n}건 합 {tot*100:+.1f}% | csv V0 {len(c3)}건 합 {c3.ret.sum()*100:+.1f}%")
    with open(os.path.join(RES, "verify_dz_d3.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
