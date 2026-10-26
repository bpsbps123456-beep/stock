"""같은 규칙을 5분봉 60일(7월 급락 포함 여부 확인)로: EMA 36(=3분봉 EMA 60과 같은 180분) -2%, RSI 14/9 > 70 매도.
3분봉 규칙의 근사다. 자는 동안 처리는 hold(그대로 보유)와 bed(자기 전 보라선 지정가, 깨면 취소)."""
import numpy as np, pandas as pd
from osc import rsi
SYMS = ["AMD", "ARM", "DELL", "INTC", "SNDK", "STX", "WDC"]
P = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/e846c1a2-c78d-40ed-885f-777d23bba52f/scratchpad/kalman/data"


def prep(s, L, rl):
    b = pd.read_parquet(f"{P}/{s}_5m.parquet")
    et = b.index
    m = et.hour * 60 + et.minute
    kst = et.tz_convert("Asia/Seoul")
    return dict(o=b.Open.values, h=b.High.values, l=b.Low.values, c=b.Close.values,
                ema=b.Close.ewm(span=L, adjust=False).mean().values, r=rsi(b.Close.values, rl),
                reg=np.asarray((m >= 570) & (m < 960)), pre=np.asarray((m >= 240) & (m < 570)),
                awake=np.asarray(kst.hour >= 8), kst=kst, et=et)


def sim(d, g, x, mode):
    o, h, l, c, ema, r, reg, pre, awake = (d[k] for k in ("o", "h", "l", "c", "ema", "r", "reg", "pre", "awake"))
    n = len(c); out = []; hold = False; lim = np.nan; ent = ei = 0
    for i in range(40, n):
        ses = reg[i] or pre[i]
        if hold:
            if awake[i] and not (i + 1 < n and not awake[i + 1]):
                lim = np.nan                                     # 깨어 있으면 지정가 취소(직접 관리)
            px = np.nan
            if r[i - 1] > x and i - 1 >= ei and ses and awake[i]:
                px, why = o[i], "rsi"
            elif not np.isnan(lim) and reg[i] and h[i] >= lim:
                px, why = max(o[i], lim), "lim"
            if not np.isnan(px):
                out.append((ei, i, ent, px, why)); hold = False; lim = np.nan; continue
            if mode == "bed" and awake[i] and i + 1 < n and not awake[i + 1]:
                lim = max(ema[i], c[i] * 1.001)
            continue
        if ses and awake[i] and l[i] <= ema[i - 1] * (1 - g):
            ent, ei, hold, lim = min(o[i], ema[i - 1] * (1 - g)), i, True, np.nan
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for rl in (14, 9):
        for mode in ("hold", "bed"):
            rows = []
            for s in SYMS:
                d = prep(s, 36, rl)
                for ei, xi, ent, px, why in sim(d, 0.02, 70, mode):
                    rows.append((s, d["et"][ei], px / ent - 1, (xi - ei) * 5, d["l"][ei:xi + 1].min() / ent - 1, why))
            t = pd.DataFrame(rows, columns=["sym", "et", "ret", "min", "mae", "why"])
            t["mon"] = t.et.map(lambda z: z.strftime("%m"))
            print(f"RSI{rl} {mode}: {len(t)}회 승률 {(t.ret > 0).mean():.0%} 평균 {t.ret.mean():+.2%} 중앙 {t.ret.median():+.2%} "
                  f"최악 {t.ret.min():+.2%} 보유중앙 {t['min'].median():.0f}분 MAE중앙 {t.mae.median():+.2%} 지정가 {(t.why == 'lim').mean():.0%}")
            print("   월별:", {k: f"{len(v)}회 {v.ret.mean():+.2%}" for k, v in t.groupby("mon")})
    d = prep("SNDK", 36, 14)
    print("5분봉 범위:", d["et"][0], "->", d["et"][-1])
