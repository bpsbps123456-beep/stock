"""critic: improve 플레인 MU 1위(20분 연장 N10 count follow 0.7/0.3, 밤 보유)의 체결 시각·손익분기 비용·깨어 있음 판·뒤 절반 안정성."""
import numpy as np, pandas as pd, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from engine import load, tf_bars, ratio, segments, awake_mask, positions, sim, WIN, bh
H = pathlib.Path(__file__).resolve().parent.parent
d = load("MU", "5m"); win = WIN["5m"]
tb = tf_bars(d, 20, "eth"); tb = tb[tb.day >= win["warm"]].reset_index(drop=True)
O, C, V = (tb[k].values.astype(float) for k in "OCV")
seg = segments(tb.day.values, win); lod = tb.lod.values
p = ratio(O, C, V, 10, "count")
pct = lambda x: (np.exp(x) - 1) * 100
ones = np.ones(len(tb), np.bool_); aw = awake_mask(tb.start)
for name, a in (("ideal", ones), ("awake", aw)):
    for cost in (0, 0.0005, 0.001, 0.002, 0.003):
        out = np.zeros(11); sim(O, C, p, lod, seg, a, 0.7, 0.3, True, False, cost, out)
        print(name, cost, f"sel {pct(out[0]):+.1f}% conf {pct(out[1]):+.1f}% ent {int(out[2])}/{int(out[3])} hold {out[5]/out[7]:.2f} mdd_conf {pct(out[9]):+.1f}%")
pos, ev = positions(O, C, p, lod, ones, 0.7, 0.3, True, False)
t = pd.DatetimeIndex(tb.start)
m = t.hour * 60 + t.minute
sess = np.where(m < 570, "pre", np.where(m < 960, "rth", "post"))
e = pd.DataFrame({"t": t, "ev": ev, "sess": sess, "seg": seg})
e = e[e.ev != 0]
print(e.groupby(["seg", "sess", "ev"]).size().unstack(fill_value=0))
# 확인 구간 손익을 체결 세션별로: 매매 단위 수익
trades = []; ent = None
for j in range(len(O)):
    if ev[j] == 1: ent = j
    elif ev[j] == -1 and ent is not None:
        trades.append((t[ent], sess[ent], t[j], sess[j], np.log(O[j] / O[ent]), seg[ent])); ent = None
tr = pd.DataFrame(trades, columns=["t_in", "s_in", "t_out", "s_out", "lr", "seg"])
print(tr.groupby(["seg"]).lr.agg(["count", "sum", "median"]))
c = tr[tr.seg == 1].sort_values("lr")
print("conf 상위 3 매매 기여(로그%):", (c.lr.tail(3) * 100).round(1).tolist(), "합 전체", round(c.lr.sum() * 100, 1))
# 확인 구간 반으로
days = np.array(sorted(set(tb.day[seg == 1]))); mid = days[len(days) // 2]
for lab, msk in (("conf1", (seg == 1) & (tb.day.values < mid)), ("conf2", (seg == 1) & (tb.day.values >= mid))):
    s2 = np.where(msk, 1, -1).astype(np.int64)
    out = np.zeros(11); sim(O, C, p, lod, s2, ones, 0.7, 0.3, True, False, 0.0, out)
    r = np.r_[np.log(O[1:] / O[:-1]), 0.0]
    print(lab, f"strat {pct(out[1]):+.1f}% bh {pct(r[msk].sum()):+.1f}%")
