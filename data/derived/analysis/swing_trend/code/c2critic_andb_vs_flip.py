"""반박 검산: and_b가 flip(뒤집기 단독)보다 나은가. c2trade_actual.csv의 종목별 수익으로 비교."""
import numpy as np, pandas as pd
a = pd.read_csv("results/c2trade_actual.csv")
a = a[a.stock & (a.cost == 0.0)]
for per in ["P1", "P2", "Y25", "Y24", "Y23"]:
    w = a[a.per == per].pivot_table(index="sym", columns="strat", values="ret")
    xx = a[a.per == per].pivot_table(index="sym", columns="strat", values="x")
    d = (w.and_b - w.flip).dropna()
    same = (np.abs(w.and_b - w.flip) < 1e-12).mean()
    print(per, f"n={len(d)} median(and_b-flip)={d.median()*100:+.1f}%p mean={d.mean()*100:+.1f}%p win={(d>0).mean():.0%} identical={same:.0%}",
          f"x and_b={xx.and_b.median():.2f} flip={xx.flip.median():.2f} zz={xx.zz.median():.2f}",
          f"| ex_sw med and_b={(w.and_b - a[a.per==per].pivot_table(index='sym',columns='strat',values='sw_ret').and_b).median()*100:+.1f}",
          f"flip={(w.flip - a[a.per==per].pivot_table(index='sym',columns='strat',values='sw_ret').flip).median()*100:+.1f}")
