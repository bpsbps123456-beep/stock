import pandas as pd
from evalcfg2 import evaluate2, VARIANTS
F = lambda x: f"{x:+.1%}"
P = dict(tf=240, eth=False, L=21, norm="rms", n=50, a=0.75, b=1.75)
rows = []
for name, (sm, bf, db) in VARIANTS.items():
    ml = evaluate2("long", **P, stop_mode=sm, buy_filter=bf, dip_buy=db)[0]
    ms = evaluate2("short", **P, stop_mode=sm, buy_filter=bf, dip_buy=db)[0]
    rows.append({"variant": name, "L_sel": ml["sel_ret"], "L_selsh": ml["sel_sh"], "L_val": ml["val_ret"], "L_valmdd": ml["val_mdd"],
                 "L_all": ml["all_ret"], "L_allsh": ml["all_sh"], "L_allmdd": ml["all_mdd"], "L_expo": ml["expo"],
                 "S_all": ms["all_ret"], "S_mdd": ms["all_mdd"], "S_sh": ms["all_sh"]})
t = pd.DataFrame(rows)
print("보유: 2026 선택 +254.1% / 확인 +6.1% (낙폭 -39.3%) / 전체 +275.6% 샤프 2.96 | 최근60일 +18.2% 낙폭 -21.1% 샤프 1.57")
print(t.to_string(index=False, formatters={c: F for c in t.columns if c not in ("variant", "L_selsh", "L_allsh", "S_sh")} | {c: "{:.2f}".format for c in ("L_selsh", "L_allsh", "S_sh")}))
