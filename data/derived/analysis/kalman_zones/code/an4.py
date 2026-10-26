import pandas as pd, numpy as np
from evalcfg import evaluate
pd.set_option("display.width", 250)
F = lambda x: f"{x:+.1%}"
rows = []
# (1) 2026년 창에서 고른 설정을 최근 60일 창(5분 단위 체결)에서 교차 확인
picks = [(60, 89, "rms", 100, 1.0, 1.75), (120, 34, "rms", 100, 0.75, 1.75), (180, 21, "rms", 100, 0.75, 1.75),
         (240, 21, "rms", 50, 0.75, 1.75), (360, 21, "rms", 50, 0.75, 1.75)]
print("(1) 2026년 창 선택값 → 최근 60일 창")
for tf, L, norm, n, a, b in picks:
    m = evaluate("short", tf, False, L, norm, n, a, b)[0]
    o = evaluate("short", tf, False, original=True)[0]
    print(f"  {tf:>3} RTH L{L} {norm}{n} a{a} b{b}: 전체 {F(m['all_ret'])} 낙폭 {F(m['all_mdd'])} 샤프 {m['all_sh']:.2f} | 확인 {F(m['val_ret'])} | 원본 전체 {F(o['all_ret'])} 낙폭 {F(o['all_mdd'])}")
# (2) 같은 평활 기간(약 10거래일)으로 맞춘 시간봉 비교: rms50 a0.75 b1.75, 절반 익절 + 패닉 보유
def bars_per_day(tf, eth):
    span = 960 if eth else 390
    return int(np.ceil(span / tf))
print("\n(2) 평활 ≈10거래일 고정, 시간봉만 바꿈 (rms50 a0.75 b1.75)")
for win in ("long", "short"):
    tfs = [60, 120, 180, 240, 360] if win == "long" else [15, 20, 30, 40, 60, 80, 90, 120, 160, 180, 240, 320, 360]
    for eth in ((False,) if win == "long" else (False, True)):
        for tf in tfs:
            L = max(5, round(10 * bars_per_day(tf, eth)))
            n = max(20, 5 * bars_per_day(tf, eth))  # RMS 창 ≈ 5거래일 이상
            m = evaluate(win, tf, eth, L, "rms", n, 0.75, 1.75)[0]
            rows.append({"win": win, "tf": tf, "ses": "ETH" if eth else "RTH", "L": L, "n": n, **m})
t = pd.DataFrame(rows)
print(t[["win", "tf", "ses", "L", "n", "sel_ret", "val_ret", "val_sh", "val_mdd", "all_ret", "all_sh", "all_mdd", "trades", "expo"]]
      .to_string(index=False, formatters={c: F for c in ["sel_ret", "val_ret", "val_mdd", "all_ret", "all_mdd", "expo"]} | {c: "{:.2f}".format for c in ["val_sh", "all_sh"]}))
t.to_csv("horizon_compare.csv", index=False)
