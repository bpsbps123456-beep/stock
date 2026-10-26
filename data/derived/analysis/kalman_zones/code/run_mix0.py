"""수수료·스프레드 0%: 추세 지표 vs 초단타(종가-1%, +0.5%, 1세션) vs 이전안(종가-2%, +1%, 2세션) vs 자금 나누기. 느슨/엄격 체결."""
import numpy as np, pandas as pd
from core import *
from grid_short import portfolio, window_metrics
from run_on import ctx, day_arrays, levels
from run_v2 import exec_masks
from sim2 import simulate_v2
from sim_on import sim_overnight
from evalcfg import data, indicator
D, HALF = data()


def curves(win, margin):
    S, common, sel = ctx(win)
    for s in SYMS:
        day_arrays(S, s)
    out = {"추세": [], "초단타(-1%,+0.5%,1세션)": [], "이전안(-2%,+1%,2세션)": []}
    fills = {"초단타(-1%,+0.5%,1세션)": 0, "이전안(-2%,+1%,2세션)": 0}
    for s in SYMS:
        A = S[s]
        tb = tf_bars(D[(s, "1h")], 240, False, HALF)
        K, u, z = indicator(tb, 21, "rms", 50, 0.75, 1.75)
        ex, aw = exec_masks(S, s)["M1"]
        nn = len(tb)
        e, *_ = simulate_v2(A["o"], A["h"], A["l"], A["c"], ex, aw, A["is_end"], A["j0"], map_completed(A["start_ns"], tb.end_ns.values),
                            z, K + 1.75 * u, K - 1.75 * u, True, True, 0.0, np.full(len(A["o"]), -1, np.int64), np.zeros(nn, bool),
                            np.full(nn, 50.0), np.full(nn, np.nan), 0, 0.0, 0.01, 6, 0, 0, ex, 0.0, margin)
        out["추세"].append(e)
        for name, lvname, tg, hd in (("초단타(-1%,+0.5%,1세션)", "종가-1%", 0.005, 1), ("이전안(-2%,+1%,2세션)", "종가-2%", 0.01, 2)):
            e, t, *_ = sim_overnight(A["o"], A["h"], A["l"], A["c"], A["is_reg"], A["is_pre"], A["is_awake"], A["is_end"], A["first_reg"],
                                     A["day_idx"], A["j0"], levels(S, s, win)[(lvname, "없음")], tg, hd, 0.0, 0.0, margin,
                                     np.full(len(A["days"]), np.nan), 0.005, 0.03)
            out[name].append(e); fills[name] += t
    bh = portfolio([buy_hold(S[s]["o"], S[s]["c"], S[s]["is_reg"], S[s]["is_end"], S[s]["j0"]) for s in SYMS], S)
    return {k: portfolio(v, S) for k, v in out.items()}, bh, sel, {k: v / len(common) for k, v in fills.items()}


F = lambda x: f"{x:+.0%}"
for margin, lab in ((0.0, "느슨(지정가에 닿으면 체결)"), (0.001, "엄격(0.1% 넘게 뚫어야 체결)")):
    print(f"\n=== 수수료·스프레드 0%, 체결 {lab} ===")
    for win, wl in (("long", "2026"), ("y2025", "2025"), ("short", "최근60일")):
        cv, bh, sel, fpd = curves(win, margin)
        mb = window_metrics(bh, sel)
        parts = [f"보유 {F(mb['all_ret'])}({F(mb['all_mdd'])})"]
        for k, v in cv.items():
            m = window_metrics(v, sel)
            parts.append(f"{k} {F(m['all_ret'])}({F(m['all_mdd'])}, 샤프 {m['all_sh']:.2f})")
        for w in (0.7, 0.5):
            m = window_metrics(w * cv["추세"] + (1 - w) * cv["초단타(-1%,+0.5%,1세션)"], sel)
            parts.append(f"추세{int(w*100)}+초단타{int(100-w*100)} {F(m['all_ret'])}({F(m['all_mdd'])})")
        print(f" {wl}: " + " | ".join(parts) + f" | 하루 체결 {fpd['초단타(-1%,+0.5%,1세션)']:.1f}회")
