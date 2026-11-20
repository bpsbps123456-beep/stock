"""Claude 마무리 점검 — Codex가 keep으로 판정한 C2(연장 5분 EMA200 ±2% 종가)와 느린 선 C4(정규 일봉 EMA10·20 종가)의 실행 조건 민감도 (2026-11-20).
1) 체결을 정규장으로만 제한 2) 대피·복귀 모두 한국시간 08~24시에만(그 밖의 신호는 건너뛰고, 깨어 있는 첫 봉에서 조건이 여전히 참이면 실행)
3) 전환이 일어난 세션 분포 4) 같은 길이(약 하루) 이평선의 이웃(봉 크기·세션) 5) 편도 비용 0.1%
결과: ../results/fast_check.json"""
import json, os
import numpy as np, pandas as pd
import engine as E

OUT = os.path.join(E.ROOT, "results", "fast_check.json")
WINS = ("SEL", "CONF", "Y2025")
L = lambda x: 100 * np.log1p(x / 100)   # noqa: E731


def awake_mask(b):
    """봉 완성 시각(=체결 시각)이 한국시간 08:00~24:00 인가."""
    k = pd.DatetimeIndex(b["end"]).tz_convert("Asia/Seoul")
    return pd.Series(k.hour >= 8, index=b.index)


def row(tag, pos, corr, sw, cost=0.001):
    out = {"name": tag}
    S = E.evaluate(pos, corr, sw, "SEL")
    for w in WINS:
        e = E.evaluate(pos, corr, sw, w)
        en = E.evaluate(pos, corr, sw, w, cost)
        out[w] = {"ret%": e["ret%"], "ret_net%": en["ret%"], "bh%": e["bh%"], "tv%p": e["tv%p"], "tv_net%p": round(e["tv%p"] - 100 * cost * e["switches"], 2),
                  "mdd%": e["mdd%"], "bh_mdd%": e["bh_mdd%"], "expo%": e["expo%"], "switches": e["switches"]}
    Cn = out["CONF"]
    out["CONF"]["tvL_net%p"] = round(L(Cn["ret%"]) - S["expo%"] / 100 * L(Cn["bh%"]) - 100 * cost * Cn["switches"], 2)
    return out


res = {"variants": [], "neighbors": []}
d1 = E.load("SOXL")
# ── C2 ──
b = E.bars("SOXL", 5, "ext"); C = b.Close; e = E.ema(C, 200)
x, n = C < e * 0.98, C > e * 1.02
base = E.simulate(b, exit_close=x, entry_close=n, exec_sess="ext")
res["variants"].append(row("C2 기본(연장 체결)", *base))
res["variants"].append(row("C2 체결을 정규장으로만", *E.simulate(b, exit_close=x, entry_close=n, exec_sess="reg")))
aw = awake_mask(b)
res["variants"].append(row("C2 복귀만 08~24시", *E.simulate(b, exit_close=x, entry_close=n & aw, exec_sess="ext")))
res["variants"].append(row("C2 대피·복귀 모두 08~24시", *E.simulate(b, exit_close=x & aw, entry_close=n & aw, exec_sess="ext")))
# 자는 동안은 스탑 주문: 잠들 때(한국 24시 직전 완성봉)의 경고선 값에 스탑을 걸어 둔 것으로 근사 → 그 값에 닿으면 대피
k = pd.DatetimeIndex(b["end"]).tz_convert("Asia/Seoul")
bed = pd.Series(np.where((k.hour == 23) & (k.minute >= 55) | ((k.hour == 0) & (k.minute == 0)), (e * 0.98).to_numpy(), np.nan), index=b.index).ffill()
stop_lvl = bed.where(~aw.to_numpy())          # 자는 시간 봉에서만 유효한 고정 스탑
res["variants"].append(row("C2 깨어 있을 때 종가 판정 + 자는 동안 고정 스탑", *E.simulate(b, exit_close=x & aw, entry_close=n & aw, exit_touch=stop_lvl, exec_sess="ext")))
# 전환 세션 분포(2026)
pos, corr, sw = base
m = d1["mod"].to_numpy()[sw]; yr = d1.index.year.to_numpy()[sw] == 2026
res["c2_switch_session_2026"] = {"pre(04:00~09:30)": int(((m < 570) & yr).sum()), "regular": int(((m >= 570) & (m < 960) & yr).sum()), "after(16:00~20:00)": int(((m >= 960) & yr).sum())}
kk = pd.DatetimeIndex(d1.index[sw]).tz_convert("Asia/Seoul")
res["c2_switch_awake_share_2026%"] = round(100 * float(((kk.hour >= 8) & yr).sum() / yr.sum()), 1)
# ── C4 ──
bd = E.bars("SOXL", "D", "reg"); Cd = bd.Close; a, c = E.ema(Cd, 10), E.ema(Cd, 20); W, R = np.minimum(a, c), np.maximum(a, c)
res["variants"].append(row("C4 일봉 EMA10·20 종가(다음 정규장 시가)", *E.simulate(bd, exit_close=Cd < W, entry_close=Cd > R, exec_sess="reg")))
# 프리장 개장(04:00 ET, 겨울 한국 18:00)에 실행하면
res["variants"].append(row("C4 같은 신호를 다음 날 프리장 첫 봉에 체결", *E.simulate(bd, exit_close=Cd < W, entry_close=Cd > R, exec_sess="ext")))
# ── 이웃: '약 하루 길이 이평선 ±2% 띠'를 다른 봉 크기·세션으로 ──
for tf, sess, nn in ((5, "ext", 200), (5, "ext", 100), (5, "ext", 400), (15, "ext", 64), (30, "ext", 32), (60, "ext", 16), (5, "reg", 78), (5, "reg", 156), (15, "reg", 26), (30, "reg", 13), (60, "reg", 7)):
    for band in (0.01, 0.02, 0.03):
        bb = E.bars("SOXL", tf, sess); cc = bb.Close; ee = E.ema(cc, nn)
        res["neighbors"].append(row(f"tf={tf},{sess},EMA{nn},band={band}", *E.simulate(bb, exit_close=cc < ee * (1 - band), entry_close=cc > ee * (1 + band), exec_sess=sess)))
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
def tab(rows):
    return pd.DataFrame([dict(name=r["name"], S_ret=r["SEL"]["ret_net%"], S_tvnet=r["SEL"]["tv_net%p"], S_mdd=r["SEL"]["mdd%"], S_sw=r["SEL"]["switches"],
                              C_ret=r["CONF"]["ret_net%"], C_tvnet=r["CONF"]["tv_net%p"], C_tvL=r["CONF"]["tvL_net%p"], C_mdd=r["CONF"]["mdd%"], C_expo=r["CONF"]["expo%"],
                              Y_ret=r["Y2025"]["ret_net%"], Y_tvnet=r["Y2025"]["tv_net%p"], Y_mdd=r["Y2025"]["mdd%"]) for r in rows])
print(tab(res["variants"]).to_string(index=False))
print(res["c2_switch_session_2026"], "깨어 있는 시간 전환 비율", res["c2_switch_awake_share_2026%"])
t = tab(res["neighbors"]); print(t.to_string(index=False))
print("이웃 세 구간 모두 양수(순비용)", round(float(((t.S_tvnet > 0) & (t.C_tvL > 0) & (t.Y_tvnet > 0)).mean()), 2), "| ext만", round(float(((t.S_tvnet > 0) & (t.C_tvL > 0) & (t.Y_tvnet > 0))[t.name.str.contains("ext")].mean()), 2), "| reg만", round(float(((t.S_tvnet > 0) & (t.C_tvL > 0) & (t.Y_tvnet > 0))[t.name.str.contains("reg")].mean()), 2))
