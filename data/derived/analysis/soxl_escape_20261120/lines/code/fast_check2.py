"""C2(연장 5분 EMA200 ±2%) 프리장 체결 현실성 점검 (2026-11-20).
비용: 정규장 전환 편도 0.05%, 프리·애프터 전환 편도 0.3%(호가 차이를 크게 잡음). 얇은 새벽(04:00~07:00 ET) 신호는 건너뜀.
결과: ../results/fast_check2.json"""
import json, os
import numpy as np, pandas as pd
import engine as E

OUT = os.path.join(E.ROOT, "results", "fast_check2.json")
B = E._base(); d1 = E.load("SOXL"); mod1 = d1["mod"].to_numpy()
L = lambda x: 100 * np.log1p(x / 100)   # noqa: E731


def score(tag, pos, corr, sw):
    out = {"name": tag}
    for w in ("SEL", "CONF", "Y2025"):
        a, z = E._win(w)
        s = sw[(sw >= a) & (sw < z)]
        reg = (mod1[s] >= 570) & (mod1[s] < 960)
        cost = 0.0005 * reg.sum() + 0.003 * (~reg).sum()
        r, p = B["r"][a:z], pos[a:z]
        f = p.mean()
        tot = (p * r + corr[a:z]).sum() - cost
        inc = p * r + corr[a:z]
        cc = np.zeros(z - a); np.add.at(cc, s - a, -np.where(reg, 0.0005, 0.003)); eq = np.exp(np.cumsum(inc + cc))
        out[w] = {"ret_net%": round(100 * np.expm1(tot), 2), "bh%": round(100 * np.expm1(r.sum()), 2), "tv_net%p": round(100 * (tot - f * r.sum()), 2),
                  "mdd%": round(100 * float((eq / np.maximum.accumulate(eq) - 1).min()), 2), "expo%": round(100 * f, 1),
                  "sw_reg": int(reg.sum()), "sw_ext": int((~reg).sum())}
    out["CONF"]["tvL_net%p"] = round(L(out["CONF"]["ret_net%"]) - out["SEL"]["expo%"] / 100 * L(out["CONF"]["bh%"]), 2)
    return out


rows = []
for n, band in ((200, 0.02), (100, 0.02)):
    b = E.bars("SOXL", 5, "ext"); C = b.Close; e = E.ema(C, n)
    x, en = C < e * (1 - band), C > e * (1 + band)
    endET = pd.DatetimeIndex(b["end"])
    em = endET.hour * 60 + endET.minute
    late = pd.Series(em >= 420, index=b.index)                       # 07:00 ET 이후 완성봉만
    kst = endET.tz_convert("Asia/Seoul")
    awake = pd.Series(kst.hour >= 8, index=b.index)
    tag = f"5분 EMA{n} ±{int(band*100)}%"
    rows.append(score(f"{tag} · 기본", *E.simulate(b, exit_close=x, entry_close=en, exec_sess="ext")))
    rows.append(score(f"{tag} · 07시(ET) 이후만", *E.simulate(b, exit_close=x & late, entry_close=en & late, exec_sess="ext")))
    rows.append(score(f"{tag} · 07시 이후 + 한국 08~24시만", *E.simulate(b, exit_close=x & late & awake, entry_close=en & late & awake, exec_sess="ext")))
bd = E.bars("SOXL", "D", "reg"); Cd = bd.Close; a, c = E.ema(Cd, 10), E.ema(Cd, 20)
rows.append(score("일봉 EMA10·20 종가 · 다음 정규장 시가", *E.simulate(bd, exit_close=Cd < np.minimum(a, c), entry_close=Cd > np.maximum(a, c), exec_sess="reg")))
json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50)
print(pd.DataFrame([dict(name=r["name"], S_ret=r["SEL"]["ret_net%"], S_tv=r["SEL"]["tv_net%p"], S_mdd=r["SEL"]["mdd%"], S_sw=f'{r["SEL"]["sw_reg"]}+{r["SEL"]["sw_ext"]}',
                         C_ret=r["CONF"]["ret_net%"], C_tv=r["CONF"]["tv_net%p"], C_tvL=r["CONF"]["tvL_net%p"], C_mdd=r["CONF"]["mdd%"],
                         Y_ret=r["Y2025"]["ret_net%"], Y_tv=r["Y2025"]["tv_net%p"], Y_mdd=r["Y2025"]["mdd%"]) for r in rows]).to_string(index=False))
