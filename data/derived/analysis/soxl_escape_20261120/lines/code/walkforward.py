"""이동 전방 검증 + 최근 분기 성적 (2026-11-20, AGENTS.md 연구 보고 규칙 개정 반영).

원칙(사용자 2026-11-20): 고정 규칙 하나 대신 '최근 창에서 통한 것을 바로 다음 창에 쓰는' 검증을 함께 보고하고, 결론에 가장 최근 분기 성적을 따로 적는다.
사전 선언
- 후보: 이평선 724개(ma_scan.configs) + 두 줄 띠 60개(ma_band와 같은 격자) = 784개.
- 달마다 직전 L개월(L = 1, 3, 6)의 시점 값 합(편도 0.1% 비용 차감)이 가장 큰 후보를 골라 다음 한 달에 쓴다. 후보를 바꿀 때 보유 상태가 달라지면 0.1% 한 번 더 뺀다.
- 비교: 그냥 보유, 고정 느린 선(정규 일봉 EMA10·20 종가), 고정 빠른 선(연장 5분 EMA200 ±2%), 같은 평균 비중 고정.
- 구간: 2026 전체(01-02~11-19), 가장 최근 분기(08-20~11-19), 급락 구간(10-15~11-19), 2025(보조).
- 추가(첫 결과를 본 뒤): 지표의 두 규칙(느린 선·빠른 선) 사이에서만 직전 L개월 시점 값이 큰 쪽으로 바꾸는 경우.
결과: ../results/walkforward.json
"""
import json, os
import numpy as np, pandas as pd
import engine as E
import ma_scan as M

OUT = os.path.join(E.ROOT, "results", "walkforward.json")
COST = 0.001
B = E._base()
R1 = B["r"]
N1 = len(R1)
DATES = pd.DatetimeIndex(B["date"])
WIN = {"2026": ("2026-01-02", "2026-11-19"), "최근 분기": ("2026-08-20", "2026-11-19"),
       "급락 구간": ("2026-10-15", "2026-11-19"), "2025": ("2025-03-03", "2025-12-31")}


def band_configs():
    c = []
    for tf in (120, 240, "D"):
        for sess in ("reg", "ext"):
            for a, b in ((10, 20), (10, 50), (20, 50), (20, 100), (50, 100)):
                for mode in ("close", "touch"):
                    c.append(dict(fam="R", sym="SOXL", tf=tf, sess=sess, a=a, c=b, mode=mode))
    return c


def run(c):
    if c["fam"] != "R":
        return M.run(c)
    b = E.bars("SOXL", c["tf"], c["sess"])
    C = b["Close"]
    ea, ec = E.ema(C, c["a"]), E.ema(C, c["c"])
    W, R = np.minimum(ea, ec), np.maximum(ea, ec)
    if c["mode"] == "close":
        return E.simulate(b, exit_close=C < W, entry_close=C > R, exec_sess=c["sess"])
    return E.simulate(b, exit_touch=W.shift(1), entry_close=C > R, exec_sess=c["sess"])


def name(c):
    return M.name(c) if c["fam"] != "R" else f"R:{c['mode']}:tf={c['tf']},sess={c['sess']},pair=({c['a']},{c['c']})"


# 달 경계(1분봉 위치). 달 m의 수익 = 그 달 첫 1분봉부터 다음 달 첫 1분봉 직전까지(마지막 봉의 밤사이 갭 포함)
months = pd.period_range("2025-01", "2026-11", freq="M")
starts = [int(np.searchsorted(DATES.values, np.datetime64(p.start_time.date()))) for p in months] + [N1 - 1]
NM = len(months)

cfgs = M.configs() + band_configs()
stat = {k: np.zeros((len(cfgs), NM)) for k in ("lr", "ex", "bh", "nsw", "p0", "p1", "tv")}
for i, c in enumerate(cfgs):
    pos, corr, sw = run(c)
    inc = pos * R1 + corr
    cs = np.r_[0.0, np.cumsum(inc)]
    cp = np.r_[0.0, np.cumsum(pos)]
    cr = np.r_[0.0, np.cumsum(R1)]
    for m in range(NM):
        a, z = starts[m], starts[m + 1]
        n = max(z - a, 1)
        stat["lr"][i, m] = cs[z] - cs[a]
        stat["ex"][i, m] = (cp[z] - cp[a]) / n
        stat["bh"][i, m] = cr[z] - cr[a]
        stat["nsw"][i, m] = int(((sw >= a) & (sw < z)).sum())
        stat["p0"][i, m] = pos[a]
        stat["p1"][i, m] = pos[z - 1]
stat["tv"] = stat["lr"] - stat["ex"] * stat["bh"] - COST * stat["nsw"]

res = {"configs": len(cfgs), "months": [str(p) for p in months], "variants": {}}
names = [name(c) for c in cfgs]
I_SLOW = names.index("R:close:tf=D,sess=reg,pair=(10,20)")
I_FAST = names.index("A:SOXL:tf=5,sess=ext,kind=EMA,n=200,band=0.02")
for pool_name, pool in (("784개 중", np.arange(len(cfgs))), ("두 규칙 중", np.array([I_SLOW, I_FAST]))):
  for L in (1, 3, 6):
    sel = {}
    for m in range(L, NM):
        score = stat["tv"][pool, m - L:m].sum(axis=1)
        sel[m] = int(pool[int(np.argmax(score))])
    # 고른 후보들의 1분봉 경로를 이어 붙인다
    need = sorted(set(sel.values()))
    paths = {i: run(cfgs[i]) for i in need}
    pos = np.zeros(N1); corr = np.zeros(N1); sws = []
    prev_state = None
    for m in range(L, NM):
        i = sel[m]
        a, z = starts[m], starts[m + 1]
        p, c_, s = paths[i]
        pos[a:z] = p[a:z]; corr[a:z] = c_[a:z]
        sws += [int(x) for x in s[(s >= a) & (s < z)]]
        if prev_state is not None and p[a] != prev_state:
            sws.append(a)                                   # 후보를 바꿔 생긴 매매
        prev_state = p[z - 1]
    sws = np.array(sorted(set(sws)), dtype=int)
    first = starts[L]
    pos[:first] = 1.0                                       # 첫 선택 전은 보유(평가 구간 밖)
    ev = {w: E.evaluate(pos, corr, sws, rng, COST) for w, rng in WIN.items()}
    picks = [{"month": str(months[m]), "pick": names[sel[m]], "picked_tv_next%p": round(100 * float(stat["tv"][sel[m], m]), 2)} for m in range(L, NM)]
    res["variants"][f"{pool_name} 직전 {L}개월 최고"] = {"eval": ev, "picks": picks, "distinct_picks": len(need)}

# 고정 규칙과 보유
bd = E.bars("SOXL", "D", "reg"); Cd = bd.Close; a10, a20 = E.ema(Cd, 10), E.ema(Cd, 20)
slow = E.simulate(bd, exit_close=Cd < np.minimum(a10, a20), entry_close=Cd > np.maximum(a10, a20), exec_sess="reg")
b5 = E.bars("SOXL", 5, "ext"); C5 = b5.Close; e200 = E.ema(C5, 200)
fast = E.simulate(b5, exit_close=C5 < e200 * 0.98, entry_close=C5 > e200 * 1.02, exec_sess="ext")
res["fixed"] = {"느린 선(기본)": {w: E.evaluate(*slow, rng, COST) for w, rng in WIN.items()},
                "빠른 선": {w: E.evaluate(*fast, rng, COST) for w, rng in WIN.items()}}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)

rows = []
for nm, ev in [(k, v["eval"]) for k, v in res["variants"].items()] + list(res["fixed"].items()):
    r = {"규칙": nm}
    for w in WIN:
        e = ev[w]
        r[f"{w} 수익%"] = e["ret%"]; r[f"{w} 낙폭%"] = e["mdd%"]; r[f"{w} 시점값"] = e["tv%p"]; r[f"{w} 비중%"] = e["expo%"]
    rows.append(r)
e = res["fixed"]["느린 선(기본)"]
rows.append({"규칙": "그냥 보유", **{f"{w} 수익%": e[w]["bh%"] for w in WIN}, **{f"{w} 낙폭%": e[w]["bh_mdd%"] for w in WIN}})
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
t = pd.DataFrame(rows)
print(t[["규칙"] + [c for w in WIN for c in (f"{w} 수익%", f"{w} 낙폭%", f"{w} 시점값")]].to_string(index=False))
for k, v in res["variants"].items():
    print(k, "고른 후보 종류", v["distinct_picks"])
    for p in v["picks"][-6:]:
        print("  ", p)
