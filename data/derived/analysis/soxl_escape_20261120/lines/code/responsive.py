"""더 기민한 경고선·재매수선 (2026-11-20, 사용자: "대피선과 재매수를 조금 더 자주 나오게, 지금보다 기민하게").

사전 선언(결과를 보기 전에 정함)
- 기준 봉: 정규장 일봉(사용자가 따라 할 수 있는 시간). 종가 신호는 다음 정규장 첫 1분봉 시가(겨울 한국 23:30) 체결.
  '닿으면'은 전날 확정된 경고선에 정규장 중 닿는 순간 체결(자는 동안 걸어 둘 수 있는 손절 주문, docs/desk_system.md §1).
- 후보 66개 = 계열 33개 × 대피 방식 2개(종가, 닿으면)
  band  두 줄 띠: 경고선 min(EMA a, EMA b), 재매수선 max(EMA a, EMA b) — (a,b) 15쌍 a∈{3,5,8,10} < b∈{10,15,20,30}
  asym  따로 지정: 경고선 EMA x, 재매수선 EMA y — x≠y ∈ {3,5,10,20} 12쌍
  single 한 줄: 경고선 EMA n×(1−띠), 재매수선 EMA n — n∈{5,10,20} × 띠{0,1%} 6개
- 자격: 고르는 구간(2026-01-02~10-14) 전환 35~100번(지금 기본값 21번의 1.7배 이상, 한 달 10번 이하).
- 순위: 고르는 구간의 '가장 좋은 3일을 뺀 시점 값'(편도 0.1% 차감). 큰 날 몇 번에 기대는 규칙을 거른다(Codex 반박 근거).
- 확인(선택에 쓰지 않음): 급락 구간 10-15~11-19, 가장 최근 분기 08-20~11-19, 2026 전체, 2025(보조), 이웃(길이 한 단계), 위약 백분위.
- 이동 전방 검증: 같은 66개 안에서 직전 3·6개월 시점 값 1위를 다음 달에 쓰기.
결과: ../results/responsive.json
"""
import json, os
import numpy as np, pandas as pd
import engine as E

OUT = os.path.join(E.ROOT, "results", "responsive.json")
COST = 0.001
WIN = {"SEL": ("2026-01-02", "2026-10-14"), "CONF": ("2026-10-15", "2026-11-19"), "Q": ("2026-08-20", "2026-11-19"),
       "Y26": ("2026-01-02", "2026-11-19"), "Y2025": ("2025-03-03", "2025-12-31")}
L = lambda x: 100 * np.log1p(x / 100)   # noqa: E731
b = E.bars("SOXL", "D", "reg")
C = b["Close"]
EMA = {n: E.ema(C, n) for n in (3, 5, 8, 10, 15, 20, 30)}


def configs():
    out = []
    for a in (3, 5, 8, 10):
        for c in (10, 15, 20, 30):
            if a < c:
                out.append(dict(kind="band", a=a, b=c))
    for x in (3, 5, 10, 20):
        for y in (3, 5, 10, 20):
            if x != y:
                out.append(dict(kind="asym", x=x, y=y))
    for n in (5, 10, 20):
        for band in (0.0, 0.01):
            out.append(dict(kind="single", n=n, band=band))
    return [dict(k, mode=m) for k in out for m in ("close", "touch")]


def lines(k):
    if k["kind"] == "band":
        return np.minimum(EMA[k["a"]], EMA[k["b"]]), np.maximum(EMA[k["a"]], EMA[k["b"]])
    if k["kind"] == "asym":
        return EMA[k["x"]], EMA[k["y"]]
    return EMA[k["n"]] * (1 - k["band"]), EMA[k["n"]]


def run(k):
    W, R = lines(k)
    kw = dict(exit_close=C < W) if k["mode"] == "close" else dict(exit_touch=W.shift(1))
    return E.simulate(b, **kw, entry_close=C > R, exec_sess="reg")


def name(k):
    m = "종가" if k["mode"] == "close" else "닿으면"
    if k["kind"] == "band":
        return f"띠 EMA{k['a']}·{k['b']} {m}"
    if k["kind"] == "asym":
        return f"경고 EMA{k['x']}·재매수 EMA{k['y']} {m}"
    return f"한 줄 EMA{k['n']}" + (f" −{int(k['band']*100)}%" if k["band"] else "") + f" {m}"


def score(pos, corr, sw):
    ev = {w: E.evaluate(pos, corr, sw, rng, COST) for w, rng in WIN.items()}
    f = ev["SEL"]["expo%"] / 100
    for w in ("CONF", "Q"):
        ev[w]["tvL%p"] = round(L(ev[w]["ret%"]) - f * L(ev[w]["bh%"]), 2)     # 평소(고르는 구간) 비중 기준
    return ev


cfgs = configs()
rows, paths = [], {}
for i, k in enumerate(cfgs):
    p = run(k)
    paths[i] = p
    ev = score(*p)
    rows.append({"i": i, "name": name(k), "cfg": k, **{w: ev[w] for w in WIN}})

def flat(r):
    S, Cn, Q, Y, Y5 = (r[w] for w in ("SEL", "CONF", "Q", "Y26", "Y2025"))
    return {"규칙": r["name"], "전환/월": round(S["switches"] / 9.4, 1), "고른구간 시점값": S["tv%p"], "고른구간 큰3일뺌": S["tv_ex_top3%p"],
            "고른구간 수익%": S["ret%"], "고른구간 낙폭%": S["mdd%"], "급락 tvL": Cn["tvL%p"], "급락 수익%": Cn["ret%"],
            "최근분기 수익%": Q["ret%"], "최근분기 낙폭%": Q["mdd%"], "최근분기 tvL": Q["tvL%p"],
            "2026 수익%": Y["ret%"], "2026 낙폭%": Y["mdd%"], "2025 시점값": Y5["tv%p"], "2025 수익%": Y5["ret%"]}

t = pd.DataFrame([flat(r) for r in rows])
t["i"] = range(len(t))
sw = np.array([r["SEL"]["switches"] for r in rows])
elig = t[(sw >= 35) & (sw <= 100)].sort_values("고른구간 큰3일뺌", ascending=False)
ref = t[t["규칙"].isin(["띠 EMA10·20 종가", "띠 EMA10·20 닿으면"])]
win_i = int(elig.iloc[0]["i"])
wk = cfgs[win_i]

# 이웃: 같은 계열·방식에서 길이 한 단계
grid = {"band_a": [3, 5, 8, 10], "band_b": [10, 15, 20, 30], "asym": [3, 5, 10, 20], "single": [5, 10, 20]}
def neighbors(k):
    out = []
    def step(lst, v):
        j = lst.index(v)
        return [lst[x] for x in (j - 1, j + 1) if 0 <= x < len(lst)]
    if k["kind"] == "band":
        out += [dict(k, a=v) for v in step(grid["band_a"], k["a"]) if v < k["b"]]
        out += [dict(k, b=v) for v in step(grid["band_b"], k["b"]) if v > k["a"]]
    elif k["kind"] == "asym":
        out += [dict(k, x=v) for v in step(grid["asym"], k["x"]) if v != k["y"]]
        out += [dict(k, y=v) for v in step(grid["asym"], k["y"]) if v != k["x"]]
    else:
        out += [dict(k, n=v) for v in step(grid["single"], k["n"])]
        out += [dict(k, band=0.01 if k["band"] == 0 else 0.0)]
    out.append(dict(k, mode="touch" if k["mode"] == "close" else "close"))
    return out
nb = []
for k in neighbors(wk):
    j = next(ii for ii, c in enumerate(cfgs) if c == k)
    nb.append(flat(rows[j]))
plc = {w: E.placebo(*paths[win_i][:2], w) for w in ("SEL", "CONF")}

# 이동 전방 검증(66개 안)
B = E._base(); N1 = len(B["r"]); DATES = pd.DatetimeIndex(B["date"])
months = pd.period_range("2025-01", "2026-11", freq="M")
starts = [int(np.searchsorted(DATES.values, np.datetime64(p.start_time.date()))) for p in months] + [N1 - 1]
NM = len(months)
tvm = np.zeros((len(cfgs), NM))
for i in range(len(cfgs)):
    pos, corr, s = paths[i]
    inc = pos * B["r"] + corr
    for m in range(NM):
        a, z = starts[m], starts[m + 1]
        n = int(((s >= a) & (s < z)).sum())
        tvm[i, m] = inc[a:z].sum() - pos[a:z].mean() * B["r"][a:z].sum() - COST * n
wf = {}
for LB in (3, 6):
    pos = np.ones(N1); corr = np.zeros(N1); sws = []; prev = None
    for m in range(LB, NM):
        i = int(np.argmax(tvm[:, m - LB:m].sum(axis=1)))
        a, z = starts[m], starts[m + 1]
        p, c_, s = paths[i]
        pos[a:z] = p[a:z]; corr[a:z] = c_[a:z]
        sws += [int(x) for x in s[(s >= a) & (s < z)]]
        if prev is not None and p[a] != prev:
            sws.append(a)
        prev = p[z - 1]
    sws = np.array(sorted(set(sws)), dtype=int)
    wf[f"66개 중 직전 {LB}개월 1위"] = {w: E.evaluate(pos, corr, sws, WIN[w], COST) for w in ("Y26", "Q", "CONF", "Y2025")}

res = {"configs": len(cfgs), "eligible": int(len(elig)), "winner": name(wk), "winner_cfg": wk, "ranking": elig.to_dict("records"),
       "reference": ref.to_dict("records"), "neighbors": nb, "placebo": plc, "walkforward": wf, "all": rows}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)

pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30)
cols = ["규칙", "전환/월", "고른구간 큰3일뺌", "고른구간 시점값", "고른구간 수익%", "고른구간 낙폭%", "급락 tvL", "최근분기 수익%", "최근분기 낙폭%", "2026 수익%", "2026 낙폭%", "2025 수익%", "2025 시점값"]
print("자격", len(elig), "/", len(t))
print(elig[cols].head(10).to_string(index=False))
print("--- 지금 기본값"); print(ref[cols].to_string(index=False))
print("--- 1위 이웃"); print(pd.DataFrame(nb)[cols].to_string(index=False))
print("--- 위약", plc)
for k, v in wf.items():
    print(k, {w: (v[w]["ret%"], v[w]["mdd%"]) for w in v})
print("세 확인 모두 양수(급락 tvL·최근분기 tvL·2025 시점값) 비율(자격 안):",
      round(float(((elig["급락 tvL"] > 0) & (elig["최근분기 tvL"] > 0) & (elig["2025 시점값"] > 0)).mean()), 2))
