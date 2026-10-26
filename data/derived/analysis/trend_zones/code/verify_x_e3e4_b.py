"""검증 B 보충: (a) 자기 종목 20일선 위/아래 기본 순위(모멘텀) 대비 초과로 E4 다시 보기, (b) 우연 개수 순열을 고쳐서 다시.
verify_x_e3e4.py의 첫 순열은 종목별 맥락(자기 종목·자기 20일선 기울기·같은 섹터)을 같은 구간 안 다른 주로 옮겨서
'그 구간에 자주 20일선 아래였던 종목 = 그 구간 약한 종목'이라는 사후 정보가 섞였다(혼자 같은 부호 58%). 그래서
시장 전체 맥락 8개는 주 블록 순서 섞기, 종목별 맥락 3개는 같은 날 안에서 유효 종목끼리 이름표 섞기로 바꾼다.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/verify_x_e3e4_b.py
결과: results/verify_x_e3e4_b.txt
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_x_e3e4 as X  # noqa: E402  (본 검증을 그대로 다시 돌리고 객체를 빌림)

L = []


def log(s=""):
    print(s, flush=True)
    L.append(s)


rng = np.random.default_rng(7)
V, T, N, PN, R10 = X.V, X.T, X.N, X.PN, X.R10
rk = X.rk10
own = X.MY["자기 종목<20일선"]
sd = X.SD["아래"]

# (a) 기본 순위: 20일선 아래/위 모든 종목의 날짜 평균 순위
log("== (a) 극하방 자기 종목<20일선: 기본(그 상태 모든 종목) 대비 초과로 본 차")
base = X.ctx_table(sd, own, R10)
for p in PN:
    r = R10[p]
    fin = np.isfinite(rk[r])
    on_m = np.nanmean([rk[t][fin[i] & own[t]].mean() if (fin[i] & own[t]).any() else np.nan for i, t in enumerate(r)])
    off_m = np.nanmean([rk[t][fin[i] & ~own[t]].mean() if (fin[i] & ~own[t]).any() else np.nan for i, t in enumerate(r)])
    b = base[p]
    ok = b["ok"]
    exd = (b["von"] - on_m) - (b["voff"] - off_m)
    log(f"  {p:<5} 기본 20일선 아래 {on_m * 100:+5.2f} 위 {off_m * 100:+5.2f} (기본 차 {(on_m - off_m) * 100:+5.2f}) | "
        f"극하방 차 {b['diff'] * 100:+5.2f} → 기본 뺀 차 {np.median(exd[ok]) * 100:+5.2f}(양수 {np.mean(exd[ok] > 0) * 100:3.0f}%) | "
        f"꺼짐 초과 {np.median(b['voff'][ok] - off_m) * 100:+5.2f} 켜짐 초과 {np.median(b['von'][ok] - on_m) * 100:+5.2f}")

# (b) 고친 순열
CT = X.CT
market = [k for k in X.cn_list if CT[k].ndim == 1]
stock = [k for k in X.cn_list if CT[k].ndim == 2]
log(f"\n== (b) 22개 검정 우연 개수(고친 순열): 시장 {len(market)}개 = 주 블록 섞기(같이), 종목별 {len(stock)}개 = 같은 날 유효 종목끼리 섞기(같이)")
vidx = [np.where(V[t])[0] for t in range(T)]


def run(perm_t, colp):
    hits, which = 0, []
    for side in ("위", "아래"):
        s_ = X.SD[side]
        for k in X.cn_list:
            c = CT[k]
            c = c[perm_t] if c.ndim == 1 else np.take_along_axis(c, colp, axis=1)
            on_, off_ = s_.split(c)
            ds = []
            for p in PN:
                von, don, _, _ = s_.val(on_, R10[p])
                voff, doff, _, _ = s_.val(off_, R10[p])
                ok = (don >= 5) & (doff >= 5)
                ds.append(np.median((von - voff)[ok]) if ok.any() else np.nan)
            ds = np.array(ds)
            if np.all(ds > 0) or np.all(ds < 0):
                hits += 1
                which.append(f"{side}·{k}")
    return hits, which


ident = np.arange(T)
NP = 300
hs, wh_m, wh_s = [], 0, 0
own_alone = 0
for bb in range(NP):
    perm = ident.copy()
    for p in PN:
        bl = X.blocks[p]
        order = rng.permutation(len(bl))
        perm[np.concatenate(bl)] = np.concatenate([bl[i] for i in order])
    colp = np.tile(np.arange(N), (T, 1))
    for t in range(T):
        v = vidx[t]
        colp[t, v] = rng.permutation(v)
    h_, w_ = run(perm, colp)
    hs.append(h_)
    own_alone += "아래·자기 종목<20일선" in w_
    wh_m += sum(any(w.endswith(m) for m in market) for w in w_)
    wh_s += sum(any(w.endswith(m) for m in stock) for w in w_)
hs = np.array(hs)
u, c = np.unique(hs, return_counts=True)
log(f"순열 {NP}회: 평균 {hs.mean():.2f}개(시장 {wh_m / NP:.2f}·종목별 {wh_s / NP:.2f}), 1개 이상 {np.mean(hs >= 1) * 100:.0f}%, "
    f"2개 이상 {np.mean(hs >= 2) * 100:.0f}%, 분포 " + ", ".join(f"{int(a)}:{int(b)}" for a, b in zip(u, c))
    + f"; '아래·자기 종목<20일선' 혼자 같은 부호 {own_alone / NP * 100:.1f}% (독립·대칭이면 6.25%)")
log(f"참고: 22개가 서로 독립이면 기대 {22 / 16:.2f}개, 1개 이상 {(1 - (15 / 16) ** 22) * 100:.0f}%")

with open(os.path.join(X.OUT, "verify_x_e3e4_b.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
