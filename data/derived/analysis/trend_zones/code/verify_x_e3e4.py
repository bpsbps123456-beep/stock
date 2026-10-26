"""검증 B: extremes.py의 E3(개별주 폭 vs SOXX)·E4(자기 종목 20일선 맥락) 공격. 기존 파일은 고치지 않는다.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/verify_x_e3e4.py
결과: results/verify_x_e3e4.txt, verify_x_e3e4.json
"""
import json, os, re, sys, time
from datetime import date

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402
import extremes as E  # noqa: E402  (정의 함수·원 맥락만 빌림, 점수·맥락 3개는 따로 계산)

OUT = os.path.join(HERE, "../results")
PER = {"앞": ("2026-01-02", "2026-05-29", True), "뒤": ("2026-06-01", "2026-10-09", False),
       "2025": ("2025-01-02", "2025-12-31", False), "2024": ("2024-01-02", "2024-12-31", False),
       "2023하": ("2023-07-01", "2023-12-29", False)}
PN = list(PER)
rng = np.random.default_rng(20261011)
L = []
J = {}


def log(s=""):
    print(s, flush=True)
    L.append(s)


t0 = time.time()
I = Zm.Ind(*Zm.load())
T, N = len(I.dates), len(I.syms)
V = I.valid.to_numpy()
D = E.definitions(I)
D = [(nm, None if t is None else (t & V), None if b is None else (b & V)) for nm, t, b in D]
WK = np.array([(lambda c: c[0] * 100 + c[1])(date.fromisoformat(d).isocalendar()) for d in I.dates])


def rank_h(h, drop=None):
    f = (I.O.shift(-(1 + h)) / I.O.shift(-1) - 1).where(I.valid)
    if drop:
        f = f.copy()
        f[drop] = np.nan
    return (f.rank(axis=1, pct=True) - 0.5).to_numpy()


def rows_h(p, h):
    a, b, purge = PER[p]
    idx = [i for i, d in enumerate(I.dates) if a <= d <= b and i + 1 + h < T]
    if purge:
        end = max(i for i, d in enumerate(I.dates) if a <= d <= b)
        idx = [i for i in idx if i + 1 + h <= end]
    return np.array(idx)


# ---------------------------------------------------------------- 독립 맥락 3개
def weekly(daily):
    """신호일 i → i+1이 속한 ISO 주의 직전 거래일 종가 상태."""
    first = {}
    for i, w in enumerate(WK):
        first.setdefault(w, i)
    out = np.zeros(T, dtype=bool)
    for i in range(T - 1):
        f0 = first[WK[i + 1]]
        out[i] = bool(daily[f0 - 1]) if f0 > 0 else False
    return out


o = pd.read_parquet(E.SOXX_SRC, columns=["date", "sym", "Close"])
s = o[o.sym == "SOXX"].assign(d=lambda x: pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d"))
s = s.drop_duplicates("d").set_index("d")["Close"].astype(float).sort_index()
soxx_d = (s < s.rolling(20).mean()).reindex(I.dates).ffill().fillna(False).to_numpy(dtype=bool)
C = I.C
sma = C.rolling(20, min_periods=20).mean()
ab = ((C > sma) & I.valid).to_numpy().sum(1) / V.sum(1)
br_d = ab < 0.5
MY = {"SOXX<20일선 주간": weekly(soxx_d), "개별주 폭<50% 주간": weekly(br_d), "자기 종목<20일선": (C < sma).to_numpy()}
CTX, _ = E.contexts(I)
for k, v in MY.items():
    vv = v if v.ndim == 2 else np.repeat(v[:, None], N, 1)
    log(f"맥락 독립 재계산 일치 [{k}]: {np.array_equal(vv & V, CTX[k] & V)}  (불일치 칸 {(vv & V != CTX[k] & V).sum()})")


# ---------------------------------------------------------------- 점수(정의 묶음 벡터화)
class Side:
    def __init__(self, side, names, rk):
        self.sg = -1 if side == "위" else 1
        self.names = names
        fin = np.isfinite(rk)
        r0 = np.where(fin, rk, 0).astype(np.float32)
        idx = {nm: (t if side == "위" else b) for nm, t, b in D}
        self.Mf = np.stack([(idx[nm] & fin) for nm in names]).astype(np.float32)
        self.MR = self.Mf * r0[None]
        self.S, self.n = self.MR.sum(2), self.Mf.sum(2)

    def split(self, c):
        if c.ndim == 1:
            Son, non = self.S * c, self.n * c
        else:
            cf = c.astype(np.float32)
            Son, non = np.einsum("ktn,tn->kt", self.MR, cf), np.einsum("ktn,tn->kt", self.Mf, cf)
        return (Son, non), (self.S - Son, self.n - non)

    def val(self, Sn, rows):
        S_, n_ = Sn[0][:, rows], Sn[1][:, rows]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(n_ > 0, S_ / n_, np.nan)
        v = self.sg * np.nanmean(np.where(np.isfinite(m), m, np.nan), axis=1) if m.shape[1] else np.full(len(S_), np.nan)
        return v, (n_ > 0).sum(1), n_.sum(1), m


def ctx_table(sd, c, rows_map, same_day=False):
    on, off = sd.split(c)
    res = {}
    for p, rows in rows_map.items():
        von, don, non, mon = sd.val(on, rows)
        voff, doff, noff, moff = sd.val(off, rows)
        ok = (don >= 5) & (doff >= 5)
        d = von - voff
        r = {"on": float(np.median(von[ok])) if ok.any() else np.nan, "off": float(np.median(voff[ok])) if ok.any() else np.nan,
             "diff": float(np.median(d[ok])) if ok.any() else np.nan, "pos": float(np.mean(d[ok] > 0)) if ok.any() else np.nan,
             "k": int(ok.sum()), "ok": ok, "d": d, "von": von, "voff": voff, "noff": noff, "doff": doff, "non": non}
        if same_day:
            both = np.isfinite(mon) & np.isfinite(moff)
            with np.errstate(invalid="ignore"):
                sdd = sd.sg * np.array([np.nanmean((mon[k] - moff[k])[both[k]]) if both[k].sum() >= 5 else np.nan for k in range(len(d))])
            r["same_day"] = float(np.nanmedian(sdd)); r["same_day_pos"] = float(np.nanmean(sdd[np.isfinite(sdd)] > 0))
            r["same_day_k"] = int(np.isfinite(sdd).sum())
        res[p] = r
    return res


def fmt(res, extra=False):
    out = []
    for p in PN:
        r = res[p]
        out.append(f"{p} 켜짐 {r['on'] * 100:+5.2f} 꺼짐 {r['off'] * 100:+5.2f} 차 {r['diff'] * 100:+5.2f}({r['pos'] * 100:3.0f}%, k{r['k']})"
                   + (f" 같은날짝 {r['same_day'] * 100:+5.2f}({r['same_day_pos'] * 100:3.0f}%,k{r['same_day_k']})" if extra else ""))
    return " | ".join(out)


rk10 = rank_h(10)
rk_o, _, _ = E.rank_fwd(I, 10)
log(f"순위 수익 재계산 일치(h=10): {np.allclose(np.nan_to_num(rk10, nan=9), np.nan_to_num(rk_o, nan=9))}")
R10 = {p: rows_h(p, 10) for p in PN}
fin = np.isfinite(rk10)
ELIG = {}
for side in ("위", "아래"):
    names = []
    for nm, t, b in D:
        M = t if side == "위" else b
        if M is None:
            continue
        r = R10["앞"]
        sh = (M[r] & fin[r]).sum() / fin[r].sum()
        if 0.01 <= sh <= 0.20:
            names.append(nm)
    ELIG[side] = names
log(f"자격 정의: 위 {len(ELIG['위'])}개, 아래 {len(ELIG['아래'])}개 (주장 115·122)")
SD = {s: Side(s, ELIG[s], rk10) for s in ("위", "아래")}

# ---------------------------------------------------------------- (1) 켜짐/꺼짐 중앙값 재계산
log("\n== (1) 맥락 켜짐/꺼짐 중앙값 독립 재계산 (h=10)")
J["recompute"] = {}
for side in ("위", "아래"):
    for cn, c in MY.items():
        res = ctx_table(SD[side], c, R10, same_day=(cn == "자기 종목<20일선"))
        log(f"[{'극상방' if side == '위' else '극하방'}] {cn}\n    " + fmt(res, extra=(cn == "자기 종목<20일선")))
        J["recompute"][f"{side}·{cn}"] = {p: {k: res[p][k] for k in ("on", "off", "diff", "pos", "k")} for p in PN}

# ---------------------------------------------------------------- (2) E4
log("\n== (2) E4: 극하방 & 자기 종목 20일선 위(꺼짐) 관측이 충분한 정의")
sd = SD["아래"]
own = MY["자기 종목<20일선"]
base = ctx_table(sd, own, R10, same_day=True)


def fam(nm):
    if nm.startswith("RSI"):
        return "RSI" + re.match(r"RSI(\d+)", nm).group(1)
    if nm.startswith("StochRSI"):
        return nm.split(" ")[0]
    if nm.startswith("볼린저"):
        return nm.split(" ")[0]
    if nm.startswith("칼만식"):
        return "칼만식 " + nm.split(" ")[1].split("·")[0]
    if "일 수익 z" in nm:
        return nm.split(" ")[0] + " 수익z"
    if nm.startswith("52주"):
        return "52주 고점"
    return "연속"


def horizon(nm):
    f = fam(nm)
    m = re.search(r"(\d+)", f.replace("52주", "x"))
    short = {"RSI2", "RSI3", "RSI5", "StochRSI(7)", "볼린저10", "칼만식 EMA5", "칼만식 EMA10", "3일 수익z", "5일 수익z", "연속"}
    long_ = {"RSI21", "RSI28", "볼린저50", "칼만식 EMA35", "칼만식 EMA50", "칼만식 EMA100", "20일 수익z", "60일 수익z", "52주 고점"}
    return "짧은(2~10일)" if f in short else "긴(20일+·52주)" if f in long_ else "중간"


okall = np.all([base[p]["ok"] for p in PN], axis=0)
offn = np.array([[base[p]["noff"][k] for p in PN] for k in range(len(sd.names))])
offshare = np.array([[base[p]["noff"][k] / max(base[p]["noff"][k] + base[p]["non"][k], 1) for p in PN] for k in range(len(sd.names))])
log(f"정의 {len(sd.names)}개 중 다섯 구간 모두 꺼짐·켜짐 5일 이상 = {okall.sum()}개; 구간별 통과 수: "
    + ", ".join(f"{p} {base[p]['k']}" for p in PN))
log(f"꺼짐(20일선 위) 비중 중앙: " + ", ".join(f"{p} {np.median(offshare[:, i]) * 100:.1f}%" for i, p in enumerate(PN))
    + " | 꺼짐 관측수 중앙: " + ", ".join(f"{p} {np.median(offn[:, i]):.0f}" for i, p in enumerate(PN)))
big = (offn >= 30).all(1) & okall
log(f"다섯 구간 모두 꺼짐 관측 ≥30 & 5일 이상: {big.sum()}개")
fams = {}
for k, nm in enumerate(sd.names):
    fams.setdefault(fam(nm), []).append(k)
log("계열별: 자격 수 / 다섯 구간 통과 / ≥30관측 통과 / 꺼짐 관측 중앙(앞·뒤·2025·2024·2023하) / 차 중앙(통과 정의)")
J["families"] = {}
for f, ks in sorted(fams.items(), key=lambda x: -len(x[1])):
    ks = np.array(ks)
    dd = [np.median(base[p]["d"][ks][base[p]["ok"][ks]]) * 100 if base[p]["ok"][ks].any() else np.nan for p in PN]
    log(f"  {f:<14} {len(ks):3d} / {okall[ks].sum():3d} / {big[ks].sum():3d} / "
        + "·".join(f"{np.median(offn[ks, i]):.0f}" for i in range(5)) + " / " + " ".join(f"{x:+5.1f}" for x in dd))
    J["families"][f] = {"n": int(len(ks)), "ok5": int(okall[ks].sum()), "ok30": int(big[ks].sum()), "diff": dd}

log("\n-- 짧은 지표 vs 긴 지표: 차(켜짐-꺼짐) 중앙 %p (양수 비율, 정의 수) — 구간별 5일 필터")
J["horizon_groups"] = {}
grp = np.array([horizon(nm) for nm in sd.names])
for g in ("짧은(2~10일)", "중간", "긴(20일+·52주)"):
    ks = grp == g
    parts = []
    gg = {}
    for p in PN:
        m = ks & base[p]["ok"]
        d = base[p]["d"][m]
        parts.append(f"{p} {np.median(d) * 100:+5.2f}({np.mean(d > 0) * 100:3.0f}%,{m.sum()})" if m.any() else f"{p} —")
        gg[p] = float(np.median(d)) if m.any() else None
    J["horizon_groups"][g] = gg
    log(f"  {g:<12} " + " | ".join(parts))
# 다섯 구간 모두 통과한 정의만
m5 = okall
parts = [f"{p} {np.median(base[p]['d'][m5]) * 100:+5.2f}({np.mean(base[p]['d'][m5] > 0) * 100:3.0f}%)" for p in PN]
log(f"  다섯 구간 모두 통과 정의만({m5.sum()}) " + " | ".join(parts))
m30 = big
if m30.any():
    parts = [f"{p} {np.median(base[p]['d'][m30]) * 100:+5.2f}({np.mean(base[p]['d'][m30] > 0) * 100:3.0f}%)" for p in PN]
    log(f"  꺼짐 ≥30관측 정의만({m30.sum()}) " + " | ".join(parts))

# h=5, h=20
log("\n-- 보유 기간 바꿔 보기 (자격 정의 고정, 자기 종목<20일선, 극하방 / 참고로 극상방 SOXX·폭 주간)")
J["h"] = {}
for h in (5, 20):
    rk = rank_h(h)
    Rh = {p: rows_h(p, h) for p in PN}
    for side, cn in (("아래", "자기 종목<20일선"), ("위", "SOXX<20일선 주간"), ("위", "개별주 폭<50% 주간")):
        s2 = Side(side, ELIG[side], rk)
        res = ctx_table(s2, MY[cn], Rh, same_day=(cn == "자기 종목<20일선"))
        log(f"h={h:<2} [{'극상방' if side == '위' else '극하방'}] {cn}\n    " + fmt(res, extra=(cn == "자기 종목<20일선")))
        J["h"][f"{h}·{side}·{cn}"] = {p: {k: res[p][k] for k in ("on", "off", "diff", "pos", "k")} for p in PN}
    del rk

# DRUG 제외
log("\n-- DRUG 제외(순위 다시 매김 + 극단 표시 제거), h=10")
rkd = rank_h(10, drop=["DRUG"])
s3 = Side("아래", ELIG["아래"], rkd)
resd = ctx_table(s3, own, R10, same_day=True)
log("[극하방] 자기 종목<20일선\n    " + fmt(resd, extra=True))
J["noDRUG"] = {p: {k: resd[p][k] for k in ("on", "off", "diff", "pos", "k")} for p in PN}
j_drug = I.syms.index("DRUG")
for p in PN:
    r = R10[p]
    log(f"   DRUG 극하방·20일선 위 관측 {p}: {int(sum((D_[2][r, j_drug] & ~own[r, j_drug]).sum() for D_ in D if D_[0] in ELIG['아래']))} (전 정의 합)")
del s3

# 합친 구간 주 단위 블록 부트스트랩
log("\n-- 합친 구간(2023하~2026, 다섯 구간 행 합집합) 주 단위 블록 부트스트랩 B=2000")
pool = np.unique(np.concatenate([R10[p] for p in PN]))
on, off = sd.split(own)


def mm(Sn):
    S_, n_ = Sn[0][:, pool], Sn[1][:, pool]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n_ > 0, S_ / n_, np.nan)
    return np.nan_to_num(m * sd.sg), (n_ > 0).astype(float)


m_on, h_on = mm(on)
m_off, h_off = mm(off)
okp = (h_on.sum(1) >= 5) & (h_off.sum(1) >= 5)
vo, vf = (m_on.sum(1) / h_on.sum(1)), (m_off.sum(1) / h_off.sum(1))
dpool = (vo - vf)[okp]
wk = WK[pool]
uw, inv = np.unique(wk, return_inverse=True)
nw = len(uw)
B = 2000
cnt = np.zeros((nw, B))
for bb in range(B):
    cnt[:, bb] = np.bincount(rng.integers(0, nw, nw), minlength=nw)
W = cnt[inv]  # dates × B
with np.errstate(invalid="ignore", divide="ignore"):
    bon = (m_on @ W) / (h_on @ W)
    boff = (m_off @ W) / (h_off @ W)
bd = np.nanmedian((bon - boff)[okp], axis=0)
lo, hi = np.percentile(bd, [2.5, 97.5])
log(f"정의 {okp.sum()}개, 주 {nw}개. 합친 켜짐 중앙 {np.median(vo[okp]) * 100:+.2f}, 꺼짐 중앙 {np.median(vf[okp]) * 100:+.2f}, "
    f"차 중앙 {np.median(dpool) * 100:+.2f}%p (음수 정의 {np.mean(dpool < 0) * 100:.0f}%), 95% 구간 [{lo * 100:+.2f}, {hi * 100:+.2f}], "
    f"차≥0 비율 {np.mean(bd >= 0) * 100:.1f}%")
J["pool_boot"] = {"k": int(okp.sum()), "weeks": int(nw), "on": float(np.median(vo[okp])), "off": float(np.median(vf[okp])),
                  "diff": float(np.median(dpool)), "ci": [float(lo), float(hi)], "p_ge0": float(np.mean(bd >= 0))}
# 2023하 제외 (2024~2026) 도 같이
for lab, keep in (("2024~2026", np.isin(pool, np.concatenate([R10[p] for p in ("앞", "뒤", "2025", "2024")]))),):
    Wk = W * keep[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        b2 = np.nanmedian(((m_on @ Wk) / (h_on @ Wk) - (m_off @ Wk) / (h_off @ Wk))[okp], axis=0)
    log(f"  {lab}만: 차 95% 구간 [{np.percentile(b2, 2.5) * 100:+.2f}, {np.percentile(b2, 97.5) * 100:+.2f}], 차≥0 {np.mean(b2 >= 0) * 100:.1f}%")
del W, cnt

# ---------------------------------------------------------------- (2e) 22개 검정의 '다섯 구간 같은 부호' 우연 확률
log("\n-- 22개 맥락 검정의 '다섯 구간 같은 부호' 우연 개수: 구간 안 주 블록 순서를 섞어 11개 맥락에 같이 적용(맥락 간 상관 유지)")
cn_list = list(CTX)
CT = {k: (v[:, 0] if np.all(v == v[:, :1]) else v) for k, v in CTX.items()}


def count_same(perm):
    hits, which = 0, []
    for side in ("위", "아래"):
        s_ = SD[side]
        for k in cn_list:
            c = CT[k][perm]
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
h0, w0 = count_same(ident)
log(f"실제: {h0}개 {w0}")
blocks = {}
for p in PN:
    r = R10[p]
    bl = []
    for i in r:
        if bl and WK[bl[-1][-1]] == WK[i]:
            bl[-1].append(i)
        else:
            bl.append([i])
    blocks[p] = bl
NP = 300
hs, own_hit = [], 0
for bb in range(NP):
    perm = ident.copy()
    for p in PN:
        bl = blocks[p]
        order = rng.permutation(len(bl))
        perm[np.concatenate([bl[i] for i in range(len(bl))])] = np.concatenate([bl[i] for i in order])
    h_, w_ = count_same(perm)
    hs.append(h_)
    own_hit += "아래·자기 종목<20일선" in w_
hs = np.array(hs)
log(f"순열 {NP}회: 평균 {hs.mean():.2f}개, 0개 {np.mean(hs == 0) * 100:.0f}%, 1개 이상 {np.mean(hs >= 1) * 100:.0f}%, 2개 이상 {np.mean(hs >= 2) * 100:.0f}%, "
    f"분포 {dict(zip(*np.unique(hs, return_counts=True)))}; '아래·자기 종목<20일선'만 따로 같은 부호 {own_hit / NP * 100:.1f}% (독립 가정 6.25%)")
J["perm"] = {"real": h0, "which": w0, "mean": float(hs.mean()), "p_ge1": float(np.mean(hs >= 1)), "p_ge2": float(np.mean(hs >= 2)),
             "own_alone": own_hit / NP}

# ---------------------------------------------------------------- (3) E3: 개별주 폭 = SOXX 다시 붙이기인가
log("\n== (3) E3: 개별주 폭<50% 주간 vs SOXX<20일선 주간")
a, b = MY["SOXX<20일선 주간"], MY["개별주 폭<50% 주간"]
J["e3"] = {}
for p in PN + ["전체"]:
    r = pool if p == "전체" else R10[p]
    x, y = a[r], b[r]
    phi = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan
    log(f"  {p:<5} 일치 {np.mean(x == y) * 100:4.0f}%  phi {phi:+.2f}  SOXX켜짐 {x.mean() * 100:3.0f}% 폭켜짐 {y.mean() * 100:3.0f}%  "
        f"P(폭|SOXX) {y[x].mean() * 100 if x.any() else np.nan:3.0f}%  P(SOXX|폭) {x[y].mean() * 100 if y.any() else np.nan:3.0f}%")
    J["e3"][p] = {"agree": float(np.mean(x == y)), "phi": float(phi)}
# 네 칸
log("  극상방 타당도 중앙(%p, 정의별 날짜 5일 이상) — 둘 다 켜짐 / SOXX만 / 폭만 / 둘 다 꺼짐, 괄호 날짜 수")
sdu = SD["위"]
cells = {"둘다": a & b, "SOXX만": a & ~b, "폭만": ~a & b, "둘다꺼짐": ~a & ~b}
J["e3_cells"] = {}
for p in PN:
    parts = []
    for cname, cm in cells.items():
        on_, _ = sdu.split(cm.astype(bool))
        v, d_, _, _ = sdu.val(on_, R10[p])
        ok = d_ >= 5
        nd = int(cm[R10[p]].sum())
        parts.append(f"{cname} {np.median(v[ok]) * 100:+5.2f}({nd})" if ok.any() else f"{cname} —({nd})")
        J["e3_cells"][f"{p}·{cname}"] = float(np.median(v[ok])) if ok.any() else None
    log(f"  {p:<5} " + " | ".join(parts))
# 정의별 차의 상관
cc = []
for p in PN:
    d1, d2 = [ctx_table(sdu, MY[k], {p: R10[p]})[p]["d"] for k in ("SOXX<20일선 주간", "개별주 폭<50% 주간")]
    f_ = np.isfinite(d1) & np.isfinite(d2)
    cc.append(f"{p} {np.corrcoef(d1[f_], d2[f_])[0, 1]:+.2f}(k{f_.sum()})")
log("  정의별 (켜짐-꺼짐) 차의 SOXX·폭 상관(극상방): " + ", ".join(cc))

# E2 보충: 다섯 구간 모두 양수의 기대 개수(구간별 양수 비율 독립 가정)
log("\n== 보충: '다섯 구간 모두 양수' 우연 기대 개수(구간별 실제 양수 비율의 곱 × 정의 수)")
for side in ("위", "아래"):
    s_ = SD[side]
    vs = np.array([s_.val((s_.S, s_.n), R10[p])[0] for p in PN])
    pr = np.nanmean(vs > 0, axis=1)
    allp = np.all(vs > 0, axis=0).sum()
    log(f"  {'극상방' if side == '위' else '극하방'}: 양수 비율 " + "/".join(f"{x * 100:.0f}%" for x in pr)
        + f" → 기대 {np.prod(pr) * len(s_.names):.2f}개 (50% 가정이면 {len(s_.names) / 32:.1f}개), 실제 {allp}개")
log(f"\n소요 {time.time() - t0:.0f}초")

with open(os.path.join(OUT, "verify_x_e3e4.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
with open(os.path.join(OUT, "verify_x_e3e4.json"), "w", encoding="utf-8") as f:
    json.dump(J, f, ensure_ascii=False, indent=1, default=lambda x: None if x is None else float(x) if np.isscalar(x) else str(x))
