"""칼손절 연구 — 결합 모델 + 40분봉 RSI 특징·조건 (g_model, 2026-10-07).

사용자 가설(10/7): "40분봉 RSI가 30 이하(이미 과매도)일 때는 손절하면 안 좋다."
지난 결합 모델(stops/model.py — 원본 수정 없음, ctx·daily·fit_logit·predict·auc·eval_front·early_split만 불러 씀)에
  (1) 40분봉 RSI 특징 5개를 더한 모델(FR)과
  (2) 기존 특징 모델(F)의 결정에 '40분봉 RSI dev[i] > g' 조건을 붙인 규칙(F&g)을 시험한다.

판단 시점(거래마다 한 번만 판단, 판단 봉 i = cp + 1 의 시가에 판다):
  T15·T30·T60 : cp = ei + k/mins (매수 k분 뒤 봉)
  E3          : 매수 뒤 처음으로 c[i-1] <= 매수가 x0.97 이 된 봉 i (cp = i-1, '종가 -3% 첫 도달')
라벨(early 기준 거래, 24시간, 137종목, 판단 봉까지 들고 있던 것; 라벨은 결과값이라 미래를 써도 된다):
  ret3 : 최종 수익 <= -3%     fwd2 : 최종 수익 - (o[cp+1]/매수가 - 1) <= -2% (계속 들면 2% 넘게 더 잃음)
특징(모두 cp 봉 종가까지 + 봉 cp+1 시가에 보이는 40분봉 값):
  F  = 지난 모델 15개(rk lowk rsi_cp rsi_e slope60 dev_cp gap_e pre aft tod prev_ret today_ret vol qqq mkt) + el(cp까지 경과 분)
  FR = F + d40_cp=dev[cp+1], c40_cp=conf[cp+1], d40_e=dev[ei], dd40=dev[cp+1]-dev[ei], dc40=conf[cp+1]-conf[ei]
       (dev, conf) = stop_lab.htf_rsi(A, 40, 14): 인덱스 i 값은 봉 i 시가에 쓸 수 있는 값(dev = 진행 중 40분봉 RSI, c[i-1] 반영)
문턱 p: 앞 2/3 모델의 학습 표본 예측값 상위 25%·12%·5% 분위(같은 p를 early 전체 재학습 모델에도 그대로 씀).
결정 팔: F(p) / FR(p) / F(p)&dev40>30 / F(p)&dev40>40 / FR(p)&dev40>30(중간 p만).
선택: 앞 2/3 날짜로 학습한 모델을 early 전체에 돌려 뒤 1/3 날짜 사용자78 24시간 차이 최대(정규장만 >= 0 우선).
      고른 설정을 early 전체로 다시 학습해 compare()로 late 전방 확인. late는 고르는 데 안 쓴다.
사용: py stops/g_model.py         (전체, 결과 ../results/stops/g_model.txt)
      py stops/g_model.py check   (미래 참조 검사만)
"""
import os, sys, time, io
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from stop_lab import load, run, compare, base_trades, metrics, stop_audit, htf_rsi, UNIV, GAP, RAW  # noqa: E402
import model as M  # noqa: E402  (지난 결합 모델, 읽기만)

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "g_model.txt"))
PER = {5: "early", 3: "late"}
FCOLS = list(M.FULL) + ["el"]
RCOLS = ["d40_cp", "c40_cp", "d40_e", "dd40", "dc40"]
ALLC = FCOLS + RCOLS
CI = {"F": list(range(len(FCOLS))), "FR": list(range(len(ALLC)))}
EV_X = 0.03
QS = (0.25, 0.12, 0.05)
SLOTS = [(("T", 15), "ret3"), (("T", 30), "ret3"), (("T", 60), "ret3"),
         (("T", 15), "fwd2"), (("T", 30), "fwd2"), (("T", 60), "fwd2"), (("E", 3), "fwd2")]

_LOG = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.write(s + "\n")


def dump(tail=""):
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(_LOG.getvalue() + tail)


def sname(slot, label):
    return f"{slot[0]}{slot[1]}-{label}"


# ---------------------------------------------------------------- 특징(판단 시점 하나)
_RI, _FC = {}, {}


def asof1(t_arr, v_arr, t):
    j = np.searchsorted(t_arr, t, side="right") - 1
    return v_arr[j] if j >= 0 else np.nan


def feat_at(A, e, cp, hr=None, cache=True):
    """매수 봉 e, 체크포인트 cp(판단 봉 cp+1)의 특징 21개. A[...][:cp+1] + dev/conf[cp+1], dev/conf[e] 만 쓴다."""
    key = (A["sym"], A["mins"], e, cp)
    if cache and key in _FC:
        return _FC[key]
    C = M.ctx(PER[A["mins"]])
    o, l, c, v, r, ema, tk = A["o"], A["l"], A["c"], A["v"], A["r"], A["ema"], A["tk"]
    mins = A["mins"]
    ent = min(o[e], ema[e - 1] * (1 - GAP))
    f = np.full(len(ALLC), np.nan)
    f[0] = c[cp] / ent - 1
    f[1] = l[e:cp + 1].min() / ent - 1
    f[2] = r[cp]
    f[3] = r[e - 1]
    f[4] = ema[cp] / ema[max(cp - 60 // mins, 0)] - 1
    f[5] = c[cp] / ema[cp] - 1
    f[6] = ent / ema[e - 1] - 1
    f[7] = float(A["ses"][e] == "프리")
    f[8] = float(A["ses"][e] == "애프터")
    f[9] = A["mod"][e] / 60.0
    dk, dc = M.daily(A["sym"])
    d = A["day"][e]
    j = int(np.searchsorted(dk, d.year * 10000 + d.month * 100 + d.day, side="left")) - 1
    c1 = dc[j] if j >= 0 else np.nan
    c2 = dc[j - 1] if j >= 1 else np.nan
    f[10] = np.clip(c1 / c2 - 1, -0.3, 0.3)
    f[11] = np.clip(ent / c1 - 1, -0.3, 0.3)
    rk = (A["sym"], mins, len(o))
    if rk not in _RI:
        _RI[rk] = np.flatnonzero(A["reg"])
    ri = _RI[rk]
    win = v[e:cp + 1].mean()
    pos = int(np.searchsorted(ri, e, side="left"))
    lo = max(pos - 390 // mins, 0)
    base = v[ri[lo:pos]].mean() if pos > lo else np.nan
    f[12] = np.clip(np.log(win / base), -3, 3) if (A["reg"][e] and base > 0 and win > 0) else 0.0
    q0 = asof1(C["q_tk"], C["q_c"], tk[e - 1])
    q1 = asof1(C["q_tk"], C["q_c"], tk[cp])
    f[13] = np.clip(q1 / q0 - 1, -0.1, 0.1)
    f[14] = asof1(C["m_tk"], C["m_v"], tk[cp])
    f[15] = (cp - e) * mins
    dev, conf = hr if hr is not None else htf_rsi(A)
    f[16] = dev[cp + 1]
    f[17] = conf[cp + 1]
    f[18] = dev[e]
    f[19] = dev[cp + 1] - dev[e]
    f[20] = conf[cp + 1] - conf[e]
    f[~np.isfinite(f)] = np.nan
    if cache:
        _FC[key] = f
    return f


def checkpoint(A, ei, xi, ent, slot):
    """기준 거래(ei 매수, xi 매도)에서 판단 직전 봉 cp. 판단 전에 이미 팔렸으면 None."""
    kind, k = slot
    if kind == "T":
        cp = ei + k // A["mins"]
        return cp if xi > cp else None          # model.py train_table과 같은 조건
    seg = A["c"][ei:xi - 1]                      # i = ei+1 .. xi-1 의 c[i-1]
    hit = np.flatnonzero(seg <= ent * (1 - EV_X))
    return ei + int(hit[0]) if len(hit) else None


# ---------------------------------------------------------------- 학습 표
def table(slot, label):
    S = load("early")
    b = base_trades("early", False)
    X, y, d, sy = [], [], [], []
    for x in b.itertuples():
        A = S[x.sym]
        cp = checkpoint(A, x.ei, x.xi, x.ent, slot)
        if cp is None:
            continue
        f = feat_at(A, x.ei, cp)
        if np.isnan(f[0]):
            continue
        fwd = x.ret - (A["o"][cp + 1] / x.ent - 1)
        X.append(f)
        y.append(x.ret <= -0.03 if label == "ret3" else fwd <= -0.02)
        d.append(x.day)
        sy.append(x.sym)
    return np.array(X), np.array(y, dtype=float), np.array(d, dtype=object), np.array(sy, dtype=object)


# ---------------------------------------------------------------- 규칙
def factory(slot, model, ci, p, gate=None):
    """gate: (src 'dev'|'conf', op '>'|'<=', g). 값이 NaN이면 어느 조건에서도 팔지 않는다."""
    kind, k = slot

    def make_rule(A):
        hr = htf_rsi(A)
        dev, conf = hr
        c = A["c"]
        kb = k // A["mins"] if kind == "T" else None
        done = set()

        def rule(i, ei, ent):
            if kind == "T":
                if i - ei != kb + 1:
                    return False
            else:
                if ei in done or c[i - 1] > ent * (1 - EV_X):
                    return False
                done.add(ei)
            if gate is not None:
                src, op, g = gate
                gv = dev[i] if src == "dev" else conf[i]
                if not np.isfinite(gv):
                    return False
                if (op == ">" and not gv > g) or (op == "<=" and not gv <= g):
                    return False
            if model is None:                     # 보충: 모델 없이 판단 시점 + 조건만
                return True
            f = feat_at(A, ei, i - 1, hr=hr)
            return bool(M.predict(model, f[ci][None, :])[0] > p)
        return rule
    return make_rule


def gname(gate):
    if gate is None:
        return ""
    src, op, g = gate
    return f" & {'dev40' if src == 'dev' else 'conf40'}{op}{g:g}"


def cells(cm):
    out = {}
    for per in ("early", "late"):
        for rng in ("24시간", "정규장만"):
            for un in ("사용자78", "전체137"):
                x = cm[(cm.기간 == per) & (cm.범위 == rng) & (cm.종목 == un)].iloc[0]
                tag = f"{per}_{'24' if rng == '24시간' else 'reg'}_{'u' if un == '사용자78' else 'a'}"
                for kk, src in (("", "차이"), ("_t", "t"), ("_stop", "손절"), ("_worst", "최악"), ("_oct6", "차이_10월6일")):
                    out[tag + kk] = x[src]
    return out


def fmt_audit(a):
    return ", ".join(f"{k} {v:+.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items())


# ---------------------------------------------------------------- 미래 참조 검사
def htf_from(A, m5, tf=40, n=14):
    """stop_lab.htf_rsi와 같은 계산을 주어진 5분봉(m5)으로(검사용)."""
    from htf import intraday_bars, rma_state, _rsi_from
    hs, hc = intraday_bars(m5, tf)
    hc = hc.astype(float)
    g, l, r = rma_state(hc, n)
    hs_ns = hs.as_unit("ns").asi8
    step = tf * 60 * 10 ** 9
    tk = A["tk"]
    prev_t = np.r_[tk[0], tk[:-1]]
    idx = A["idx"]
    anchor = (idx.normalize() + pd.Timedelta(hours=4)).as_unit("ns").asi8
    anchor_prev = np.r_[anchor[0], anchor[:-1]]
    cur_prev = anchor_prev + ((prev_t - anchor_prev) // step) * step
    j = np.searchsorted(hs_ns, cur_prev, side="left") - 1
    ok, jj = j >= 0, np.maximum(j, 0)
    c_prev = np.r_[np.nan, A["c"][:-1]]
    dev = np.where(ok, _rsi_from(g[jj], l[jj], hc[jj], c_prev, n), np.nan)
    cur_i = anchor + ((tk - anchor) // step) * step
    jc = np.searchsorted(hs_ns, cur_i, side="left") - 1
    conf = np.where(jc >= 0, r[np.maximum(jc, 0)], np.nan)
    dev[0] = conf[0] = np.nan
    return dev, conf


def lookahead_check(nsym=10, reps=3, seed=1):
    """봉 j 이후 o·h·l·c·v·ema·r 무작위 교란 + 5분봉을 '끝 시각 <= tk[j]'로 잘라도, 판단 봉 cp+1 <= j 인 특징(21개)이 그대로인지.
    또 htf_from(전체 5분봉) == stop_lab.htf_rsi 인지, 기존 model.feats(T 체크포인트)와 feat_at 15개 특징이 같은지 본다."""
    rng = np.random.default_rng(seed)
    bad = tot = same_bad = same_tot = feat_bad = feat_tot = 0
    for per in ("early", "late"):
        S = load(per)
        syms = list(S)
        pick = [syms[int(x)] for x in rng.choice(len(syms), nsym, replace=False)]
        for s in pick:
            A = S[s]
            n = len(A["o"])
            m5 = pd.read_parquet(os.path.join(RAW, f"5m_{s}.parquet"))
            d0, c0 = htf_rsi(A)
            d1, c1 = htf_from(A, m5)
            same_tot += 1
            same_bad += not (np.allclose(d0, d1, equal_nan=True) and np.allclose(c0, c1, equal_nan=True))
            for kb_min in (15, 30, 60):        # model.feats와 일치
                kb = kb_min // A["mins"]
                XF = M.feats(A, kb)
                es = rng.integers(70, n - kb - 2, 40)
                for e in es:
                    f = feat_at(A, int(e), int(e) + kb, cache=False)
                    feat_tot += 1
                    a, b2 = np.nan_to_num(f[:15], nan=-999), np.nan_to_num(XF[e], nan=-999)
                    feat_bad += not np.allclose(a, b2, atol=1e-9)
            for _ in range(reps):
                j = int(rng.integers(n // 3, n - 50))
                B = dict(A)
                for key in ("o", "h", "l", "c", "v", "ema", "r"):
                    arr = A[key].copy()
                    arr[j:] = arr[j:] * rng.uniform(0.5, 1.5, n - j)
                    B[key] = arr
                T = A["tk"][j]
                m5t = m5[(m5.index.as_unit("ns").asi8 + 5 * 60 * 10 ** 9) <= T]
                hrB = htf_from(B, m5t)
                hrA = (d0, c0)
                for _k in range(60):
                    e = int(rng.integers(max(70, j - 400), j - 1))
                    cp = int(rng.integers(e, j))          # 판단 봉 cp+1 <= j
                    fa = feat_at(A, e, cp, hr=hrA, cache=False)
                    fb = feat_at(B, e, cp, hr=hrB, cache=False)
                    tot += 1
                    bad += not np.allclose(np.nan_to_num(fa, nan=-999), np.nan_to_num(fb, nan=-999), atol=1e-12)
    say(f"미래 참조 검사: (1) htf_from(전체 5분봉) == stop_lab.htf_rsi {same_tot}종목 중 불일치 {same_bad} / "
        f"(2) feat_at 앞 15개 == model.feats {feat_tot}건 중 불일치 {feat_bad} / "
        f"(3) 봉 j 이후 가격 교란 + 5분봉 tk[j] 이후 삭제 → 판단 봉 <= j 특징 21개 불변: {tot}건 중 위반 {bad}")
    return bad + same_bad


# ---------------------------------------------------------------- 실행
def main():
    t0 = time.time()
    for p_ in ("early", "late"):
        S = load(p_)
        base_trades(p_, False)
        base_trades(p_, True)
        M.ctx(p_)
        for A in S.values():
            htf_rsi(A)
    tr_days, va_days = M.early_split()
    ntr, nva = len(tr_days), len(va_days)
    say(f"[g_model] early 날짜 {ntr + nva}일: 학습 {tr_days[0]}~{tr_days[-1]} ({ntr}일), 검증 {va_days[0]}~{va_days[-1]} ({nva}일)")
    say(f"특징 F {FCOLS}\n특징 R40 {RCOLS}  (L2 λ={M.LAM}, 문턱 = 앞 모델 학습 예측 상위 {[f'{q:.0%}' for q in QS]})")
    lookahead_check()
    dump("\n(진행 중)\n")

    say("\n=== 1. 분류 성능: 40분봉 RSI 특징을 넣으면 '잃을 거래'를 더 잘 가려내나 (early 137종목) ===")
    models = {}
    for slot, label in SLOTS:
        nm = sname(slot, label)
        X, y, d, sy = table(slot, label)
        tr = np.isin(d, tr_days)
        va = ~tr
        mm = dict(X=X, y=y, d=d, tr=tr)
        for arm in ("F", "FR"):
            ci = CI[arm]
            fm = M.fit_logit(X[tr][:, ci], y[tr])
            gm = M.fit_logit(X[:, ci], y)
            ptr, pva = M.predict(fm, X[tr][:, ci]), M.predict(fm, X[va][:, ci])
            ps = [float(np.quantile(ptr, 1 - q)) for q in QS]
            mm[arm] = dict(front=fm, full=gm, ps=ps, auc_tr=M.auc(ptr, y[tr]), auc_va=M.auc(pva, y[va]),
                           auc_full=M.auc(M.predict(gm, X[:, ci]), y))
            say(f"{nm:>11} {arm:<2}: 학습 {tr.sum()}건 양성 {y[tr].mean():.1%} AUC {mm[arm]['auc_tr']:.3f} | 검증 {va.sum()}건 양성 "
                f"{y[va].mean():.1%} AUC {mm[arm]['auc_va']:.3f} | 전체 재학습 AUC {mm[arm]['auc_full']:.3f} | p "
                + " ".join(f"{q:.0%}:{p_:.3f}(검증 표시 {(pva > p_).mean():.1%})" for q, p_ in zip(QS, ps)))
        w = mm["FR"]["full"]["w"]
        say(f"{'':>11} FR 전체학습 계수(표준화) " + " ".join(f"{cn}={wv:+.2f}" for cn, wv in zip(["절편"] + ALLC, w)
                                                       if cn in ["절편", "rk", "lowk", "rsi_cp", "dev_cp", "pre", "tod"] + RCOLS))
        # 40분봉 RSI 단면: 판단 시점 dev40 구간별 라벨 양성률과 '판단 봉에서 팔기 - 끝까지' 평균(early 137, 기준 거래)
        d40 = X[:, ALLC.index("d40_cp")]
        say(f"{'':>11} 판단 시점 dev40 구간별 양성률: " + " ".join(
            f"{lo_}~{hi_}:{(y[(d40 > lo_) & (d40 <= hi_)]).mean():.1%}({int(((d40 > lo_) & (d40 <= hi_)).sum())})"
            for lo_, hi_ in ((0, 30), (30, 40), (40, 50), (50, 100))))
        models[nm] = mm
    dump("\n(진행 중)\n")

    say("\n=== 2. 규칙 격자 (va=앞 2/3 모델의 early 뒤 1/3 검증, tr=같은 모델 앞 2/3 학습 날짜, early_/late_=early 전체 재학습 모델 compare) ===")
    rows = []
    nrule = 0
    for slot, label in SLOTS:
        nm = sname(slot, label)
        mm = models[nm]
        for qi, q in enumerate(QS):
            arms = [("F", None), ("FR", None), ("F", ("dev", ">", 30)), ("F", ("dev", ">", 40))]
            if qi == 1:
                arms.append(("FR", ("dev", ">", 30)))
            for arm, gate in arms:
                nrule += 1
                p_ = mm[arm]["ps"][qi]
                lab = f"{nm} {arm} top{q:.0%}(p>{p_:.3f}){gname(gate)}"
                fr = M.eval_front(factory(slot, mm[arm]["front"], CI[arm], p_, gate), tr_days, va_days)
                cm = compare(factory(slot, mm[arm]["full"], CI[arm], p_, gate), lab, show=False)
                row = dict(규칙=lab, slot=nm, arm=arm, q=q, p=p_, gate=gname(gate).strip(" &") or "없음",
                           va24=fr[("va", "24h", "사용자78")]["차이"], vareg=fr[("va", "reg", "사용자78")]["차이"],
                           va24_t=fr[("va", "24h", "사용자78")]["t"], va_stop=fr[("va", "24h", "사용자78")]["손절"],
                           tr24=fr[("tr", "24h", "사용자78")]["차이"], trreg=fr[("tr", "reg", "사용자78")]["차이"],
                           va24_a=fr[("va", "24h", "전체137")]["차이"], **cells(cm))
                rows.append(row)
                say(f"[{nrule:>3}] {lab:<52} va 24h {row['va24']:+.2f} 정규 {row['vareg']:+.2f} 손절 {row['va_stop']:>3} | "
                    f"tr 24h {row['tr24']:+.2f} 정규 {row['trreg']:+.2f} | 전체모델 early 24h {row['early_24_u']:+.2f} "
                    f"정규 {row['early_reg_u']:+.2f} · late 24h {row['late_24_u']:+.2f} 정규 {row['late_reg_u']:+.2f} "
                    f"(손절 e{row['early_24_u_stop']}/l{row['late_24_u_stop']})  ({time.time() - t0:.0f}s)")
                dump("\n(진행 중)\n")
    R = pd.DataFrame(rows)
    R.to_csv(OUT.replace(".txt", "_grid.csv"), index=False, encoding="utf-8-sig")

    say(f"\n시험한 규칙 수(격자): {nrule}")
    four = (R.early_24_u > 0) & (R.early_reg_u > 0) & (R.late_24_u > 0) & (R.late_reg_u > 0)
    four_v = (R.va24 > 0) & (R.vareg > 0) & (R.late_24_u > 0) & (R.late_reg_u > 0)
    say(f"사용자78 네 칸(전체모델 early 24h·정규, late 24h·정규) 모두 플러스: {int(four.sum())}개 {list(R[four].규칙)}")
    say(f"  (early 칸을 앞 모델 뒤 1/3 검증으로 바꾼 네 칸): {int(four_v.sum())}개 {list(R[four_v].규칙)}")
    R["tr_day"], R["va_day"] = R.tr24 / ntr, R.va24 / nva
    say(f"과적합(사용자78 24h 하루 평균 차이, 격자 평균): 학습 날짜 {R.tr_day.mean():+.4f} vs 검증 날짜 {R.va_day.mean():+.4f}; "
        f"규칙별 학습-검증 상관 {R.tr_day.corr(R.va_day):+.2f}; 전체모델 early(학습 안) 평균 {R.early_24_u.mean():+.3f} vs late(전방) 평균 "
        f"{R.late_24_u.mean():+.3f}; 규칙별 va24-late24 상관 {R.va24.corr(R.late_24_u):+.2f}, early24-late24 상관 {R.early_24_u.corr(R.late_24_u):+.2f}")

    # 조건 효과: 같은 슬롯·같은 p에서 F(조건 없음) 대비
    say("\n=== 3. 40분봉 RSI 조건 효과 (같은 슬롯·같은 p의 F 무조건 규칙 대비 차이, 사용자78; 21쌍) ===")
    keyc = ["slot", "q"]
    Fb = R[(R.arm == "F") & (R.gate == "없음")].set_index(keyc)
    for arm, gate in (("FR", "없음"), ("F", "dev40>30"), ("F", "dev40>40"), ("FR", "dev40>30")):
        G = R[(R.arm == arm) & (R.gate == gate)].set_index(keyc)
        J = G.join(Fb, rsuffix="_F", how="inner")
        msg = []
        for col in ("va24", "vareg", "early_24_u", "early_reg_u", "late_24_u", "late_reg_u", "early_24_a", "late_24_a"):
            dd = J[col] - J[col + "_F"]
            msg.append(f"{col} {dd.mean():+.3f}(+{int((dd > 1e-9).sum())}/-{int((dd < -1e-9).sum())})")
        say(f"{arm}{'' if gate == '없음' else ' & ' + gate} vs F ({len(J)}쌍): 평균 차이 " + " ".join(msg))
    say("팔별 평균(사용자78): " + " | ".join(
        f"{a}{'' if g == '없음' else '&' + g}: va24 {x.va24.mean():+.3f} early24 {x.early_24_u.mean():+.3f} late24 {x.late_24_u.mean():+.3f} "
        f"late정규 {x.late_reg_u.mean():+.3f} 손절 e{x.early_24_u_stop.mean():.0f}/l{x.late_24_u_stop.mean():.0f}"
        for (a, g), x in R.groupby(["arm", "gate"])))

    # 선택(early 뒤 1/3만)
    cand = R[R.vareg >= 0]
    pick = (cand if len(cand) else R).sort_values("va24", ascending=False)
    best = pick.iloc[0]
    say(f"\n=== 4. early(뒤 1/3 검증)로 고른 최선안: {best.규칙} (va 24h {best.va24:+.2f}, 정규 {best.vareg:+.2f}, t {best.va24_t:+.2f}) ===")
    tab = R.sort_values("va24", ascending=False)[["규칙", "va24", "vareg", "va24_t", "va_stop", "tr24", "trreg", "early_24_u",
                                                    "early_reg_u", "late_24_u", "late_reg_u", "early_24_a", "late_24_a"]]
    say(tab.head(15).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    slot = next(s for s, lb in SLOTS if sname(s, lb) == best.slot)
    mm = models[best.slot]
    qi = QS.index(best.q)
    gate = None if best.gate == "없음" else ("dev", ">", float(best.gate.split(">")[1]))
    mk = factory(slot, mm[best.arm]["full"], CI[best.arm], best.p, gate)
    say("\n최선안 전체 표(early 전체 재학습 모델; early 칸은 학습 안쪽, late 칸은 전방 확인):")
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    compare(mk, best.규칙, show=True)
    sys.stdout = old
    say(buf.getvalue())
    for per in ("early", "late"):
        for reg in (False, True):
            say(f"stop_audit {per} {'정규장만' if reg else '24시간'} 사용자78: {fmt_audit(stop_audit(mk, per, reg_only=reg))}")
    # 앞 모델 기준 학습/검증 하루 평균
    say(f"과적합(최선안): 앞 모델 학습 날짜 하루 평균 {best.tr24 / ntr:+.4f} vs 검증 날짜 {best.va24 / nva:+.4f}; "
        f"전체모델 early 하루 평균 {best.early_24_u / (ntr + nva):+.4f} vs late 하루 평균 {best.late_24_u / 19:+.4f}(late 19거래일 근사)")

    # 조건 분리: 같은 모델·같은 p, 조건 없음 / 반대 조건 / conf 조건
    say("\n=== 5. 조건 분리(최선안과 같은 모델·같은 p) ===")
    extra = []
    if best.arm == "F" and gate is not None:
        g = gate[2]
        variants = [("조건 없음(같은 손절)", None), (f"반대 조건 dev40<={g:g}", ("dev", "<=", g)),
                    (f"conf40>{g:g}", ("conf", ">", g))]
    elif best.arm == "FR" and gate is not None:
        g = gate[2]
        variants = [("조건 없음(같은 FR 손절)", None), (f"반대 조건 dev40<={g:g}", ("dev", "<=", g))]
    elif best.arm == "F":
        variants = [("dev40>30 붙임", ("dev", ">", 30)), ("반대 조건 dev40<=30", ("dev", "<=", 30))]
    else:  # FR 무조건
        variants = [("반대 조건 dev40<=30", ("dev", "<=", 30)), ("dev40>30 붙임", ("dev", ">", 30))]
    for vn, vg in variants:
        hit = R[(R.slot == best.slot) & (R.arm == best.arm) & (R.q == best.q) & (R.gate == (gname(vg).strip(" &") or "없음"))]
        if len(hit):
            x = hit.iloc[0]
            say(f"{vn}: 격자에 있음 → {x.규칙}")
            extra.append((vn, dict(x)))
            continue
        nrule += 1
        lab = f"{best.slot} {best.arm} top{best.q:.0%}(p>{best.p:.3f}){gname(vg)}"
        fr = M.eval_front(factory(slot, mm[best.arm]["front"], CI[best.arm], best.p, vg), tr_days, va_days)
        cm = compare(factory(slot, mm[best.arm]["full"], CI[best.arm], best.p, vg), lab, show=False)
        row = dict(규칙=lab, va24=fr[("va", "24h", "사용자78")]["차이"], vareg=fr[("va", "reg", "사용자78")]["차이"], **cells(cm))
        extra.append((vn, row))
        say(f"[{nrule:>3}] {vn}: {lab}")
    if best.arm == "FR" and gate is None:      # FR 무조건이면 '조건 없음' 대조 = 같은 슬롯·같은 분위 F
        x = R[(R.slot == best.slot) & (R.arm == "F") & (R.q == best.q) & (R.gate == "없음")].iloc[0]
        extra.insert(0, ("40분봉 특징 뺀 F(같은 분위)", dict(x)))
    say(f"{'':<24} {'va24':>6} {'va정규':>6} | {'e24':>6} {'e정규':>6} {'l24':>6} {'l정규':>6} | {'e24전체':>7} {'l24전체':>7} | 손절 e/l")
    say(f"{'최선안':<24} {best.va24:+6.2f} {best.vareg:+6.2f} | {best.early_24_u:+6.2f} {best.early_reg_u:+6.2f} {best.late_24_u:+6.2f} "
        f"{best.late_reg_u:+6.2f} | {best.early_24_a:+7.2f} {best.late_24_a:+7.2f} | {best.early_24_u_stop}/{best.late_24_u_stop}")
    for vn, x in extra:
        say(f"{vn:<24} {x['va24']:+6.2f} {x['vareg']:+6.2f} | {x['early_24_u']:+6.2f} {x['early_reg_u']:+6.2f} {x['late_24_u']:+6.2f} "
            f"{x['late_reg_u']:+6.2f} | {x['early_24_a']:+7.2f} {x['late_24_a']:+7.2f} | {x['early_24_u_stop']}/{x['late_24_u_stop']}")

    say(f"\n시험한 규칙 수(격자 + 조건 분리 추가): {nrule}")
    say("\n=== 6. 차순위(va24 순, 최선안 제외 상위 4) ===")
    for _, x in tab.iloc[1:5].iterrows():
        y_ = R[R.규칙 == x.규칙].iloc[0]
        say(f"{x.규칙}: va24 {x.va24:+.2f} 정규 {x.vareg:+.2f} | early 24h {y_.early_24_u:+.2f} 정규 {y_.early_reg_u:+.2f} · "
            f"late 24h {y_.late_24_u:+.2f} 정규 {y_.late_reg_u:+.2f} | 전체137 e24 {y_.early_24_a:+.2f} l24 {y_.late_24_a:+.2f}")

    say("\n=== 7. 전체 격자 ===")
    show = ["규칙", "va24", "vareg", "va_stop", "tr24", "trreg", "early_24_u", "early_reg_u", "late_24_u", "late_reg_u",
            "early_24_a", "early_reg_a", "late_24_a", "late_reg_a", "early_24_u_stop", "late_24_u_stop"]
    say(R[show].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    dump()
    say(f"\n저장: {OUT}  ({time.time() - t0:.0f}s)")
    dump()


def supp():
    """보충(격자 결과를 본 뒤 추가, 선택에는 안 씀): E3(종가 -3% 첫 도달) 판단 시점의 조건 분리와 최선안 손절의 세션 구성.
    결과는 g_model.txt 끝에 덧붙인다."""
    global _LOG
    with open(OUT, encoding="utf-8") as f:
        prev = f.read()
    _LOG = io.StringIO()
    t0 = time.time()
    for p_ in ("early", "late"):
        S = load(p_)
        base_trades(p_, False)
        base_trades(p_, True)
        M.ctx(p_)
        for A in S.values():
            htf_rsi(A)
    tr_days, va_days = M.early_split()

    def build(slot, label, arm):
        X, y, d, _ = table(slot, label)
        tr = np.isin(d, tr_days)
        ci = CI[arm]
        fm, gm = M.fit_logit(X[tr][:, ci], y[tr]), M.fit_logit(X[:, ci], y)
        return fm, gm, [float(np.quantile(M.predict(fm, X[tr][:, ci]), 1 - q)) for q in QS]

    E = ("E", 3)
    eF = build(E, "fwd2", "F")
    say("\n=== 8. 보충(격자 결과를 본 뒤 추가 — 선택에 쓰지 않음, 규칙 6개 추가 → 합계 98) ===")
    say("E3 = 매수 뒤 처음으로 i-1 종가 <= 매수가 x0.97 인 봉 i의 시가에 판단(거래당 한 번). 모델 없음 = 그 순간 무조건 판다.")
    sup = [("E3 무모델(첫 -3% 즉시 매도)", None, None, None),
           ("E3 무모델 & dev40>30", None, None, ("dev", ">", 30)),
           ("E3 무모델 & dev40<=30 (반대)", None, None, ("dev", "<=", 30)),
           ("E3 무모델 & dev40>40", None, None, ("dev", ">", 40)),
           ("E3 F top25% & dev40<=30 (반대)", eF, 0, ("dev", "<=", 30)),
           ("E3 F top12% & dev40<=30 (반대)", eF, 1, ("dev", "<=", 30))]
    rows = []
    for k_, (nm, mod, qi, gate) in enumerate(sup, 93):
        fm = gm = None
        p_ = 0.0
        if mod is not None:
            fm, gm, p_ = mod[0], mod[1], mod[2][qi]
        fr = M.eval_front(factory(E, fm, CI["F"], p_, gate), tr_days, va_days)
        cm = compare(factory(E, gm, CI["F"], p_, gate), nm, show=False)
        row = dict(규칙=nm, va24=fr[("va", "24h", "사용자78")]["차이"], vareg=fr[("va", "reg", "사용자78")]["차이"], **cells(cm))
        rows.append(row)
        say(f"[{k_:>3}] {nm} ({time.time() - t0:.0f}s)")
    R = pd.read_csv(OUT.replace(".txt", "_grid.csv"))
    for lab in ("E3-fwd2 F top25%(p>0.269)", "E3-fwd2 F top25%(p>0.269) & dev40>30", "E3-fwd2 F top12%(p>0.354)",
                "E3-fwd2 F top12%(p>0.354) & dev40>30"):
        x = R[R.규칙 == lab]
        if len(x):
            rows.append(dict(x.iloc[0]))
    say(f"{'':<34} {'va24':>6} {'va정규':>6} | {'e24':>6}({'t':>5}) {'e정규':>6} {'l24':>6}({'t':>5}) {'l정규':>6} | {'e24전체':>7} {'l24전체':>7} | 손절 e/l | late 최악")
    for x in rows:
        say(f"{x['규칙']:<34} {x['va24']:+6.2f} {x['vareg']:+6.2f} | {x['early_24_u']:+6.2f}({x['early_24_u_t']:+5.2f}) {x['early_reg_u']:+6.2f} "
            f"{x['late_24_u']:+6.2f}({x['late_24_u_t']:+5.2f}) {x['late_reg_u']:+6.2f} | {x['early_24_a']:+7.2f} {x['late_24_a']:+7.2f} | "
            f"{x['early_24_u_stop']}/{x['late_24_u_stop']} | {x['late_24_u_worst']:+.1%}")

    # stop_audit + late 날짜 집중도
    aud = [("E3 무모델 & dev40>30", factory(E, None, CI["F"], 0.0, ("dev", ">", 30))),
           ("E3 F top25% & dev40>30", factory(E, eF[1], CI["F"], eF[2][0], ("dev", ">", 30))),
           ("E3 F top12% & dev40>30", factory(E, eF[1], CI["F"], eF[2][1], ("dev", ">", 30)))]
    for nm, mk in aud:
        for per in ("early", "late"):
            say(f"stop_audit {nm} {per} 24시간 사용자78: {fmt_audit(stop_audit(mk, per))}")
            t = run(load(per), mk, per, False, syms=UNIV["사용자78"])
            b = base_trades(per, False)
            b = b[b.sym.isin(UNIV["사용자78"])]
            dd = (t.groupby("day").ret.sum() - b.groupby("day").ret.sum()).fillna(0).sort_values()
            say(f"   {per} 날짜별 차이: 합 {dd.sum():+.3f}, 나은 날 {int((dd > 1e-9).sum())} 나쁜 날 {int((dd < -1e-9).sum())}, "
                f"최고 {dd.index[-1]} {dd.iloc[-1]:+.3f} (빼면 {dd.sum() - dd.iloc[-1]:+.3f}), 최저 {dd.index[0]} {dd.iloc[0]:+.3f}; "
                f"손절 세션 {t[t.why == '손절'].ses.value_counts().to_dict()}")

    # 최선안 손절 세션 구성
    tF = build(("T", 60), "fwd2", "FR")
    mk = factory(("T", 60), tF[1], CI["FR"], tF[2][1], None)
    for per in ("early", "late"):
        t = run(load(per), mk, per, False, syms=UNIV["사용자78"])
        st = t[t.why == "손절"]
        say(f"최선안 T60-fwd2 FR top12% {per} 24시간 손절 {len(st)}건 매수 세션 {st.ses.value_counts().to_dict()} 손절 평균 {st.ret.mean():+.2%}")
    # late 최악 거래(-16.3%)가 E3 조건에서 어떻게 됐나
    b = base_trades("late", False)
    b = b[b.sym.isin(UNIV["사용자78"])]
    w = b.sort_values("ret").head(3)
    S = load("late")
    for x in w.itertuples():
        A = S[x.sym]
        cp = checkpoint(A, x.ei, x.xi, x.ent, E)
        dev, conf = htf_rsi(A)
        msg = "첫 -3% 도달 없음" if cp is None else (f"첫 -3% 도달 판단 봉 {A['idx'][cp + 1]} dev40 {dev[cp + 1]:.1f} "
                                                f"그때 팔면 {A['o'][cp + 1] / x.ent - 1:+.1%}")
        say(f"late 최악 거래 {x.sym} {x.t} {x.ses} 수익 {x.ret:+.1%}: {msg}")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(prev.rstrip("\n") + "\n" + _LOG.getvalue())
    say(f"덧붙임: {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "supp":
        supp()
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        for p_ in ("early", "late"):
            M.ctx(p_)
        lookahead_check()
    else:
        main()
