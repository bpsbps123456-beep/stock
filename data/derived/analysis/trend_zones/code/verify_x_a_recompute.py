"""검증 A: extremes.py의 E1·E2·E5 독립 재계산 + 감사(연구용, 기존 파일은 고치지 않음).

zones.load()로 원자료(OHLC)만 받고, 지표·순위·구간·채점은 여기서 따로 짠다.
- 지수이동평균/와일더 평균: pandas ewm(adjust=False, ignore_na=False) 규칙을 numpy 반복으로 직접 구현
- 순위: 원 코드 방식(rank/n - 0.5, 평균 +1/(2n) 치우침)과 중심 맞춘 방식((rank-0.5)/n - 0.5, 평균 0) 둘 다
- 우연 기대치: 구간별 양수 비율의 곱 × 정의 수, 그리고 구간마다 정의 이름을 섞는 순열 2만 번
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/trend_zones/code/verify_x_a_recompute.py
"""
import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402  (원자료 load만 사용)

OUT = os.path.join(HERE, "../results")
H = 10
PER = [("앞", "2026-01-02", "2026-05-29", True), ("뒤", "2026-06-01", "2026-10-09", False),
       ("2025", "2025-01-02", "2025-12-31", False), ("2024", "2024-01-02", "2024-12-31", False),
       ("2023하", "2023-07-01", "2023-12-29", False)]
PN = [p[0] for p in PER]


def ewm_np(x, alpha, minp):
    """pandas ewm(alpha, adjust=False, ignore_na=False, min_periods) 규칙을 그대로 반복."""
    x = np.asarray(x, float)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    w = np.full(N, np.nan)
    ow = np.ones(N)
    nobs = np.zeros(N)
    for i in range(T):
        cur = x[i]
        obs = np.isfinite(cur)
        has = np.isfinite(w)
        ow = np.where(has, ow * (1 - alpha), ow)
        upd = has & obs
        w = np.where(upd, (ow * w + alpha * np.where(obs, cur, 0)) / (ow + alpha), w)
        ow = np.where(upd, 1.0, ow)
        w = np.where(~has & obs, cur, w)
        ow = np.where(~has & obs, 1.0, ow)
        nobs += obs
        out[i] = np.where(nobs >= minp, w, np.nan)
    return out


def my_rsi(C, n):
    d = np.diff(C, axis=0, prepend=np.nan)
    up = ewm_np(np.where(np.isfinite(d), np.maximum(d, 0), np.nan), 1 / n, n)
    dn = ewm_np(np.where(np.isfinite(d), np.maximum(-d, 0), np.nan), 1 / n, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, r))      # 파인: 하락 0이면 100이 우선
    return np.where(np.isfinite(up) & np.isfinite(dn), r, np.nan)


def roll(a, n, fn):
    return getattr(pd.DataFrame(a).rolling(n, min_periods=n), fn)().to_numpy()


def main():
    O, Hh, L, C = Zm.load()
    dates = list(C.index)
    syms = list(C.columns)
    Ca, Oa, Ha, La = (x.to_numpy(float) for x in (C, O, Hh, L))
    T, N = Ca.shape
    nb = np.cumsum(np.isfinite(Ca), axis=0)
    valid = (nb >= 130) & np.isfinite(Ca) & np.isfinite(Oa)
    gaps = int(sum(((~np.isfinite(Ca[:, j])) & (nb[:, j] > 0)).sum() for j in range(N)))

    # 앞으로 10일 수익(t+1 시가 → t+11 시가)과 그날 순위
    f = np.full((T, N), np.nan)
    f[: T - 1 - H] = Oa[1 + H:] / Oa[1: T - H] - 1
    f = np.where(valid, f, np.nan)
    fin = np.isfinite(f)
    n_t = fin.sum(axis=1)
    rank = pd.DataFrame(f).rank(axis=1, method="average").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        rk_orig = rank / n_t[:, None] - 0.5
        rk_cent = (rank - 0.5) / n_t[:, None] - 0.5

    rows = {}
    for p, a, b, purge in PER:
        idx = [i for i, d in enumerate(dates) if a <= d <= b and i + 1 + H <= T - 1]
        if purge:
            end = max(i for i, d in enumerate(dates) if a <= d <= b)
            idx = [i for i in idx if i + 1 + H <= end]
        rows[p] = np.array(idx)

    # ---------------- 지표(독립 구현)
    defs = []
    rsis = {n: my_rsi(Ca, n) for n in (2, 3, 5, 7, 9, 14, 21, 28)}
    for n, r in rsis.items():
        for hi in (60, 65, 70, 75, 80, 85, 90, 95):
            defs.append((f"RSI{n} {hi}/{100 - hi}", r >= hi, r <= 100 - hi))
    for n in (7, 14):
        r = rsis[n]
        lo, hi_ = roll(r, 14, "min"), roll(r, 14, "max")
        with np.errstate(invalid="ignore", divide="ignore"):
            st = 100 * (r - lo) / (hi_ - lo)
        st = np.where(np.isfinite(st), st, np.nan)
        k = roll(st, 3, "mean")
        for hi in (80, 90, 95, 98):
            defs.append((f"StochRSI({n}) {hi}/{100 - hi}", k >= hi, k <= 100 - hi))
    for n in (10, 20, 50):
        m = roll(Ca, n, "mean")
        s = pd.DataFrame(Ca).rolling(n, min_periods=n).std(ddof=0).to_numpy()   # 모집단 표준편차(파인 ta.stdev와 같음)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (Ca - m) / s
        for k_ in (1.0, 1.5, 2.0, 2.5, 3.0):
            defs.append((f"볼린저{n} ±{k_}", z >= k_, z <= -k_))
    pc = np.vstack([np.full((1, N), np.nan), Ca[:-1]])
    tr = np.fmax(np.fmax(Ha - La, np.abs(Ha - pc)), np.abs(La - pc))
    atr = ewm_np(tr, 1 / 14, 14)
    ema_cache = {}
    for Ln in (5, 10, 20, 35, 50, 100):
        e = ewm_np(Ca, 2 / (Ln + 1), min(Ln, 130))
        ema_cache[Ln] = e
        dev = Ca - e
        widths = {"rms20": np.sqrt(roll(dev ** 2, 20, "mean")), "rms60": np.sqrt(roll(dev ** 2, 60, "mean")), "atr14": atr}
        for wn in ("rms20", "rms60", "atr14"):
            with np.errstate(invalid="ignore", divide="ignore"):
                z = dev / widths[wn]
            for b_ in (1.0, 1.5, 2.0, 2.5, 3.0):
                defs.append((f"칼만식 EMA{Ln}·{wn} ±{b_}", z >= b_, z <= -b_))
    lr = np.log(Ca) - np.log(pc)
    sd60 = pd.DataFrame(lr).rolling(60, min_periods=60).std(ddof=1).to_numpy()
    for Nn in (3, 5, 10, 20, 60):
        prev = np.vstack([np.full((Nn, N), np.nan), Ca[:-Nn]])
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.log(Ca / prev) / (sd60 * np.sqrt(Nn))
        for k_ in (1.0, 1.5, 2.0, 2.5, 3.0):
            defs.append((f"{Nn}일 수익 z ±{k_}", z >= k_, z <= -k_))
    hh = pd.DataFrame(Ha).rolling(252, min_periods=130).max().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = Ca / hh
    for x in (0.95, 0.98, 1.0):
        defs.append((f"52주 고점의 {x:.0%} 이상", pos >= x - 1e-12, None))
    for x in (0.5, 0.6, 0.7):
        defs.append((f"52주 고점의 {x:.0%} 이하", None, pos <= x))
    # 연속 상승·하락(벡터 방식, 원 코드의 반복문과 다르게)
    sgn = np.sign(np.diff(Ca, axis=0, prepend=np.nan))
    up = np.zeros((T, N)); dn = np.zeros((T, N))
    for i in range(1, T):
        up[i] = np.where(sgn[i] > 0, up[i - 1] + 1, 0)
        dn[i] = np.where(sgn[i] < 0, dn[i - 1] + 1, 0)
    for n in (4, 5, 6, 7):
        defs.append((f"연속 {n}일 상승/하락", up >= n, dn >= n))
    defs = [(nm, None if t is None else (t & valid), None if b is None else (b & valid)) for nm, t, b in defs]

    def score(M, rk, p, sign, extra=None):
        r = rows[p]
        Mr = M[r] & np.isfinite(rk[r])
        if extra is not None:
            Mr &= extra[r]
        tot = np.isfinite(rk[r]).sum()
        cnt = Mr.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(cnt > 0, np.where(Mr, rk[r], 0).sum(axis=1) / cnt, np.nan)
        v = sign * np.nanmean(m) if np.isfinite(m).any() else np.nan
        return float(v), float(Mr.sum() / tot), int(np.isfinite(m).sum())

    res = {}
    for nm, t, b in defs:
        res[nm] = {}
        for side, M, sg in (("위", t, -1), ("아래", b, 1)):
            if M is None:
                continue
            res[nm][side] = {p: {"orig": score(M, rk_orig, p, sg), "cent": score(M, rk_cent, p, sg)} for p in PN}

    # ---------------- 원 결과(json)와 비교
    J = json.load(open(os.path.join(OUT, "extremes.json"), encoding="utf-8"))
    jd = {d["name"]: d for d in J["defs"]}
    maxdiff, worst, nmatch = 0.0, ("-", "-", "-", 0, 0), 0
    for nm in res:
        for side in res[nm]:
            for p in PN:
                a = res[nm][side][p]["orig"][0]
                bj = jd[nm][side][p]["타당도"]
                if bj is None or not np.isfinite(bj) or not np.isfinite(a):
                    continue
                nmatch += 1
                if abs(a - bj) > maxdiff:
                    maxdiff, worst = abs(a - bj), (nm, side, p, a, bj)

    lines = [f"검증 A — 종목 {N}, 날짜 {T}({dates[0]}~{dates[-1]}), 정의 {len(defs)}개, 중간 빈 종가 {gaps}칸",
             f"원 json과 타당도 비교 {nmatch}칸: 최대 차이 {maxdiff * 100:.4f}%p ({worst[0]} {worst[1]} {worst[2]})"]
    lines.append("구간별 하루 평균 종목 수 n과 원 순위의 치우침 1/(2n): " + ", ".join(
        f"{p} n {n_t[rows[p]].mean():.0f} → {np.mean(0.5 / n_t[rows[p]]) * 100:.2f}%p" for p in PN))

    pct = lambda v: "   —  " if v is None or not np.isfinite(v) else f"{v * 100:+6.2f}"
    out = {"maxdiff": maxdiff, "worst": worst, "sides": {}}
    rng = np.random.default_rng(20261011)
    for side in ("위", "아래"):
        ok = [nm for nm in res if side in res[nm] and 0.01 <= res[nm][side]["앞"]["orig"][1] <= 0.20]
        lines += ["", f"== {'극상방' if side == '위' else '극하방'}: 자격 {len(ok)}개"]
        so = {"n_ok": len(ok)}
        for kind in ("orig", "cent"):
            V = np.array([[res[nm][side][p][kind][0] for p in PN] for nm in ok])
            fins = np.isfinite(V)
            shares = [float(np.mean(V[fins[:, k], k] > 0)) for k in range(5)]
            pos = np.where(fins, V > 0, False)
            allp = [ok[i] for i in np.where(pos.all(axis=1))[0]]
            exp_ind = len(ok) * np.prod(pos.mean(axis=0))
            perm = np.empty(20000, int)
            for it in range(20000):
                cols = [pos[rng.permutation(len(ok)), k] for k in range(5)]
                perm[it] = np.all(np.vstack(cols), axis=0).sum()
            p_ge = float(np.mean(perm >= len(allp)))
            ge4 = int((pos.sum(axis=1) >= 4).sum())
            lines.append(f"[{'원 순위' if kind == 'orig' else '중심 순위'}] 양수 비율 " + " ".join(
                f"{p} {s * 100:4.1f}%" for p, s in zip(PN, shares)) + f" | 다섯 구간 양수 {len(allp)}개 {allp}")
            lines.append(f"      우연 기대 {exp_ind:.2f}개(곱), 순열 평균 {perm.mean():.2f}개, 순열에서 {len(allp)}개 이상 나올 확률 {p_ge:.2f}"
                         f", 0개일 확률 {np.mean(perm == 0):.2f} | 4구간 이상 양수 {ge4}개")
            so[kind] = {"shares": shares, "allpos": allp, "exp_prod": float(exp_ind), "perm_mean": float(perm.mean()),
                        "p_ge_obs": p_ge, "p_zero": float(np.mean(perm == 0)), "ge4": ge4}
        best = max(ok, key=lambda nm: res[nm][side]["앞"]["orig"][0])
        so["best"] = best
        for nm in dict.fromkeys([best, "RSI14 70/30", "칼만식 EMA50·rms20 ±2.0", "칼만식 EMA5·rms60 ±2.5", "60일 수익 z ±2.0"]):
            if side not in res[nm]:
                continue
            r = res[nm][side]
            lines.append(f"  {nm:<26} 원 " + " ".join(pct(r[p]['orig'][0]) for p in PN) + " | 중심 "
                         + " ".join(pct(r[p]['cent'][0]) for p in PN) + f" | 앞 비중 {r['앞']['orig'][1] * 100:.1f}% 날짜 "
                         + "/".join(str(r[p]['orig'][2]) for p in PN))
            so[nm] = {p: r[p] for p in PN}
        out["sides"][side] = so

    # ---------------- E5 종목별(RSI14 70/30)
    r14 = rsis[14]
    top, bot = (r14 >= 70) & valid, (r14 <= 30) & valid
    lines += ["", "== 종목별 RSI14 70/30: 극단 뒤 10일 순위가 그 종목 평소(같은 구간 비극단 날)보다 맞는 방향인 종목 수"]
    e5 = {}
    for p in PN:
        rr = rows[p]
        parts = []
        for side, M, sg in (("위(덜 오름)", top, -1), ("아래(더 오름)", bot, 1)):
            good = cnt = 0
            diffs = []
            for j in range(N):
                x = rk_orig[rr, j]
                fx = np.isfinite(x)
                m = M[rr, j] & fx
                if m.sum() >= 5 and fx.sum() > m.sum():
                    cnt += 1
                    d = sg * (x[m].mean() - x[fx & ~m].mean())
                    diffs.append(d)
                    good += d > 0
            from math import comb
            pbin = sum(comb(cnt, k) for k in range(good, cnt + 1)) / 2 ** cnt if cnt else np.nan
            parts.append(f"{side} {good}/{cnt} (단측 이항 p {pbin:.3f}, 중앙 차 {np.median(diffs) * 100 if diffs else np.nan:+.2f}%p)")
            e5[f"{p}·{side}"] = [good, cnt, pbin]
        lines.append(f"[{p}] " + "  ".join(parts))
    out["e5"] = e5

    # ---------------- 감사: 원 extremes.py 방식 rank 평균, 2025 말 구간 겹침
    allmean = {p: float(np.nanmean(np.nanmean(np.where(np.isfinite(rk_orig[rows[p]]), rk_orig[rows[p]], np.nan), axis=1))) for p in PN}
    lines += ["", "감사: 전 종목 원 순위 평균(무작위 선택의 기대 rk) " + ", ".join(f"{p} {v * 100:+.2f}%p" for p, v in allmean.items())]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "verify_x_a_recompute.txt"), "w", encoding="utf-8") as fo:
        fo.write(txt)
    with open(os.path.join(OUT, "verify_x_a_recompute.json"), "w", encoding="utf-8") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
