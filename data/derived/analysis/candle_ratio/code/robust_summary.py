"""robust 갈래 요약: robust_runs_{5m,1h}.npy → 144종목 중앙값·+ 비율과 귀무 분위.

종목별 값(단순 %):
  d_bh     = 전략 수익 − 보유 수익
  d_static = 전략 수익 − 같은 비중 그냥 보유 수익(x = 전략 평균 보유 비율)
  ddred    = 전략 최대 낙폭 − 같은 비중 그냥 보유 최대 낙폭(+면 전략이 덜 빠짐)
  ddred_bh = 전략 최대 낙폭 − 보유 최대 낙폭
귀무 분위 = 같은 통계(종목 중앙값 또는 + 비율)를 k별로 낸 귀무 분포에서 실제보다 작은 비율.
py code/robust_summary.py → results/robust_summary_{5m,1h}.csv, robust_named_{5m,1h}.csv, robust_summary.json
"""
import json

import numpy as np
import pandas as pd

from robust_common import RES

NAMED = ["MU", "BE", "NBIS"]
STATS = ["ret", "bh", "d_bh", "d_static", "ddred", "ddred_bh", "x", "mdd", "static_mdd"]


def per_symbol(a):
    """a[..., 8] → dict of 단순 % 배열."""
    ret = np.expm1(a[..., 0]) * 100; mdd = np.expm1(a[..., 1]) * 100
    bh = np.expm1(a[..., 2]) * 100; bhm = np.expm1(a[..., 3]) * 100
    st = a[..., 5] * 100; stm = a[..., 6] * 100
    return dict(ret=ret, bh=bh, d_bh=ret - bh, d_static=ret - st, ddred=mdd - stm, ddred_bh=mdd - bhm, x=a[..., 4],
                mdd=mdd, static_mdd=stm)


def summarize(base):
    meta = json.loads((RES / f"robust_runs_{base}_meta.json").read_text(encoding="utf-8"))
    A = np.load(RES / f"robust_runs_{base}.npy").astype(np.float64)  # [sym, var, mode, k, win, 8]
    syms = meta["syms"]
    rows, named = [], []
    for vi, vn in enumerate(meta["variants"]):
        for mi, mn in enumerate(meta["modes"]):
            for wi, wn in enumerate(meta["windows"]):
                ps = per_symbol(A[:, vi, mi, :, wi])  # [sym, k]
                ok = ~np.isnan(ps["ret"][:, 0])
                r = dict(base=base, variant=vn, mode=mn, window=wn, n_sym=int(ok.sum()))
                for s in STATS:
                    v = ps[s]
                    med = np.nanmedian(v, axis=0)  # [k]
                    r[s + "_med"] = med[0]
                    if s in ("d_bh", "d_static", "ddred", "ddred_bh"):
                        pos = np.nanmean(np.where(np.isnan(v), np.nan, v > 0), axis=0)
                        r[s + "_pos"] = pos[0]
                        r[s + "_null_med"] = float(np.median(med[1:]))
                        r[s + "_null_pct"] = float((med[1:] < med[0]).mean())
                        r[s + "_pos_null_med"] = float(np.median(pos[1:]))
                        r[s + "_pos_null_pct"] = float((pos[1:] < pos[0]).mean())
                rows.append(r)
                for sym in NAMED:
                    i = syms.index(sym)
                    named.append(dict(base=base, variant=vn, mode=mn, window=wn, sym=sym,
                                      **{s: float(ps[s][i, 0]) for s in STATS},
                                      d_static_null_pct=float((ps["d_static"][i, 1:] < ps["d_static"][i, 0]).mean()),
                                      ddred_null_pct=float((ps["ddred"][i, 1:] < ps["ddred"][i, 0]).mean())))
    df = pd.DataFrame(rows); nd = pd.DataFrame(named)
    df.to_csv(RES / f"robust_summary_{base}.csv", index=False)
    nd.to_csv(RES / f"robust_named_{base}.csv", index=False)
    return df, nd, meta


def show(df, base, meta):
    f1 = lambda v: f"{v:+.1f}"
    f0 = lambda v: f"{v * 100:.0f}"
    print(f"\n## {base}  (종목 {df.n_sym.max()}, 귀무 이동 {len(meta['ks']) - 1}개 k={meta['ks'][1]}~{meta['ks'][-1]})")
    for mode in ("ideal", "real"):
        print(f"\n### {mode}  — 열: 확인 수익 | 확인 보유 대비 차 중앙(+%)[귀무분위] | 확인 static 대비 수익 차(+%)[분위] | "
              f"확인 static 대비 낙폭 축소(+%)[분위] || 전체: 수익 | static 대비 수익 차[분위] | 낙폭 축소[분위] | x")
        x = df[df["mode"] == mode]
        for vn in meta["variants"]:
            c = x[(x.variant == vn) & (x.window == "conf")].iloc[0]
            a = x[(x.variant == vn) & (x.window == "full")].iloc[0]
            print(f"| {vn} | {f1(c.ret_med)} | {f1(c.d_bh_med)} ({f0(c.d_bh_pos)}) [{f0(c.d_bh_null_pct)}] | "
                  f"{f1(c.d_static_med)} ({f0(c.d_static_pos)}) [{f0(c.d_static_null_pct)}] | "
                  f"{f1(c.ddred_med)} ({f0(c.ddred_pos)}) [{f0(c.ddred_null_pct)}] || {f1(a.ret_med)} | "
                  f"{f1(a.d_static_med)} ({f0(a.d_static_pos)}) [{f0(a.d_static_null_pct)}] | "
                  f"{f1(a.ddred_med)} ({f0(a.ddred_pos)}) [{f0(a.ddred_null_pct)}] | {a.x_med:.2f} |")


if __name__ == "__main__":
    out = {}
    for base in ("5m", "1h"):
        df, nd, meta = summarize(base)
        show(df, base, meta)
        print("\n보유 중앙(확인/전체):", round(df[(df.window == "conf")].bh_med.iloc[0], 1), round(df[(df.window == "full")].bh_med.iloc[0], 1))
        print(nd[(nd.window != "sel") & nd.variant.isin([meta["variants"][0], "지연1봉", "시작+5분", "시작+20분", "35분", "45분"])]
              [["variant", "mode", "window", "sym", "ret", "bh", "d_static", "mdd", "static_mdd", "ddred", "x"]].round(1).to_string(index=False))
        out[base] = dict(n_null=len(meta["ks"]) - 1, ks=[meta["ks"][1], meta["ks"][-1]], D=meta["D"])
    (RES / "robust_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
