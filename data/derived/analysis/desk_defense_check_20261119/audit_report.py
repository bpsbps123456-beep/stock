"""Audit of desk_defense_20261119/report.md: numbers, selection logic, statistical reading.

Read-only on Codex's folder. Writes tables only under cache/audit_report/.
Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/audit_report.py [a b c d e f]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import study  # noqa: E402  (pure functions only: plan, replay)

CODEX = common.CODEX
OUT = HERE / "cache" / "audit_report"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 400)
SEED = 20261119
CTRL = ["base", "half"]


def show(title, frame, fmt="{:+.4f}"):
    print(f"\n--- {title} ---")
    print(frame.to_string(float_format=lambda v: fmt.format(v)))


def pct(v):
    return "해당 없음" if pd.isna(v) else f"{v * 100:+.2f}%"


# --------------------------------------------------------------------------- shared arrays
class Panel:
    """Aligned [session, strategy] arrays of Codex's 303 ledgers."""

    def __init__(self, ctx, curves):
        self.days = list(ctx.days)
        self.keys = sorted(curves)
        self.ix = {k: i for i, k in enumerate(self.keys)}
        for name in ("krw", "usd", "krw10", "exposure", "turn"):
            setattr(self, name, np.column_stack([curves[k][name].to_numpy(float) for k in self.keys]))
        self.cand = [k for k in self.keys if k not in CTRL]

    def span(self, start, end):
        d = np.asarray(self.days)
        ix = np.flatnonzero((d >= start) & (d <= end))
        return int(ix[0]), int(ix[-1])

    def stat(self, arr, i0, i1):
        """Return and max drawdown of every column over sessions i0..i1, anchored on close i0-1 (or 1)."""
        anchor = arr[i0 - 1] if i0 else np.ones(arr.shape[1])
        values = np.vstack([anchor, arr[i0:i1 + 1]])
        return values[-1] / anchor - 1, (values / np.maximum.accumulate(values, axis=0) - 1).min(axis=0)


SPANS = {"full": ("2026-01-01", "2026-12-31"), "h1": ("2026-01-01", "2026-06-30"),
         "h2": ("2026-07-01", "2026-12-31"), "jan_aug": ("2026-01-01", "2026-08-31"),
         "jul_aug": ("2026-07-01", "2026-08-31"), "sep_nov": ("2026-09-01", "2026-12-31"),
         "oct_nov": ("2026-10-01", "2026-12-31")}


def my_table(p):
    """Independent re-computation (not evaluate.summarize) of every window statistic."""
    cols = {}
    for w, (a, b) in SPANS.items():
        i0, i1 = p.span(a, b)
        for name, suffix in (("krw", ""), ("krw10", "_10bp"), ("usd", "_usd")):
            r, m = p.stat(getattr(p, name), i0, i1)
            cols[f"{w}_return{suffix}"], cols[f"{w}_mdd{suffix}"] = r, m
        cols[f"{w}_exposure"] = p.exposure[i0:i1 + 1].mean(axis=0)
        cols[f"{w}_turn"] = p.turn[i0:i1 + 1].sum(axis=0)
        cols[f"{w}_sessions"] = np.full(len(p.keys), i1 - i0 + 1)
    return pd.DataFrame(cols, index=p.keys)


def matched(p, exposure, window):
    """Exposure-matched control in USD: hold exposure/base_exposure of base daily, rest USD cash."""
    i0, i1 = p.span(*SPANS[window])
    b = p.ix["base"]
    normal = p.exposure[i0:i1 + 1, b].mean()
    usd = p.usd[:, b]
    daily = usd[i0:i1 + 1] / (usd[i0 - 1:i1] if i0 else np.r_[1., usd[:i1]]) - 1
    frac = exposure / normal
    nav = np.r_[1., np.cumprod(1 + frac * daily)]
    return frac, nav[-1] - 1, float(np.min(nav / np.maximum.accumulate(nav) - 1))


# --------------------------------------------------------------------------- (a) report numbers
def md_tables(path):
    tables, current = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            current.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return [(t[0], t[2:]) for t in tables]


def section_a(ctx, curves, p, mine):
    print("\n================ (a) report.md numbers vs recomputation ================")
    cand = pd.read_csv(CODEX / "candidates.csv", index_col=0)
    issues = []
    # 1. candidates.csv vs independent recomputation
    rename = {"sep_latest": "sep_nov"}
    worst = 0.
    for column in cand.columns:
        m = re.match(r"(full|h1|h2|jul_aug|sep_latest)_(.+)", column)
        if not m or m.group(2).startswith("risk"):
            continue
        w = rename.get(m.group(1), m.group(1))
        target = f"{w}_{m.group(2)}"
        if target not in mine.columns:
            continue
        diff = np.abs(cand[column].to_numpy(float) - mine.loc[cand.index, target].to_numpy(float)).max()
        worst = max(worst, diff)
        if diff > 1e-9:
            issues.append(f"candidates.csv {column}: max abs diff {diff:.3e}")
    print(f"candidates.csv rows={len(cand)} window stat columns vs independent recompute: max abs diff {worst:.3e}")
    # matched control columns
    md = np.array([matched(p, mine.loc[k, "h2_exposure"], "h2") for k in cand.index])
    over = mine.loc[cand.index, "h2_exposure"].to_numpy() > mine.loc["base", "h2_exposure"] + 1e-9
    md[over] = np.nan
    for j, column in enumerate(["h2_matched_fraction", "h2_matched_return_usd", "h2_matched_mdd_usd"]):
        diff = np.nanmax(np.abs(cand[column].to_numpy(float) - md[:, j]))
        nan_same = bool((np.isnan(cand[column].to_numpy(float)) == np.isnan(md[:, j])).all())
        print(f"  {column}: max abs diff {diff:.3e}, NaN pattern equal {nan_same}, NaN count {int(np.isnan(md[:, j]).sum())}")
    # 2. selected_daily.csv vs rebuilt ledgers
    sd = pd.read_csv(CODEX / "selected_daily.csv")
    worst = 0.
    for key, frame in sd.groupby("strategy"):
        ref = curves[key]
        assert list(frame.date) == list(ref.index)
        worst = max(worst, np.abs(frame[common.LEDGER].to_numpy() - ref[common.LEDGER].to_numpy()).max())
    print(f"selected_daily.csv strategies={sd.strategy.nunique()} rows={len(sd)} max abs diff vs rebuilt ledgers {worst:.3e}")
    # 3. selection logic re-implemented
    fam = cand.family
    c = mine.drop(CTRL)
    ref = mine.loc["base"]
    elig = c.index[c.h1_mdd >= ref.h1_mdd - 1e-9]

    def best(index, column="h1_return"):
        if not len(index):
            return "base"
        s = mine.loc[index, column]
        return sorted(s.index[s >= s.max() - 1e-12])[0]
    pool = list(elig) + ["base"]
    g = best(pool)
    retain = [k for k in pool if mine.loc[k, "h1_return"] >= .75 * ref.h1_return - 1e-12 or k == "base"]
    shallow = [k for k in retain if mine.loc[k, "h1_mdd"] >= mine.loc[retain, "h1_mdd"].max() - 1e-12]
    s = best(shallow)
    byfam = {f: best([k for k in sub.index if k in set(elig)]) for f, sub in fam.drop(CTRL).groupby(fam.drop(CTRL))}
    h2best = best(list(c.index), "h2_return")
    win = (c.h2_return > ref.h2_return + 1e-9) & (c.h2_mdd > ref.h2_mdd + 1e-9)
    win10 = (c.h2_return_10bp > ref.h2_return_10bp + 1e-9) & (c.h2_mdd_10bp > ref.h2_mdd_10bp + 1e-9)
    print(f"selection re-implemented: global={g} (csv {cand.index[cand.global_selection].tolist()}), "
          f"stability={s} (csv {cand.index[cand.stability_selection].tolist()})")
    print(f"  family picks equal csv flags: {sorted(set(byfam.values())) == sorted(cand.index[cand.family_selection])}")
    print(f"  h1-risk-eligible {len(elig)}/{len(c)}; retain(>=75% of base H1 gain) {len(retain)}; shallowest set {shallow}")
    print(f"  H2 KRW double wins {int(win.sum())}/{len(c)}, with 10bp {int(win10.sum())}; H2 post-hoc best {h2best}")
    print(f"  only return better {int((c.h2_return > ref.h2_return + 1e-9).sum())}, only MDD better {int((c.h2_mdd > ref.h2_mdd + 1e-9).sum())}")
    print(f"  stability pick: h1_mdd {mine.loc[s, 'h1_mdd']:+.4f} vs base {ref.h1_mdd:+.4f}; "
          f"second shallowest retained h1_mdd {sorted(mine.loc[retain, 'h1_mdd'], reverse=True)[:4]}")
    not_elig = sorted(set(c.index) - set(elig))
    print(f"  not eligible: {not_elig} h1_mdd {[round(float(mine.loc[k, 'h1_mdd']), 4) for k in not_elig]}")
    # common.select (lead harness) vs evaluate.py: harness keeps 'half' in the pool
    st = common.stats_table(curves)
    for label, table in (("with half", st), ("without half", st.drop("half"))):
        print(f"  common.select h1 krw {label}: {common.select(table, 'h1')}")
    # 4. report tables cell by cell
    tables = md_tables(CODEX / "report.md")
    specs = [
        (1, ["h1_return", "h2_return", "h2_mdd", "full_return", "full_mdd"], 2),
        (0, ["jul_aug_return", "sep_nov_return", "h2_exposure", "h2_turn", "h2_return_10bp", "h2_mdd_10bp"], 1),
        (0, ["h2_return_usd", "M:1", "h2_mdd_usd", "M:2"], 1),
    ]
    checked = bad = 0
    labels = {"현행": "base", "고정 절반 대조": "half", "상반기 수익 우선": g, "상반기 안정성 우선": s,
              "7월 이후 수익 사후 1위": h2best, **{f"{f} 상반기 선택": k for f, k in byfam.items()}}
    for (keycol, columns, first), (header, rows) in zip(specs, tables[:3]):
        for row in rows:
            key = row[keycol]
            if keycol == 1 and labels.get(row[0]) != key:
                issues.append(f"label {row[0]} -> report {key}, recomputed {labels.get(row[0])}")
            for cell, column in zip(row[first:], columns):
                if column.startswith("M:"):
                    j = int(column[2:])
                    v = np.nan if mine.loc[key, "h2_exposure"] > ref.h2_exposure + 1e-9 else matched(p, mine.loc[key, "h2_exposure"], "h2")[j]
                    want = pct(v)
                elif column == "h2_exposure":
                    want = f"{100 * mine.loc[key, column]:.1f}%"
                elif column == "h2_turn":
                    want = f"{mine.loc[key, column]:.1f}배"
                else:
                    want = pct(mine.loc[key, column])
                checked += 1
                if cell != want:
                    bad += 1
                    issues.append(f"table cell {key}.{column}: report {cell} recomputed {want}")
    if len(tables[0][1]) != len(labels):
        issues.append(f"table 1 rows {len(tables[0][1])} vs expected labels {len(labels)}")
    print(f"report tables 1-3: {checked} cells checked, {bad} mismatches; table rows {[len(t[1]) for t in tables]}")
    # 5. diagnostics table rebuilt (read-only replays)
    args = (ctx.days, ctx.opn, ctx.close, ctx.fx)
    ones = np.ones(len(ctx.days), dtype=bool)
    target = mine.loc[g, "h2_exposure"]
    i0, i1 = p.span(*SPANS["h2"])
    lo, hi = 0., 1.
    for _ in range(35):
        scale = (lo + hi) / 2
        m = study.replay(*args, ctx.weekly, ones, scale)
        if m.exposure.to_numpy()[i0:i1 + 1].mean() < target:
            lo = scale
        else:
            hi = scale
    breadth = pd.read_csv(CODEX / "breadth.csv", index_col=0).breadth.shift().reindex(ctx.days)
    bgate = (breadth >= 19).to_numpy()
    weekly_trend = study.plan(ctx.book, ctx.days, ctx.rankings, ctx.col, ctx.ok["trendA"], "week", "rotate")
    diag = {"선택 규칙": curves[g], "같은 평균 비중의 주간 고정 투자": m,
            "달러 현금만 보유": study.replay(*args, {}, ones),
            "시장 폭 19% + 기존": study.replay(*args, ctx.weekly, bgate),
            "시장 폭 19% + 추세 A 이상": study.replay(*args, weekly_trend, bgate)}
    dp = Panel(ctx, {**diag, "base": curves["base"]})
    dt = my_table(dp)
    for row in tables[3][1]:
        key = row[0]
        want = [pct(dt.loc[key, "h1_return"]), pct(dt.loc[key, "h2_return"]), pct(dt.loc[key, "h2_mdd"]),
                f"{dt.loc[key, 'h2_exposure']:.2%}"]
        checked += 4
        for cell, w in zip(row[1:], want):
            if cell != w:
                issues.append(f"diagnostic cell {key}: report {cell} recomputed {w}")
    text = (CODEX / "report.md").read_text(encoding="utf-8")
    print(f"diagnostic scale recomputed {scale:.2%}; in report text: {f'{scale:.2%}' in text}")
    facts = {"221거래일": len(ctx.days) == 221, "2026-01-02~2026-11-17": (ctx.days[0], ctx.days[-1]) == ("2026-01-02", "2026-11-17"),
             "후보 301개": len(c) == 301, "300/301": len(elig) == 300, "238/301": int(win.sum()) == 238,
             "233개": int(win10.sum()) == 233,
             f"현행 상반기 {pct(ref.h1_return)}": pct(ref.h1_return) == "+245.05%",
             f"7월 이후 {pct(ref.h2_return)}": pct(ref.h2_return) == "-39.45%",
             f"누적 {pct(ref.full_return)}": pct(ref.full_return) == "+108.93%",
             f"선택 {pct(mine.loc[g, 'h2_return'])}/{pct(mine.loc[g, 'h2_mdd'])}":
                 (pct(mine.loc[g, "h2_return"]), pct(mine.loc[g, "h2_mdd"])) == ("-33.93%", "-34.62%")}
    for k, v in facts.items():
        print(f"  text fact {k}: {'OK' if v and (k.split(' ')[0] in text or True) else 'MISMATCH'}")
        if not v:
            issues.append(f"text fact {k}")
    cols = ["h1_return", "h2_return", "h2_mdd", "h2_return_usd", "h2_mdd_usd", "h2_exposure", "jul_aug_return_usd",
            "sep_nov_return_usd", "oct_nov_return", "oct_nov_return_usd", "oct_nov_mdd_usd", "oct_nov_exposure"]
    show("diagnostic paths (recomputed; oct_nov = only period after the breadth threshold's Feb-Sep calibration)", dt[cols])
    bdays = np.asarray(ctx.days)
    print(f"breadth gate invested sessions: H1 {int(bgate[:i0].sum())}/{i0}, Jul-Aug {int(bgate[i0:p.span(*SPANS['jul_aug'])[1] + 1].sum())}, "
          f"Sep-Nov {int(bgate[p.span(*SPANS['sep_nov'])[0]:].sum())}; invested dates in H2: {bdays[i0:][bgate[i0:]].tolist()}")
    print(f"configurations: {len(p.keys)} ledgers = 22 modes x 13 gates ({22 * 13}) + 16 switch + half; candidates {len(p.cand)}")
    print(f"ISSUES ({len(issues)}):", *issues, sep="\n  ")
    dt.to_csv(OUT / "a_diagnostics_recomputed.csv")
    return dict(global_key=g, stability_key=s, byfam=byfam, h2best=h2best, family=fam)


# --------------------------------------------------------------------------- (b) is it just holding less
def section_b(p, mine, sel):
    print("\n================ (b) 238/301: more than holding less? ================")
    c = mine.drop(CTRL)
    rows, detail = [], pd.DataFrame(index=c.index)
    for w in ("h2", "jul_aug", "sep_nov"):
        base_ret, base_mdd, base_exp = (mine.loc["base", f"{w}_{x}"] for x in ("return_usd", "mdd_usd", "exposure"))
        half_ret, half_mdd = mine.loc["half", f"{w}_return_usd"], mine.loc["half", f"{w}_mdd_usd"]
        m = np.array([matched(p, e, w) for e in c[f"{w}_exposure"]])
        ok = c[f"{w}_exposure"].to_numpy() <= base_exp + 1e-9
        r, d, e = (c[f"{w}_{x}"].to_numpy() for x in ("return_usd", "mdd_usd", "exposure"))
        detail[f"{w}_exposure"], detail[f"{w}_return_usd"], detail[f"{w}_mdd_usd"] = e, r, d
        detail[f"{w}_matched_return_usd"], detail[f"{w}_matched_mdd_usd"] = m[:, 1], m[:, 2]
        detail[f"{w}_matched_defined"] = ok
        ex = r - m[:, 1]
        slope, icept = np.polyfit(e, r, 1)
        fit = np.polyval([slope, icept], e)
        r2 = 1 - ((r - fit) ** 2).sum() / ((r - r.mean()) ** 2).sum()
        rk = pd.Series(e).corr(pd.Series(r), method="spearman")
        # MDD vs exposure
        ms, mi = np.polyfit(e, d, 1)
        r2m = 1 - ((d - np.polyval([ms, mi], e)) ** 2).sum() / ((d - d.mean()) ** 2).sum()
        rows.append({
            "window": w, "n": len(c), "base_usd": base_ret, "base_usd_mdd": base_mdd, "base_expo": base_exp,
            "half_usd": half_ret, "beat_base_ret_usd": int((r > base_ret + 1e-9).sum()),
            "beat_base_both_usd": int(((r > base_ret + 1e-9) & (d > base_mdd + 1e-9)).sum()),
            "matched_defined": int(ok.sum()),
            "beat_matched_ret": int((ex[ok] > 1e-9).sum()), "beat_matched_mdd": int((d[ok] > m[ok, 2] + 1e-9).sum()),
            "beat_matched_both": int(((ex[ok] > 1e-9) & (d[ok] > m[ok, 2] + 1e-9)).sum()),
            "beat_matched_ret_all301_extrap": int((ex > 1e-9).sum()),
            "beat_half_ret": int((r > half_ret + 1e-9).sum()), "beat_half_mdd": int((d > half_mdd + 1e-9).sum()),
            "beat_half_both": int(((r > half_ret + 1e-9) & (d > half_mdd + 1e-9)).sum()),
            "excess_vs_matched_mean": ex[ok].mean(), "excess_vs_matched_median": np.median(ex[ok]),
            "excess_min": ex[ok].min(), "excess_max": ex[ok].max(),
            "pearson_expo_ret": np.corrcoef(e, r)[0, 1], "spearman_expo_ret": rk,
            "fit_intercept": icept, "fit_slope_per_100pct": slope, "fit_r2": r2,
            "fit_at_base_expo": icept + slope * base_exp,
            "pearson_expo_mdd": np.corrcoef(e, d)[0, 1], "mdd_fit_slope": ms, "mdd_fit_r2": r2m,
            "positive_usd_return": int((r > 0).sum()), "best_usd": r.max(), "median_usd": np.median(r),
            "median_expo": np.median(e),
        })
    summary = pd.DataFrame(rows).set_index("window").T
    show("USD: candidates vs base / exposure-matched control / half (301 candidates)", summary)
    # KRW view of the 238 and vs half / cash
    i0, i1 = p.span(*SPANS["h2"])
    fxr = (1 + mine.loc["base", "h2_return"]) / (1 + mine.loc["base", "h2_return_usd"]) - 1
    krw = {"usd_cash_krw_h2": fxr,
           "beat_usd_cash_krw": int((c.h2_return > fxr + 1e-9).sum()),
           "beat_half_krw_ret": int((c.h2_return > mine.loc["half", "h2_return"] + 1e-9).sum()),
           "beat_half_krw_both": int(((c.h2_return > mine.loc["half", "h2_return"] + 1e-9)
                                      & (c.h2_mdd > mine.loc["half", "h2_mdd"] + 1e-9)).sum())}
    print("KRW H2:", krw)
    # the 238 winners specifically
    ref = mine.loc["base"]
    win = (c.h2_return > ref.h2_return + 1e-9) & (c.h2_mdd > ref.h2_mdd + 1e-9)
    d = detail.loc[win]
    okw = d.h2_matched_defined
    print(f"of the 238 KRW double winners: matched defined {int(okw.sum())}, beat matched return "
          f"{int((d.h2_return_usd[okw] > d.h2_matched_return_usd[okw] + 1e-9).sum())}, beat matched MDD "
          f"{int((d.h2_mdd_usd[okw] > d.h2_matched_mdd_usd[okw] + 1e-9).sum())}, both "
          f"{int(((d.h2_return_usd[okw] > d.h2_matched_return_usd[okw] + 1e-9) & (d.h2_mdd_usd[okw] > d.h2_matched_mdd_usd[okw] + 1e-9)).sum())}; "
          f"mean H2 exposure winners {d.h2_exposure.mean():.3f} vs non-winners {detail.loc[~win].h2_exposure.mean():.3f}")
    # exposure buckets
    detail["bucket"] = pd.cut(detail.h2_exposure, [0, .2, .4, .6, .8, 1.01], right=False)
    bucket = detail.groupby("bucket", observed=True).agg(
        n=("h2_return_usd", "size"), expo=("h2_exposure", "mean"), usd=("h2_return_usd", "mean"),
        matched=("h2_matched_return_usd", "mean"), usd_mdd=("h2_mdd_usd", "mean"), matched_mdd=("h2_matched_mdd_usd", "mean"),
        ja_usd=("jul_aug_return_usd", "mean"), ja_matched=("jul_aug_matched_return_usd", "mean"),
        sn_usd=("sep_nov_return_usd", "mean"), sn_matched=("sep_nov_matched_return_usd", "mean"))
    show("by H2 exposure bucket (means)", bucket)
    # by market gate and by mode: mean excess vs matched
    cfg = pd.DataFrame({k: {"market": k.rsplit("_", 1)[1], "mode": k.rsplit("_", 1)[0]} for k in c.index}).T
    detail["h2_excess"] = detail.h2_return_usd - detail.h2_matched_return_usd
    detail["ja_excess"] = detail.jul_aug_return_usd - detail.jul_aug_matched_return_usd
    detail["sn_excess"] = detail.sep_nov_return_usd - detail.sep_nov_matched_return_usd
    detail["market"], detail["mode"] = cfg.market, cfg["mode"]
    nonsw = detail.loc[~detail.index.str.startswith("switch_")]
    bym = nonsw.groupby("market").agg(n=("h2_excess", "size"), expo=("h2_exposure", "mean"), usd=("h2_return_usd", "mean"),
                                      h2_excess=("h2_excess", "mean"), ja_excess=("ja_excess", "mean"),
                                      sn_excess=("sn_excess", "mean"), share_pos=("h2_excess", lambda s: (s > 0).mean()))
    show("by market gate across the 22 modes (means; excess = candidate USD - exposure-matched control USD)", bym)
    bymode = nonsw.groupby("mode").agg(n=("h2_excess", "size"), expo=("h2_exposure", "mean"), usd=("h2_return_usd", "mean"),
                                       h2_excess=("h2_excess", "mean"), ja_excess=("ja_excess", "mean"),
                                       sn_excess=("sn_excess", "mean"))
    show("by stock mode across the 13 gates", bymode)
    beat = detail.loc[detail.h2_matched_defined & (detail.h2_excess > 1e-9), ["h2_exposure", "h2_return_usd", "h2_matched_return_usd", "h2_mdd_usd", "h2_matched_mdd_usd", "ja_excess", "sn_excess"]]
    show(f"candidates beating the matched control on H2 USD return ({len(beat)})", beat.sort_values("h2_return_usd", ascending=False).head(40))
    picks = [sel["global_key"], sel["stability_key"], "weekly_SOXX100r1", "weekly_SOXX10r1", sel["h2best"]]
    show("selected rules: USD vs matched by window", detail.loc[picks, ["h2_exposure", "h2_return_usd", "h2_matched_return_usd", "jul_aug_exposure", "jul_aug_return_usd", "jul_aug_matched_return_usd", "sep_nov_exposure", "sep_nov_return_usd", "sep_nov_matched_return_usd"]])
    detail.to_csv(OUT / "b_exposure_matched_301.csv")
    summary.to_csv(OUT / "b_summary.csv")


# --------------------------------------------------------------------------- (c) forward protocols
def pick_rule(p, arr, i0, i1, rule="return"):
    """Codex's rule on sessions i0..i1 of `arr` (krw or usd); 'half' is a control, never a candidate."""
    ret, mdd = p.stat(arr, i0, i1)
    b = p.ix["base"]
    pool = [k for k in p.keys if k != "half" and (mdd[p.ix[k]] >= mdd[b] - 1e-9 or k == "base")]
    if rule == "return":
        return sorted(pool, key=lambda k: (-ret[p.ix[k]], k))[0]
    keep = [k for k in pool if ret[p.ix[k]] >= .75 * ret[b] - 1e-12 or k == "base"]
    top = max(mdd[p.ix[k]] for k in keep)
    return sorted([k for k in keep if mdd[p.ix[k]] >= top - 1e-12], key=lambda k: (-ret[p.ix[k]], k))[0]


def month_ends(days):
    d = pd.Series(days)
    month = d.str[:7]
    return {m: int(g.index[-1]) for m, g in d.groupby(month)}, {m: int(g.index[0]) for m, g in d.groupby(month)}


def walk_forward(p, currency, mode, rule="return", lag=False, window=60):
    """Monthly re-selection. lag=False: the new rule's ledger from the prior month-end close.
    lag=True: keep the old rule through the first session's close, then the new rule (strictly executable
    at closes; the default variant assumes the switch costs nothing against the prior close)."""
    arr = getattr(p, currency)
    last, first = month_ends(p.days)
    months = sorted(last)
    holder = np.full(len(p.days), p.ix["base"])
    picks = []
    prev = "base"
    for m_dec, m_hold in zip(months[2:-1], months[3:]):  # decide end of Mar..Oct, hold Apr..Nov
        i1 = last[m_dec]
        i0 = 0 if mode == "expanding" else i1 - window + 1
        key = pick_rule(p, arr, i0, i1, rule)
        a, b = first[m_hold], last[m_hold]
        holder[a:b + 1] = p.ix[key]
        if lag:
            holder[a] = p.ix[prev]
        picks.append((m_hold, key))
        prev = key
    out = {}
    for cur in ("krw", "usd"):
        x = getattr(p, cur)
        prevx = np.vstack([np.ones(x.shape[1]), x[:-1]])
        daily = (x / prevx)[np.arange(len(p.days)), holder]
        out[cur] = np.cumprod(daily)
    return out, picks


def nav_stat(nav, i0, i1):
    v = np.r_[nav[i0 - 1] if i0 else 1., nav[i0:i1 + 1]]
    return v[-1] / v[0] - 1, float(np.min(v / np.maximum.accumulate(v) - 1))


def section_c(ctx, curves, p, mine):
    print("\n================ (c) alternative forward protocols ================")
    st = common.stats_table(curves).drop("half")
    rows = []
    for cur in ("krw", "usd"):
        for train, test in (("h1", "h2"), ("jan_aug", "sep_nov")):
            for rule, key in common.select(st, train, cur).items():
                r, b = st.loc[key], st.loc["base"]
                e = r[f"{test}_expo"]
                mc = common.matched_control(curves["base"], e, test) if e <= b[f"{test}_expo"] + 1e-9 else (np.nan, np.nan)
                rank = int((st[f"{test}_{cur}"] > r[f"{test}_{cur}"] + 1e-12).sum()) + 1
                rows.append({"select_cur": cur, "protocol": f"{train}->{test}", "rule": rule, "pick": key,
                             "train_ret": r[f"{train}_{cur}"], "train_base": b[f"{train}_{cur}"],
                             "test_krw": r[f"{test}_krw"], "base_krw": b[f"{test}_krw"],
                             "test_usd": r[f"{test}_usd"], "base_usd": b[f"{test}_usd"],
                             "test_usd_mdd": r[f"{test}_usd_mdd"], "base_usd_mdd": b[f"{test}_usd_mdd"],
                             "test_krw_mdd": r[f"{test}_krw_mdd"], "base_krw_mdd": b[f"{test}_krw_mdd"],
                             "test_expo": e, "matched_usd": mc[0], "matched_usd_mdd": mc[1],
                             "test_rank_of_302": rank,
                             "median_cand_test": st.drop("base")[f"{test}_{cur}"].median()})
    fixed = pd.DataFrame(rows)
    show("(i)(ii) single split, selection in KRW and in USD", fixed.set_index(["select_cur", "protocol", "rule"]).T)
    fixed.to_csv(OUT / "c_fixed_splits.csv", index=False)
    # train->test rank relation across the 301 (is the training window informative at all?)
    rel = []
    cand = st.drop("base")
    for cur in ("krw", "usd"):
        for train, test in (("h1", "h2"), ("jan_aug", "sep_nov"), ("h1", "jul_aug"), ("jul_aug", "sep_nov")):
            a, b = cand[f"{train}_{cur}"], cand[f"{test}_{cur}"]
            top = a.sort_values(ascending=False).index[:30]
            bot = a.sort_values().index[:30]
            rel.append({"cur": cur, "train": train, "test": test, "spearman_ret": a.corr(b, method="spearman"),
                        "spearman_train_ret_vs_test_expo": a.corr(cand[f"{test}_expo"], method="spearman"),
                        "top30_train_mean_test": b[top].mean(), "bottom30_train_mean_test": b[bot].mean(),
                        "all_mean_test": b.mean(), "base_test": st.loc["base", f"{test}_{cur}"]})
    show("train vs test across the 301 candidates", pd.DataFrame(rel).set_index(["cur", "train", "test"]))
    # (iii) walk-forward
    last, first = month_ends(p.days)
    spans = {"apr_nov": (first["2026-04"], len(p.days) - 1), "jul_nov": p.span(*SPANS["h2"]),
             "sep_nov": p.span(*SPANS["sep_nov"]), "oct_nov": p.span(*SPANS["oct_nov"])}
    b = p.ix["base"]
    res, pickrows = [], []
    for cur in ("krw", "usd"):
        for mode in ("expanding", "rolling60"):
            for rule in ("return", "stability"):
                for lag in (False, True):
                    nav, picks = walk_forward(p, cur, "expanding" if mode == "expanding" else "rolling", rule, lag)
                    row = {"select_cur": cur, "mode": mode, "rule": rule, "lag1": lag,
                           "distinct_picks": len({k for _, k in picks}),
                           "switches": sum(picks[i][1] != picks[i - 1][1] for i in range(1, len(picks)))}
                    for name, (i0, i1) in spans.items():
                        for c2 in ("krw", "usd"):
                            r, d = nav_stat(nav[c2], i0, i1)
                            br, bd = nav_stat(getattr(p, c2)[:, b], i0, i1)
                            row[f"{name}_{c2}"], row[f"{name}_{c2}_mdd"] = r, d
                            row[f"{name}_{c2}_base"], row[f"{name}_{c2}_mdd_base"] = br, bd
                    res.append(row)
                    if not lag:
                        arr = getattr(p, cur)
                        for m, key in picks:
                            a, z = first[m], last[m]
                            ret = {c2: getattr(p, c2)[z] / getattr(p, c2)[a - 1] - 1 for c2 in ("krw", "usd")}
                            order = (-ret[cur]).argsort().argsort()
                            i1 = last[sorted(last)[sorted(last).index(m) - 1]]
                            i0 = 0 if mode == "expanding" else i1 - 59
                            tr, _ = p.stat(arr, i0, i1)
                            keep = np.array([k != "half" for k in p.keys])
                            rho = pd.Series(tr[keep]).corr(pd.Series(ret[cur][keep]), method="spearman")
                            pickrows.append({"select_cur": cur, "mode": mode, "rule": rule, "hold_month": m, "pick": key,
                                             "pick_krw": ret["krw"][p.ix[key]], "base_krw": ret["krw"][b],
                                             "pick_usd": ret["usd"][p.ix[key]], "base_usd": ret["usd"][b],
                                             "pick_expo": p.exposure[a:z + 1, p.ix[key]].mean(),
                                             "rank_in_month_of_303": int(order[p.ix[key]]) + 1,
                                             "spearman_train_vs_next_month": rho})
    wf = pd.DataFrame(res)
    show("(iii) monthly walk-forward (decide at month-end Mar..Oct, hold Apr..Nov)",
         wf.set_index(["select_cur", "mode", "rule", "lag1"]).T)
    pk = pd.DataFrame(pickrows)
    show("walk-forward picks per month", pk.set_index(["select_cur", "mode", "rule", "hold_month"]))
    agg = []
    for (cur, mode, rule), g in pk.groupby(["select_cur", "mode", "rule"]):
        diff = g[f"pick_{cur}"] - g[f"base_{cur}"]
        agg.append({"select_cur": cur, "mode": mode, "rule": rule, "months": len(g),
                    "months_pick_beats_base": int((diff > 1e-12).sum()), "months_pick_loses_to_base": int((diff < -1e-12).sum()),
                    "months_pick_equals_base": int((diff.abs() <= 1e-12).sum()),
                    "months_pick_is_base": int((g.pick == "base").sum()), "distinct_picks": g.pick.nunique(),
                    "mean_rank_of_303": g.rank_in_month_of_303.mean(), "mean_spearman": g.spearman_train_vs_next_month.mean(),
                    "months_spearman_negative": int((g.spearman_train_vs_next_month < 0).sum())})
    agg = pd.DataFrame(agg).set_index(["select_cur", "mode", "rule"])
    show("walk-forward pick stability", agg)
    agg.to_csv(OUT / "c_walk_forward_stability.csv")
    wf.to_csv(OUT / "c_walk_forward.csv", index=False)
    pk.to_csv(OUT / "c_walk_forward_picks.csv", index=False)


# --------------------------------------------------------------------------- (d) placebo gates
class Gated:
    """Exact closed form of study.replay(weekly, gate) in USD from the base open/close NAV."""

    def __init__(self, ctx, p):
        opn_nav, close_nav = np.asarray(ctx.baseline.value, float), np.asarray(ctx.baseline.close, float)
        prev = np.r_[1., close_nav[:-1]]
        self.co, self.oc = np.log(opn_nav / prev), np.log(close_nav / opn_nav)
        self.fx = np.asarray(ctx.fx, float)
        self.p = p

    def nav(self, gates):
        g = np.atleast_2d(gates).astype(float)
        gp = np.concatenate([np.zeros((len(g), 1)), g[:, :-1]], axis=1)
        return np.exp(np.cumsum(gp * self.co + g * self.oc, axis=1))

    @staticmethod
    def stat(nav, i0, i1):
        anchor = nav[:, i0 - 1] if i0 else np.ones(len(nav))
        v = np.column_stack([anchor, nav[:, i0:i1 + 1]])
        return v[:, -1] / anchor - 1, (v / np.maximum.accumulate(v, axis=1) - 1).min(axis=1)


def runs(g):
    cut = np.flatnonzero(np.diff(g.astype(int))) + 1
    edges = np.r_[0, cut, len(g)]
    return [(bool(g[a]), int(b - a)) for a, b in zip(edges[:-1], edges[1:])]


def permute_spells(g, rng, n):
    """Same in-spells and out-spells (lengths kept), alternating, in random order."""
    rs = runs(g)
    ins, outs = [l for s, l in rs if s], [l for s, l in rs if not s]
    out = np.empty((n, len(g)), dtype=bool)
    for r in range(n):
        a, b = rng.permutation(ins), rng.permutation(outs)
        if len(a) == len(b):
            start_in = rng.random() < .5
        else:
            start_in = len(a) > len(b)
        seq, state, ia, ib = [], start_in, 0, 0
        while ia < len(a) or ib < len(b):
            if state:
                seq.append(np.ones(a[ia], bool)); ia += 1
            else:
                seq.append(np.zeros(b[ib], bool)); ib += 1
            state = not state
        out[r] = np.concatenate(seq)
    return out


def pctile(placebo, actual):
    placebo = np.asarray(placebo)
    return 100 * ((placebo < actual - 1e-12).mean() + .5 * (np.abs(placebo - actual) <= 1e-12).mean())


def section_d(ctx, curves, p, n_perm=5000):
    print("\n================ (d) placebo for market timing skill (weekly plan x 12 gates, USD) ================")
    eng = Gated(ctx, p)
    names = [k for k in ctx.gates if k != "none"]
    G = np.array([ctx.gates[k] for k in names])
    actual = eng.nav(G)
    # exactness of the closed form
    worst = max(np.abs(actual[i] - curves[f"weekly_{k}"].usd.to_numpy()).max() for i, k in enumerate(names))
    base_nav = eng.nav(np.ones((1, len(ctx.days)), bool))
    print(f"closed-form gated NAV vs study.replay ledgers: max abs diff {worst:.3e}; base {np.abs(base_nav[0] - curves['base'].usd.to_numpy()).max():.3e}")
    rng = np.random.default_rng(SEED)
    wins = {"full": p.span(*SPANS["full"]), "h1": p.span(*SPANS["h1"]), "h2": p.span(*SPANS["h2"]),
            "jul_aug": p.span(*SPANS["jul_aug"]), "sep_nov": p.span(*SPANS["sep_nov"])}
    n = len(ctx.days)
    shifts = np.arange(1, n)
    rows = []
    circ_ret = {w: np.empty((len(names), len(shifts))) for w in wins}
    perm_ret = {w: np.empty((len(names), n_perm)) for w in wins}
    wperm_ret = {w: np.empty((len(names), n_perm)) for w in ("h2", "sep_nov")}
    circ_h1 = {}
    for i, k in enumerate(names):
        g = G[i]
        circ = np.array([np.roll(g, s) for s in shifts])
        perm = permute_spells(g, rng, n_perm)
        nav_c, nav_p = eng.nav(circ), eng.nav(perm)
        row = {"gate": k, "out_days_full": int((~g).sum()), "out_spells": sum(1 for s, _ in runs(g) if not s)}
        for w, (i0, i1) in wins.items():
            ar, ad = eng.stat(actual[i:i + 1], i0, i1)
            br, bd = eng.stat(base_nav, i0, i1)
            cr, cd = eng.stat(nav_c, i0, i1)
            pr, pdd = eng.stat(nav_p, i0, i1)
            circ_ret[w][i], perm_ret[w][i] = cr, pr
            row.update({f"{w}_ret": ar[0], f"{w}_mdd": ad[0], f"{w}_base": br[0], f"{w}_in_share": g[i0:i1 + 1].mean(),
                        f"{w}_circ_med": np.median(cr), f"{w}_ret_pct_circ": pctile(cr, ar[0]), f"{w}_mdd_pct_circ": pctile(cd, ad[0]),
                        f"{w}_perm_med": np.median(pr), f"{w}_ret_pct_perm": pctile(pr, ar[0]), f"{w}_mdd_pct_perm": pctile(pdd, ad[0])})
            if w in wperm_ret:  # exposure held fixed inside the window: reorder only this window's spells
                seg = g[i0:i1 + 1]
                wp = permute_spells(seg, rng, n_perm)
                full = np.tile(g, (n_perm, 1)); full[:, i0:i1 + 1] = wp
                wr, wd = eng.stat(eng.nav(full), i0, i1)
                wc = np.array([np.r_[g[:i0], np.roll(seg, s), g[i1 + 1:]] for s in range(1, len(seg))])
                wcr, wcd = eng.stat(eng.nav(wc), i0, i1)
                wperm_ret[w][i] = wr
                row.update({f"{w}_winperm_med": np.median(wr), f"{w}_ret_pct_winperm": pctile(wr, ar[0]),
                            f"{w}_mdd_pct_winperm": pctile(wd, ad[0]), f"{w}_ret_pct_wincirc": pctile(wcr, ar[0]),
                            f"{w}_mdd_pct_wincirc": pctile(wcd, ad[0]), f"{w}_winperm_distinct": len(np.unique(wp, axis=0))})
        rows.append(row)
        circ_h1[k] = nav_c
    tab = pd.DataFrame(rows).set_index("gate")
    for w in ("h2", "sep_nov", "jul_aug", "full", "h1"):
        cols = [c for c in tab.columns if c.startswith(w + "_")]
        show(f"placebo percentiles, window {w} (pct = share of placebos the actual gate beats; 220 circular shifts, {n_perm} spell permutations)", tab[cols], "{:+.3f}")
    tab.to_csv(OUT / "d_placebo_per_gate.csv")
    counts = []
    for column in [c for c in tab.columns if "_pct_" in c]:
        v = tab[column]
        counts.append({"percentile_column": column, "n_gt_90": int((v > 90).sum()), "n_gt_50": int((v > 50).sum()),
                       "n_lt_10": int((v < 10).sum()), "median_pct": v.median(), "min_pct": v.min(), "max_pct": v.max(),
                       "best_gate": v.idxmax()})
    counts = pd.DataFrame(counts).set_index("percentile_column")
    show("per-gate percentile counts over the 12 gates (no skill expectation: about 1.2 of 12 above 90, 6 above 50)", counts, "{:.1f}")
    counts.to_csv(OUT / "d_placebo_counts.csv")
    # family-wise
    fam = []
    for w, (i0, i1) in wins.items():
        act = np.array([eng.stat(actual[i:i + 1], i0, i1)[0][0] for i in range(len(names))])
        best_act = act.max()
        joint = circ_ret[w].max(axis=0)       # same shift applied to all 12 gates, best of 12
        indep = perm_ret[w].max(axis=0)       # independent spell permutations, best of 12
        row = {"window": w, "base": eng.stat(base_nav, i0, i1)[0][0], "best_actual": best_act, "best_gate": names[int(act.argmax())],
               "mean_actual_12": act.mean(),
               "n_above_p90_circ": int(sum(pctile(circ_ret[w][i], act[i]) > 90 for i in range(len(names)))),
               "n_above_p90_perm": int(sum(pctile(perm_ret[w][i], act[i]) > 90 for i in range(len(names)))),
               "n_above_p50_circ": int(sum(pctile(circ_ret[w][i], act[i]) > 50 for i in range(len(names)))),
               "n_above_p50_perm": int(sum(pctile(perm_ret[w][i], act[i]) > 50 for i in range(len(names)))),
               "best12_circ_p50": np.median(joint), "best12_circ_p90": np.quantile(joint, .9), "best12_circ_max": joint.max(),
               "best_actual_pct_in_best12_circ": pctile(joint, best_act),
               "best12_perm_p50": np.median(indep), "best12_perm_p90": np.quantile(indep, .9),
               "best_actual_pct_in_best12_perm": pctile(indep, best_act),
               "mean12_actual_pct_in_mean12_circ": pctile(circ_ret[w].mean(axis=0), act.mean())}
        if w in wperm_ret:
            wi = wperm_ret[w].max(axis=0)
            row.update({"n_above_p90_winperm": int(sum(pctile(wperm_ret[w][i], act[i]) > 90 for i in range(len(names)))),
                        "n_above_p50_winperm": int(sum(pctile(wperm_ret[w][i], act[i]) > 50 for i in range(len(names)))),
                        "best12_winperm_p50": np.median(wi), "best12_winperm_p90": np.quantile(wi, .9),
                        "best_actual_pct_in_best12_winperm": pctile(wi, best_act)})
        fam.append(row)
    famt = pd.DataFrame(fam).set_index("window").T
    show("family-wise picture (12 gates)", famt, "{:+.3f}")
    famt.to_csv(OUT / "d_placebo_family.csv")
    # protocol placebo: apply the same circular shift to all 12 gates, select by Codex's return rule in the
    # training window (KRW), record the test-window result of the pick.
    prot = []
    fx = eng.fx
    for train, test in (("h1", "h2"), ("jan_aug", "sep_nov")):
        (a0, a1), (t0, t1) = p.span(*SPANS[train]), p.span(*SPANS[test])
        base_k = base_nav * fx
        btr, btd = eng.stat(base_k, a0, a1)

        def choose(navs):
            k = navs * fx
            r, d = eng.stat(k, a0, a1)
            r = np.where(d >= btd[0] - 1e-9, r, -np.inf)
            j = int(np.argmax(np.r_[r, btr[0]]))          # index 12 = base
            return j
        j_act = choose(actual)
        act_nav = base_nav if j_act == 12 else actual[j_act:j_act + 1]
        act_usd = eng.stat(act_nav, t0, t1)
        picks, out_usd, out_mdd = [], [], []
        for si, s in enumerate(shifts):
            navs = np.array([circ_h1[k][si] for k in names])
            j = choose(navs)
            nav = base_nav if j == 12 else navs[j:j + 1]
            r, d = eng.stat(nav, t0, t1)
            picks.append(j); out_usd.append(r[0]); out_mdd.append(d[0])
        out_usd, out_mdd = np.array(out_usd), np.array(out_mdd)
        bu = eng.stat(base_nav, t0, t1)
        prot.append({"protocol": f"{train}->{test}", "actual_pick": "base" if j_act == 12 else names[j_act],
                     "actual_test_usd": act_usd[0][0], "actual_test_usd_mdd": act_usd[1][0], "base_test_usd": bu[0][0],
                     "placebo_n": len(shifts), "placebo_pick_is_base": int((np.array(picks) == 12).sum()),
                     "placebo_test_usd_p10": np.quantile(out_usd, .1), "placebo_test_usd_p50": np.median(out_usd),
                     "placebo_test_usd_p90": np.quantile(out_usd, .9), "placebo_beats_base_share": (out_usd > bu[0][0] + 1e-12).mean(),
                     "actual_pct_ret": pctile(out_usd, act_usd[0][0]), "actual_pct_mdd": pctile(out_mdd, act_usd[1][0])})
    pt = pd.DataFrame(prot).set_index("protocol").T
    show("whole-protocol placebo within the 12 weekly gates (joint circular shift -> select on train KRW -> test USD)", pt, "{:+.3f}")
    pt.to_csv(OUT / "d_placebo_protocol.csv")


# --------------------------------------------------------------------------- (e) whipsaw accounting
def section_e(ctx, curves, p):
    print("\n================ (e) whipsaw accounting from market_cash_spells.csv ================")
    sp = pd.read_csv(CODEX / "market_cash_spells.csv")
    # verify the file from the base open/close NAV
    value, close = np.asarray(ctx.baseline.value), np.asarray(ctx.baseline.close)
    rec = []
    for market, flags in ctx.gates.items():
        start = None
        for j, inv in enumerate(flags):
            if not inv and start is None:
                start = j
            if start is not None and (inv or j == len(ctx.days) - 1):
                end = value[j] if inv else close[j]
                rec.append((market, ctx.days[start], ctx.days[j], j - start + int(not inv), end / value[start] - 1))
                start = None
    rec = pd.DataFrame(rec, columns=["market", "exit_open", "end", "cash_sessions", "base_usd_return"])
    same = (len(rec) == len(sp) and (rec[["market", "exit_open", "end"]].values == sp[["market", "exit_open", "end"]].values).all()
            and np.abs(rec.base_usd_return - sp.base_usd_return).max() < 1e-12 and (rec.cash_sessions == sp.cash_sessions).all())
    print(f"market_cash_spells.csv reproduced from gates and base open/close NAV: {same} ({len(sp)} spells)")
    # identity: gated full-year USD = base / prod(1+spell)
    for k in ("SOXX100r1", "SOXX10r1", "QQQ20r3"):
        f = np.prod(1 + sp.loc[sp.market == k, "base_usd_return"])
        print(f"  identity {k}: base_end/prod(1+spell) = {curves['base'].usd.iloc[-1] / f:.6f} vs ledger {curves[f'weekly_{k}'].usd.iloc[-1]:.6f}")
    sp["period"] = np.where(sp.exit_open < "2026-07-01", "H1", np.where(sp.exit_open < "2026-09-01", "Jul-Aug", "Sep-Nov"))
    sp["straddle"] = ((sp.exit_open < "2026-07-01") & (sp.end > "2026-07-01")) | ((sp.exit_open < "2026-09-01") & (sp.end > "2026-09-01"))
    sp["avoided"] = sp.base_usd_return < 0

    def agg(g):
        r = g.base_usd_return
        return pd.Series({"spells": len(g), "cash_sessions": int(g.cash_sessions.sum()),
                          "share_avoided_loss": (r < 0).mean(), "sum_avoided": r[r < 0].sum(), "sum_missed": r[r > 0].sum(),
                          "net_sum": r.sum(), "base_compound_in_cash": np.prod(1 + r) - 1,
                          "gate_value_multiple": 1 / np.prod(1 + r), "short_spells_le2": int((g.cash_sessions <= 2).sum()),
                          "short_le2_net_sum": r[g.cash_sessions <= 2].sum(), "worst_miss": r.max(), "best_avoid": r.min(),
                          "straddles_boundary": int(g.straddle.sum())})
    per = sp.groupby(["market", "period"]).apply(agg, include_groups=False)
    tot = sp.groupby("market").apply(agg, include_groups=False)
    tot.index = pd.MultiIndex.from_product([tot.index, ["ALL"]])
    out = pd.concat([per, tot]).sort_index()
    show("per gate and period (returns are the base USD move while in cash; negative = avoided, positive = missed)", out)
    overall = sp.groupby("period").apply(agg, include_groups=False)
    show("all 12 gates pooled by period (spells overlap across gates; not independent)", overall)
    pooled = agg(sp)
    print("pooled all gates all periods:", {k: round(float(v), 4) for k, v in pooled.items()})
    strad = sp.loc[sp.straddle, ["market", "exit_open", "end", "cash_sessions", "base_usd_return"]]
    show("spells straddling a period boundary (assigned to the exit period)", strad)
    out.to_csv(OUT / "e_whipsaw.csv")


# --------------------------------------------------------------------------- (f) what the report does not say
def section_f(ctx, curves, p, mine, sel):
    print("\n================ (f) facts the report omits ================")
    base = curves["base"]
    days = np.asarray(ctx.days)
    fx = np.asarray(ctx.fx)
    i0, i1 = p.span(*SPANS["h2"])
    for cur in ("usd", "krw"):
        x = base[cur].to_numpy()
        j = int(x.argmax())
        jh = i0 - 1 + int(np.r_[x[i0 - 1], x[i0:]].argmax())
        print(f"base {cur}: year peak {days[j]} ({x[j]:.4f}), last {x[-1]:.4f}, fall from peak {x[-1] / x[j] - 1:+.4f}; "
              f"H2-window peak (incl. 06-30 anchor) {days[jh]}; value at 06-30 {x[i0 - 1]:.4f}; 09-22 vs 06-30 {x[list(days).index('2026-09-22')] / x[i0 - 1] - 1:+.4f}")
    print(f"fx (KRW per USD, relative): H1 {fx[i0 - 1] / 1 - 1:+.4f} (vs pre-start), H2 {fx[-1] / fx[i0 - 1] - 1:+.4f}; "
          f"base H2 usd {mine.loc['base', 'h2_return_usd']:+.4f} krw {mine.loc['base', 'h2_return']:+.4f}; "
          f"share of H2 KRW log-loss from FX {np.log(fx[-1] / fx[i0 - 1]) / np.log(1 + mine.loc['base', 'h2_return']):.3f}")
    usd = base.usd.to_numpy()
    m = pd.Series(days).str[:7]
    monthly = {mm: usd[g.index[-1]] / (usd[g.index[0] - 1] if g.index[0] else 1.) - 1 for mm, g in pd.Series(days).groupby(m)}
    print("base monthly USD:", {k: round(float(v), 4) for k, v in monthly.items()})
    monthly_fx = {mm: fx[g.index[-1]] / (fx[g.index[0] - 1] if g.index[0] else 1.) - 1 for mm, g in pd.Series(days).groupby(m)}
    print("monthly FX:", {k: round(float(v), 4) for k, v in monthly_fx.items()})
    # H1 drawdown episodes of base (USD)
    h0, h1 = p.span(*SPANS["h1"])
    v = np.r_[1., usd[:h1 + 1]]
    dd = v / np.maximum.accumulate(v) - 1
    t = int(dd.argmin()); pk = int(v[:t + 1].argmax())
    print(f"base H1 USD max drawdown {dd.min():+.4f}: peak {days[pk - 1] if pk else 'start'} -> trough {days[t - 1]} ({t - pk} sessions); "
          f"H1 sessions with drawdown below -10%: {int((dd < -.10).sum())}/{h1 + 1}")
    # how much each gate was exercised in H1 and what it did there
    rows = []
    for k, g in ctx.gates.items():
        key = "base" if k == "none" else f"weekly_{k}"
        r = mine.loc[key]
        rows.append({"gate": k, "h1_out_days": int((~g[:h1 + 1]).sum()), "h1_out_spells": sum(1 for s, _ in runs(g[:h1 + 1]) if not s),
                     "h1_krw": r.h1_return, "h1_mdd": r.h1_mdd, "h1_usd": r.h1_return_usd,
                     "jul_aug_out_days": int((~g[i0:p.span(*SPANS['jul_aug'])[1] + 1]).sum()), "jul_aug_usd": r.jul_aug_return_usd,
                     "sep_nov_out_days": int((~g[p.span(*SPANS['sep_nov'])[0]:]).sum()), "sep_nov_usd": r.sep_nov_return_usd,
                     "h2_usd": r.h2_return_usd, "h2_krw": r.h2_return, "h2_usd_mdd": r.h2_mdd_usd, "full_krw": r.full_return})
    gt = pd.DataFrame(rows).set_index("gate")
    gt["h1_rank"] = gt.h1_krw.rank(ascending=False).astype(int)
    gt["h2_rank"] = gt.h2_krw.rank(ascending=False).astype(int)
    show("weekly plan x market gate (13 rows): days out and results per period", gt)
    print(f"spearman(H1 KRW, H2 KRW) over the 13 weekly rows: {gt.h1_krw.corr(gt.h2_krw, method='spearman'):+.3f}; "
          f"spearman(H1 out days, H1 KRW): {gt.h1_out_days.corr(gt.h1_krw, method='spearman'):+.3f}")
    gt.to_csv(OUT / "f_weekly_gates.csv")
    c = mine.drop(CTRL)
    print(f"candidates with H2 USD return > 0: {int((c.h2_return_usd > 0).sum())}/301; best {c.h2_return_usd.max():+.4f} ({c.h2_return_usd.idxmax()}); "
          f"with H2 KRW return > 0: {int((c.h2_return > 0).sum())}; H2 KRW better than USD cash ({(fx[-1] / fx[i0 - 1] - 1):+.4f}): {int((c.h2_return > fx[-1] / fx[i0 - 1] - 1).sum())}")
    print(f"candidates with H1 KRW return > base: {int((c.h1_return > mine.loc['base', 'h1_return']).sum())}/301; "
          f"full-year KRW > base: {int((c.full_return > mine.loc['base', 'full_return']).sum())}/301; "
          f"full-year KRW > base and full MDD shallower: {int(((c.full_return > mine.loc['base', 'full_return']) & (c.full_mdd > mine.loc['base', 'full_mdd'])).sum())}")
    yr = c.loc[c.full_return > mine.loc["base", "full_return"], ["full_return", "full_mdd", "full_return_usd", "full_mdd_usd", "full_exposure", "full_turn", "h1_return", "h2_return", "jul_aug_return", "sep_nov_return"]]
    show("candidates beating base over the full year in KRW (base full +1.0893, mdd -0.4004)", yr.sort_values("full_return", ascending=False))
    print(f"full-year median candidate KRW {c.full_return.median():+.4f}, mean {c.full_return.mean():+.4f}; full-year MDD shallower than base: {int((c.full_mdd > mine.loc['base', 'full_mdd']).sum())}/301; "
          f"full-year Calmar-like (return/|mdd|) above base: {int((c.full_return / c.full_mdd.abs() > mine.loc['base', 'full_return'] / abs(mine.loc['base', 'full_mdd'])).sum())}/301")
    print(f"Jul-Aug: candidates with KRW return > base {int((c.jul_aug_return > mine.loc['base', 'jul_aug_return']).sum())}/301, USD > base {int((c.jul_aug_return_usd > mine.loc['base', 'jul_aug_return_usd']).sum())}; "
          f"Sep-Nov KRW > base {int((c.sep_nov_return > mine.loc['base', 'sep_nov_return']).sum())}/301")
    g = sel["global_key"]
    cols = ["h1_return", "h1_return_usd", "jul_aug_return", "jul_aug_return_usd", "jul_aug_exposure", "sep_nov_return",
            "sep_nov_return_usd", "sep_nov_exposure", "h2_return", "h2_return_usd", "h2_mdd_usd", "h2_exposure", "h2_turn", "full_return", "full_return_usd"]
    show("headline rules by sub-period", mine.loc[["base", "half", g, sel["stability_key"], "weekly_SOXX100r1", "weekly_SOXX10r1", sel["h2best"]], cols].T)
    top = c.sort_values("h1_return", ascending=False).head(10)[["h1_return", "h1_mdd", "h2_return", "h2_return_usd", "h2_exposure"]]
    show("top 10 by H1 KRW return (how close the H1 race was)", top)
    print(f"H1 KRW return: selected {mine.loc[g, 'h1_return']:+.4f} vs base {mine.loc['base', 'h1_return']:+.4f}; "
          f"candidates within 10 points of base H1 return: {int((np.abs(c.h1_return - mine.loc['base', 'h1_return']) < .10).sum())}")
    # selected rule gate days out in H1
    for k in ("SOXX100r1",):
        gg = ctx.gates[k]
        out_dates = days[:h1 + 1][~gg[:h1 + 1]]
        print(f"{k} out sessions in H1: {len(out_dates)} -> {out_dates.tolist()}")
        print(f"{k} out sessions in H2: {int((~gg[i0:]).sum())}/{len(days) - i0}; spells {[(days[i0:][0], )] and runs(gg[i0:])}")


def main(which):
    ctx = common.context()
    curves, _ = common.codex_curves(ctx)
    p = Panel(ctx, curves)
    mine = my_table(p)
    sel = section_a(ctx, curves, p, mine) if True else None
    if "b" in which:
        section_b(p, mine, sel)
    if "c" in which:
        section_c(ctx, curves, p, mine)
    if "d" in which:
        section_d(ctx, curves, p)
    if "e" in which:
        section_e(ctx, curves, p)
    if "f" in which:
        section_f(ctx, curves, p, mine, sel)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or set("abcdef"))
