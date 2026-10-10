"""Check the desk's 7-stock engine against Codex's #774 paths and its own frozen evidence.

- Engine check, with #774's own caps (config.CODEX_774_RULE, semiconductor budget 60%):
  replayed with Codex's timing ("publication lag 1": a PDF is usable one US session after
  the first session on/after its date), every weekly target must equal Codex's #774
  projected targets, stock by stock, 2026-01-02..09-23. The desk itself uses any PDF
  dated before the session, which is one PDF newer after US holidays; those days are
  listed, not failed.
- The frozen pause-rule numbers in alerts.BACKTEST_2026 were measured on #774 (60%) and
  must still describe that path through 2026-09-25; an engine change that moves them fails.
- The live rule (config.RULE, semiconductor budget 50% since 2026-10-10) is then reported
  with the same measures for comparison; it does not fail the check.
"""

from __future__ import annotations

import json
from bisect import bisect_left

import numpy as np

from . import alerts, config, strategies as S

CODEX = (config.ROOT / "data" / "derived" / "analysis" / "2026-09-23_strategy_research"
         / "concentration_research")
PREFIX = "candidate774_publication_lag1_full"
CODEX_END = "2026-09-23"
FROZEN_END = "2026-09-25"


def codex_targets() -> tuple[dict[str, dict[str, float]], float]:
    z = np.load(CODEX / "representative_paths.npz")  # plain arrays only (numpy's safe default)
    tickers = json.loads((CODEX / "sector_map.json").read_text(encoding="utf-8"))["ordered_tickers"]
    dates = json.loads((CODEX / "neighbor_profiles.json").read_text(encoding="utf-8"))["dates"]
    events, targets = z[f"{PREFIX}_events"], z[f"{PREFIX}_projected_targets"]
    out = {day: {tickers[j]: float(targets[t, j]) for j in np.flatnonzero(targets[t] > 1e-12)}
           for t, day in enumerate(dates) if events[t]}
    return out, float(np.prod(1 + z[f"{PREFIX}_returns"]) - 1)


def codex_timing(book):
    usable_from = {p: bisect_left(book.days, p) + 1 for p in book.pdfs}

    def latest(dates, day):
        k = bisect_left(book.days, day)
        usable = [p for p in dates if usable_from.get(p, len(book.days) + 1) <= k]
        return usable[-1] if usable else None
    return latest


def _weekly(book, path) -> dict[str, dict[str, float]]:
    return {day: {book.ticker[k]: v for k, v in t.items()} for day, t in path.rebalances.items() if day <= CODEX_END}


def _diff(a: dict, b: dict) -> float:
    return max((abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in set(a) | set(b)), default=0.0)


def _fmt(weights: dict[str, float]) -> str:
    return ", ".join(f"{t} {v:.1%}" for t, v in sorted(weights.items(), key=lambda x: -x[1])) or "현금"


def pause_measures(book, path) -> tuple[float, float]:
    end = path.days.index(FROZEN_END) + 1
    m = np.concatenate([[100.0], S.model_index(book, path)[:end]])
    e = np.concatenate([[np.nan], S.etf_index(book, path.days)[:end]])
    mdd = float((m / np.maximum.accumulate(m) - 1).min())
    gap20 = float(np.nanmin(m[21:] / m[1:-20] - e[21:] / e[1:-20]))
    return mdd, gap20


def pause_evidence(book, path) -> bool:
    mdd, gap20 = pause_measures(book, path)
    want = alerts.BACKTEST_2026
    same = abs(mdd - want["mdd"]) < 0.005 and abs(gap20 - want["gap20"]) < 0.005
    print(f"pause rule (#774 caps): worst drawdown {mdd:.1%} (frozen {want['mdd']:.1%}), worst 20-session gap "
          f"{gap20:+.1%} (frozen {want['gap20']:+.1%}) {'ok' if same else 'CHANGED'}")
    return same


class codex_caps:
    """Temporarily run the engine with #774's caps (config.RULE is restored afterwards)."""

    def __enter__(self):
        self.saved = dict(config.RULE)
        config.RULE.clear()
        config.RULE.update(config.CODEX_774_RULE)

    def __exit__(self, *exc):
        config.RULE.clear()
        config.RULE.update(self.saved)


def main() -> int:
    book = S.load_book()
    with codex_caps():
        path = S.run_model(book)
    live = S.run_model(book)
    ok = True
    if (CODEX / "representative_paths.npz").exists():
        expected, codex_total = codex_targets()
        original = S.latest_before
        S.latest_before = codex_timing(book)
        try:
            with codex_caps():
                replay = _weekly(book, S.run_model(book))
        finally:
            S.latest_before = original
        days = sorted(set(expected) | set(replay))
        bad = [(d, expected.get(d, {}), replay.get(d, {})) for d in days
               if _diff(expected.get(d, {}), replay.get(d, {})) > 1e-6]
        print(f"7종목 (Codex timing): {len(days) - len(bad)}/{len(days)} rebalance days match #774 target weights")
        for day, want, got in bad[:10]:
            print(f"  {day}: codex {_fmt(want)} | desk {_fmt(got)}")
        ok &= not bad
        desk = _weekly(book, path)
        newer = [d for d in sorted(desk) if _diff(desk[d], expected.get(d, {})) > 1e-6]
        print(f"desk timing: {len(desk) - len(newer)}/{len(desk)} identical; newer PDF after US holidays on "
              + (", ".join(newer) or "none"))
        model = S.model_index(book, path)
        print(f"model #774 caps 2026-01-02..{CODEX_END}: KRW {model[path.days.index(CODEX_END)] / 100 - 1:+.1%} "
              f"(Codex {codex_total:+.1%}; FX timing and cash currency differ)")
    else:
        print("Codex #774 paths not found; skipped")
    ok &= pause_evidence(book, path)
    lm = S.model_index(book, live)
    mdd, gap20 = pause_measures(book, live)
    print(f"live rule (semiconductors {config.RULE['semis_cap']:.0%}) 2026-01-02..{CODEX_END}: KRW "
          f"{lm[live.days.index(CODEX_END)] / 100 - 1:+.1%}, through {FROZEN_END}: worst drawdown {mdd:.1%}, "
          f"worst 20-session gap {gap20:+.1%}; through {live.days[-1]}: KRW {lm[-1] / 100 - 1:+.1%} (reported, not checked)")
    return 0 if ok else 1
