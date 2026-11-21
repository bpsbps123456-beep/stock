"""Check the desk's 7-stock engine against Codex's #774 paths and its own frozen evidence.

- Engine check, with #774's own caps (config.CODEX_774_RULE, semiconductor budget 60%) and the
  sector labels it was tested with (Codex sector_map.json, frozen 2026-09-27):
  replayed with Codex's timing ("publication lag 1": a PDF is usable one US session after
  the first session on/after its date), every weekly target must equal Codex's #774
  projected targets, stock by stock, 2026-01-02..09-23. The desk itself uses any PDF
  dated before the session, which is one PDF newer after US holidays; those days are
  listed, not failed.
- The frozen pause-rule numbers in alerts.BACKTEST_2026 were measured on #774 (60%) and
  must still describe that path through 2026-09-25; an engine change that moves them fails.
- The live rule (config.RULE, semiconductor budget 50% since 2026-10-10) is then reported
  with the same measures for comparison; it does not fail the check.
- The #774 replay needs Codex's representative_paths.npz. Without it the check FAILS (a clone or
  cloud session would otherwise pass §6 unchecked); `py -m desk validate --skip-774` skips it on
  purpose and the result line then says 미검증.
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


SWITCH_LEDGER = config.ROOT / "data/derived/analysis/desk_etf_validation_20261121/results/paired_daily.csv"


def switch_pause_evidence() -> bool:
    """§5 전환 경로 기준값(alerts.BACKTEST_SWITCH_2026)이 근거 원장에서 그대로 나오는지(0.5%p 안)."""
    import pandas as pd
    if not SWITCH_LEDGER.exists():
        print(f"pause rule (§3.1 switch): FAIL — 근거 원장이 없습니다 ({SWITCH_LEDGER})")
        return False
    p = pd.read_csv(SWITCH_LEDGER)
    sw, etf = p["switch"].to_numpy(float), p["time"].to_numpy(float)
    mdd = float((sw / np.maximum.accumulate(sw) - 1).min())
    gap20 = float(((sw[20:] / sw[:-20] - 1) - (etf[20:] / etf[:-20] - 1)).min())
    want = alerts.BACKTEST_SWITCH_2026
    same = abs(mdd - want["mdd"]) < 0.005 and abs(gap20 - want["gap20"]) < 0.005
    print(f"pause rule (§3.1 switch, KRW 10bp): worst drawdown {mdd:.1%} (frozen {want['mdd']:.1%}), worst 20-session gap "
          f"{gap20:+.1%} (frozen {want['gap20']:+.1%}) {'ok' if same else 'CHANGED'}")
    return same


def codex_sectors(book) -> dict[str, str]:
    """The watchlist sectors #774 was tested with (Codex sector_map.json, 2026-09-27).
    watchlist.py keeps growing (e.g. TIME's unclassified names classified on 2026-10-10), so
    the engine check replays with the frozen labels; names Codex never saw keep today's."""
    path = CODEX / "sector_map.json"
    if not path.exists():
        return dict(book.sector)
    frozen = {r["company_key"]: r["sector"] for r in json.loads(path.read_text(encoding="utf-8"))["records"]}
    return {k: frozen.get(k, s) for k, s in book.sector.items()}


class codex_caps:
    """Temporarily run the engine as #774: its caps (config.RULE) and, given a book, the sector
    labels it was tested with. Both are restored afterwards, even on errors."""

    def __init__(self, book=None):
        self.book = book

    def __enter__(self):
        self.saved = dict(config.RULE)
        config.RULE.clear()
        config.RULE.update(config.CODEX_774_RULE)
        if self.book is not None:
            self.saved_sector = self.book.sector
            self.book.sector = codex_sectors(self.book)

    def __exit__(self, *exc):
        config.RULE.clear()
        config.RULE.update(self.saved)
        if self.book is not None:
            self.book.sector = self.saved_sector


NPZ = CODEX / "representative_paths.npz"


def check_774(book, path, skip: bool = False) -> tuple[bool, str]:
    """#774 target-weight replay. Returns (ok, status) with status "ok" | "FAIL" | "미검증"."""
    if skip:
        print("7종목 #774 재현: 미검증 (--skip-774로 건너뜀)")
        return True, "미검증"
    if not NPZ.exists():
        print(f"7종목 #774 재현: FAIL — 입력 {NPZ.relative_to(config.ROOT).as_posix()} 이(가) 없습니다. "
              "원본 연구 폴더에서 받아 오거나, 의도적으로 건너뛰려면 --skip-774를 붙이세요(결과에 '미검증' 표시).")
        return False, "FAIL"
    expected, codex_total = codex_targets()
    original = S.latest_before
    S.latest_before = codex_timing(book)
    try:
        with codex_caps(book):
            replay = _weekly(book, S.run_model(book))
    finally:
        S.latest_before = original
    days = sorted(set(expected) | set(replay))
    bad = [(d, expected.get(d, {}), replay.get(d, {})) for d in days
           if _diff(expected.get(d, {}), replay.get(d, {})) > 1e-6]
    print(f"7종목 (Codex timing): {len(days) - len(bad)}/{len(days)} rebalance days match #774 target weights")
    for day, want, got in bad[:10]:
        print(f"  {day}: codex {_fmt(want)} | desk {_fmt(got)}")
    desk = _weekly(book, path)
    newer = [d for d in sorted(desk) if _diff(desk[d], expected.get(d, {})) > 1e-6]
    print(f"desk timing: {len(desk) - len(newer)}/{len(desk)} identical; newer PDF after US holidays on "
          + (", ".join(newer) or "none"))
    model = S.model_index(book, path)
    print(f"model #774 caps 2026-01-02..{CODEX_END}: KRW {model[path.days.index(CODEX_END)] / 100 - 1:+.1%} "
          f"(Codex {codex_total:+.1%}; FX timing and cash currency differ)")
    return not bad, "ok" if not bad else "FAIL"


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    unknown = [a for a in args if a != "--skip-774"]
    if unknown:
        print(f"알 수 없는 옵션: {' '.join(unknown)} (사용: py -m desk validate [--skip-774])")
        return 2
    from .state import last_completed_session
    book = S.load_book(through=last_completed_session())
    with codex_caps(book):
        path = S.run_model(book)
    live = S.run_model(book)
    ok, status = check_774(book, path, skip="--skip-774" in args)
    pause_ok = pause_evidence(book, path) & switch_pause_evidence()
    ok &= pause_ok
    lm = S.model_index(book, live)
    mdd, gap20 = pause_measures(book, live)
    print(f"live rule (semiconductors {config.RULE['semis_cap']:.0%}) 2026-01-02..{CODEX_END}: KRW "
          f"{lm[live.days.index(CODEX_END)] / 100 - 1:+.1%}, through {FROZEN_END}: worst drawdown {mdd:.1%}, "
          f"worst 20-session gap {gap20:+.1%}; through {live.days[-1]}: KRW {lm[-1] / 100 - 1:+.1%} (reported, not checked)")
    print(f"validate: {'ok' if ok else 'FAIL'} · #774 재현 {status} · 멈춤 근거 {'ok' if pause_ok else 'CHANGED'}")
    return 0 if ok else 1
