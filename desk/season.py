"""시즌 계기판 (docs/desk_system.md §11.2, 2026-11-20). 관찰용: 목표 비중·주문 수량을 만들지 않는다.

사용자 결정(2026-11-20): "모멘텀에 맞게 살아 있는 생물처럼 대응"하되 규칙은 바꾸지 않고, 지금 시즌을 보여 주는
계기판과 다이얼 선택의 종이 기록부터 쌓는다. 근거 연구는 data/derived/analysis/season_regime_20261119/.

- 모멘텀 성적 ic_h[j]: 세션 j 개장 때 아는 120일 모멘텀 순위(7종목 규칙과 같은 후보)와 j-1 종가 → j-1+h 종가
  수익의 순위 상관. j-1+h 종가가 지나야 확정된다. 확정된 값만 최근 K개 평균한다(known_ic).
- SOXX 20일선: 직전 확정 종가가 그날 포함 20일 단순평균 이상이면 위.
- 바구니: MOM = 데스크 7종목 모델 경로(2026-11-21부터 60일 잔차 모멘텀 규칙), REV = TIME 미국 보유 중 직전 5거래일 수익 최하위 7종목 같은 비중(상한 적용),
  둘 다 ISO 주 첫 세션 개장에 교체. CASH = 달러.
- 성격 다이얼(h20·K20): 모멘텀 성적 ≥ 0(또는 미확정)이면 MOM, 아니면 REV.
- 두 다이얼(SOXX 20일선 · h10·K10): SOXX 20일선 아래면 CASH, 위면 성격 다이얼(h10·K10).
- 종이 기록: 마지막 확정 종가 다음 세션의 선택을 처음 계산한 때 한 줄로 잠근다(season_paper.json). 지난 줄은 고치지 않는다.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import config
from . import strategies as S
from .state import clean

# 테스트 서버(DESK_JOURNAL_DIR 지정)는 실제 종이 기록을 건드리지 않게 그 임시 폴더에 쓴다(§2).
PAPER = (config.JOURNAL_DIR if os.environ.get("DESK_JOURNAL_DIR") else config.CACHE) / "season_paper.json"
CACHE_SEC = 3600
DIALS = {"dial1": {"name": "성격 다이얼", "h": 20, "K": 20, "gate": False},
         "dial2": {"name": "두 다이얼", "h": 10, "K": 10, "gate": True}}
REV_N, REV_LOOKBACK, SMA = 7, 5, 20


class SeasonError(Exception):
    pass


# ---------------------------------------------------------------------------
# pure pieces (unit tested)
# ---------------------------------------------------------------------------

def known_ic(ic, horizon: int, window: int) -> np.ndarray:
    """At session t only ic[j] with j <= t - horizon has matured; trailing mean of the last `window`."""
    s = pd.Series(np.asarray(ic, dtype=float))
    return s.rolling(window, min_periods=max(3, window // 2)).mean().shift(horizon).to_numpy()


def choose(ic_known, soxx_up, gate: bool) -> str:
    """Dial pick for one session. Unmatured IC (NaN) counts as momentum working, as in the study."""
    if gate and soxx_up is False:
        return "CASH"
    works = not (ic_known is not None and np.isfinite(ic_known) and ic_known < 0)
    return "MOM" if works else "REV"


def chain(choices: list[str], opn: dict[str, np.ndarray], cls: dict[str, np.ndarray]) -> np.ndarray:
    """Close-marked value of holding choices[j] from open j to open j+1 (1.0 before the first open).

    opn/cls are each species' value at every session open/close (its own path, start 1.0)."""
    v, out = 1.0, []
    for j, c in enumerate(choices):
        if j > 0 and choices[j - 1] != "CASH":
            p = choices[j - 1]
            v *= opn[p][j] / opn[p][j - 1]
        out.append(v if c == "CASH" else v * cls[c][j] / opn[c][j])
    return np.array(out)


def append_record(ledger: dict, record: dict) -> bool:
    """Lock a session's record once. Existing sessions are never rewritten. Returns True if added."""
    have = {r["session"] for r in ledger.setdefault("records", [])}
    if record["session"] in have:
        return False
    ledger["records"].append(record)
    ledger["records"].sort(key=lambda r: r["session"])
    ledger["start"] = ledger["records"][0]["session"]
    return True


def session_after(day: str) -> str:
    d = date.fromisoformat(day) + timedelta(days=1)
    while not config.is_us_session(d):
        d += timedelta(days=1)
    return d.isoformat()


def _drawdown(v) -> float:
    v = np.asarray(v, dtype=float)
    return float((v / np.maximum.accumulate(v) - 1).min()) if len(v) else 0.0


def paper_score(records: list[dict], days: list[str], opn: dict, cls: dict) -> dict:
    """Realised paper values since the first locked session, using each locked pick.

    A session missing from the record (desk off) keeps the previous pick. Values are marked at the
    last completed close; MOM/REV/QQQ are held over the same open → close span."""
    recs = {r["session"]: r for r in records}
    if not recs:
        return {"sessions": 0}
    start = min(recs)
    idx = [j for j, d in enumerate(days) if d >= start]
    out = {"start": start, "sessions": len(idx), "locked": len(recs),
           "missing": [days[j] for j in idx if days[j] not in recs]}
    if not idx:
        return out
    for name in DIALS:
        picks, last = [], "MOM"
        for j in idx:
            last = recs[days[j]][name] if days[j] in recs else last
            picks.append(last)
        sub_o = {k: v[idx[0]:] for k, v in opn.items()}
        sub_c = {k: v[idx[0]:] for k, v in cls.items()}
        out[name] = float(chain(picks, sub_o, sub_c)[-1] - 1)
    for sp in opn:
        out[sp] = float(cls[sp][idx[-1]] / opn[sp][idx[0]] - 1)
    return out


# ---------------------------------------------------------------------------
# book-based computation
# ---------------------------------------------------------------------------

def _rev_pick(book: S.Book, k: int, ranked: list[dict]) -> dict[str, float]:
    scored = []
    for r in ranked:
        px = S._col(book, book.adj, r["key"])
        if px is None or k - 1 - REV_LOOKBACK < 0:
            continue
        a, b = px[k - 1 - REV_LOOKBACK], px[k - 1]
        if np.isfinite(a) and np.isfinite(b) and a > 0:
            scored.append((b / a - 1, r["key"]))
    scored.sort()
    pick = scored[:REV_N]
    raw = {key: 1.0 / len(pick) for _, key in pick}
    return S.project_caps(raw, book.sector) if raw else {}


def rev_path(book: S.Book, days: list[str], rankings: list[list[dict]]):
    """REV basket replayed like run_model: weekly targets at the ISO week's first session open."""
    k0 = book.days.index(days[0])
    value, close, weights, w, started = [1.0], [], [], {}, False
    for j, day in enumerate(days):
        k = k0 + j
        if j > 0:
            w, gross = S._drift(w, S._open_returns(book, w, k))
            value.append(value[-1] * (1 + gross))
        if (not started or S.iso_week(day) != S.iso_week(days[j - 1])) and rankings[j]:
            pick = _rev_pick(book, k, rankings[j])
            if pick:
                w, started = pick, True
        weights.append(dict(w))
        close.append(value[-1] * (1 + S._drift(w, S._intraday(book, w, k))[1]))
    return np.array(value), np.array(close), weights


def momentum_ic(book: S.Book, days: list[str], rankings: list[list[dict]], horizon: int) -> np.ndarray:
    k0 = book.days.index(days[0])
    out = np.full(len(days), np.nan)
    for j, ranked in enumerate(rankings):
        k = k0 + j
        if len(ranked) < 8 or k - 1 + horizon >= len(book.days):
            continue
        fwd, mom = [], []
        for r in ranked:
            px = S._col(book, book.adj, r["key"])
            if px is None or r.get("mom120") is None:  # 성적은 120일 수익률로 잰다(§11.2 정의 고정)
                continue
            a, b = px[k - 1], px[k - 1 + horizon]
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                fwd.append(b / a - 1)
                mom.append(r["mom120"])
        if len(fwd) >= 8:
            out[j] = pd.Series(mom).rank().corr(pd.Series(fwd).rank())
    return out


def compute(book: S.Book, path: S.Path) -> dict:
    """2026 replay (hindsight) plus the dial picks for the session after the last completed close."""
    days = path.days
    k0, n = book.days.index(days[0]), len(book.days)
    session = session_after(book.days[-1])
    rankings = [S.momentum_ranking(book, k0 + j, day)[1] for j, day in enumerate(days)]
    nxt_rank = S.momentum_ranking(book, n, session)[1]
    ic = {h: momentum_ic(book, days, rankings, h) for h in sorted({d["h"] for d in DIALS.values()})}
    if "SOXX" not in book.cls.columns:
        raise SeasonError("SOXX 일봉이 없습니다(데스크는 Desk.ensure에서 SOXX를 함께 받습니다).")
    soxx = book.cls["SOXX"].to_numpy(float)
    sma = pd.Series(soxx).rolling(SMA).mean().to_numpy()
    up_close = [None if not np.isfinite(m) else bool(c >= m) for c, m in zip(soxx, sma)]
    T = len(days)
    up = [up_close[k0 + j - 1] for j in range(T)] + [up_close[n - 1]]  # decision for j uses close j-1
    rv, rc, rw = rev_path(book, days, rankings)
    qa = book.adj["QQQ"].to_numpy(float)[k0:k0 + T]
    qf = (book.adj["QQQ"] / book.cls["QQQ"] * book.opn["QQQ"]).to_numpy(float)[k0:k0 + T]
    opn = {"MOM": path.value, "REV": rv, "QQQ": qf / qf[0]}
    cls = {"MOM": path.close, "REV": rc, "QQQ": qa / qf[0]}
    dials, known_now = {}, {}
    for name, d in DIALS.items():
        kn = known_ic(np.r_[ic[d["h"]], np.nan], d["h"], d["K"])  # length T+1; last = the next session
        known_now[name] = float(kn[-1]) if np.isfinite(kn[-1]) else None
        ch = [choose(kn[j], up[j], d["gate"]) for j in range(T + 1)]
        dials[name] = {**d, "pick": ch[T], "history": ch[:T], "close": chain(ch[:T], opn, cls),
                       "switches": [{"date": days[j], "to": ch[j]} for j in range(1, T) if ch[j] != ch[j - 1]]}
    if S.iso_week(session) != S.iso_week(days[-1]) and nxt_rank:
        rev_now = _rev_pick(book, n, nxt_rank)
    else:
        rev_now, _ = S._drift(rw[-1], S._intraday(book, rw[-1], n - 1))
    mom_now = S.next_target(book, path, session)["target"]
    return {"days": days, "session": session, "ic": ic, "known_now": known_now, "soxx_up_next": up[-1],
            "soxx_last": float(soxx[n - 1]), "soxx_sma": float(sma[n - 1]), "dials": dials, "opn": opn, "cls": cls,
            "rev_now": {book.ticker[k]: w for k, w in rev_now.items()},
            "mom_now": {book.ticker[k]: w for k, w in mom_now.items()}}


def season_label(ic20, ic10, soxx_up) -> tuple[str, str]:
    if soxx_up is False:
        return "급락 경계", "SOXX가 20일선 아래에서 마감했습니다. 2026에 급락 시즌을 알린 방향 신호입니다."
    bad = [x for x in (ic20, ic10) if x is not None and x < 0]
    if len(bad) == 2:
        return "되돌림 시즌", "최근 확정된 모멘텀 성적이 두 설정 모두 음수입니다. 많이 오른 종목이 그 뒤에 덜 오른 구간입니다."
    if len(bad) == 1:
        return "전환 의심", "모멘텀 성적이 한 설정에서만 음수입니다. 시즌이 바뀌는 중일 수 있습니다."
    return "모멘텀 시즌", "최근 확정된 모멘텀 성적이 양수(또는 아직 미확정)입니다. 오른 종목이 그 뒤에도 더 오른 구간입니다."


def _load_paper() -> dict:
    if PAPER.exists():
        return json.loads(PAPER.read_text(encoding="utf-8"))
    return {"schema": 1, "note": "시즌 계기판 종이 기록(docs/desk_system.md §11.2). 세션마다 처음 계산한 선택을 잠근다. "
                                 "수량·비중·개인 계좌 정보 없음.", "records": []}


_LOCK = threading.Lock()
_CACHE: dict = {}


def payload(desk, force: bool = False) -> dict:
    with _LOCK:
        desk.ensure()
        book, path = desk.book, desk.path
        if book is None or path is None:
            raise SeasonError("7종목 계산이 아직 끝나지 않았습니다. 잠시 뒤 다시 여세요.")
        if not force and _CACHE.get("key") == book.days[-1] and time.time() - _CACHE.get("at", 0) < CACHE_SEC:
            return _CACHE["data"]
        res = compute(book, path)
        session = res["session"]
        ic20, ic10 = res["known_now"]["dial1"], res["known_now"]["dial2"]
        label, why = season_label(ic20, ic10, res["soxx_up_next"])
        now_kst = datetime.now(config.KST)
        _, open_kst = config.next_us_session(datetime.now(config.NY))
        after_open = open_kst.date().isoformat() > session  # 다음 개장이 이 세션 뒤라면 이 세션은 이미 열렸다
        ledger = _load_paper()
        rec = {"session": session, "recorded_at": now_kst.isoformat(timespec="seconds"), "after_open": after_open,
               "basis_close": book.days[-1], "ic_h20_k20": ic20, "ic_h10_k10": ic10,
               "soxx_close": res["soxx_last"], "soxx_sma20": res["soxx_sma"], "soxx_up": res["soxx_up_next"],
               "dial1": res["dials"]["dial1"]["pick"], "dial2": res["dials"]["dial2"]["pick"], "season": label,
               "mom_basket": sorted(res["mom_now"]), "rev_basket": sorted(res["rev_now"])}
        if append_record(ledger, clean(rec)):
            PAPER.parent.mkdir(parents=True, exist_ok=True)
            PAPER.write_text(json.dumps(clean(ledger), ensure_ascii=False, indent=1), encoding="utf-8")
        days = res["days"]
        jul = next(j for j, d in enumerate(days) if d >= "2026-07-01")
        hist = []
        for nm, s in (("현행 7종목", res["cls"]["MOM"]), ("최근 5일 최하위 7종목", res["cls"]["REV"]),
                      ("성격 다이얼", res["dials"]["dial1"]["close"]), ("두 다이얼", res["dials"]["dial2"]["close"]),
                      ("QQQ", res["cls"]["QQQ"])):
            s = np.asarray(s, dtype=float)
            hist.append({"name": nm, "ret": float(s[-1] - 1), "mdd": _drawdown(np.r_[1.0, s]),
                         "since_jul": float(s[-1] / s[jul - 1] - 1)})
        tail = 90
        data = {
            "computed_at": now_kst.isoformat(timespec="seconds"), "as_of": book.days[-1], "session": session,
            "after_open": after_open,
            "season": {"label": label, "why": why},
            "ic": {"h20_k20": ic20, "h10_k10": ic10},
            "soxx": {"close": res["soxx_last"], "sma20": res["soxx_sma"], "up": res["soxx_up_next"]},
            "picks": {n: {"name": d["name"], "pick": d["pick"],
                          "rule": ("SOXX 20일선 아래면 현금, 위면 " if d["gate"] else "")
                          + f"최근 {d['K']}개 확정 모멘텀 성적(h{d['h']})이 음수면 최하위 7종목, 아니면 현행 7종목",
                          "switches": d["switches"][-6:]} for n, d in res["dials"].items()},
            "baskets": {"MOM": res["mom_now"], "REV": res["rev_now"]},
            "chart": {"days": days[-tail:], "ic20": known_ic(res["ic"][20], 20, 20)[-tail:],
                      "ic10": known_ic(res["ic"][10], 10, 10)[-tail:]},
            "history": hist,
            "paper": paper_score(ledger["records"], days, res["opn"], res["cls"]),
            "notes": [
                "관찰용 계기판입니다. 7종목 규칙과 주문 초안은 바꾸지 않습니다(사용자 결정 2026-11-20, 규칙 정본 §3·§3.1).",
                "모멘텀 성적은 120일 모멘텀 순위와 그 뒤 h거래일 수익의 순위 상관입니다. h거래일이 지나야 확정되므로 확정된 값만 씁니다"
                "(성격 다이얼 h20·K20, 두 다이얼 h10·K10).",
                "2026 성적표는 지금 규칙을 2026 처음부터 다시 돌린 사후 재생입니다. 이 설정들은 2026 자료로 고른 것이라 실제 기대보다 "
                "좋게 나온 쪽입니다. 2025에는 성격 다이얼 +8.3%(모멘텀 단독 +14.5%), 두 다이얼 −53.9%였습니다(season_regime_20261119).",
                "종이 기록은 마지막 확정 종가 다음 세션의 선택을 처음 계산한 순간 잠그고 고치지 않습니다. 개장 뒤에 처음 잠근 줄은 "
                "'개장 뒤 기록'으로 표시합니다(쓰는 정보는 그 전 종가까지라 미래 정보는 없습니다). 기록이 빠진 날은 직전 선택을 유지한 것으로 셉니다.",
            ],
        }
        _CACHE.update(key=book.days[-1], at=time.time(), data=data)
        return data
