"""저평가 표시: AI 산업 지도에서 이익 전망은 그대로이거나 올랐는데 주가가 덜 따라간 분야와 회사.

정본은 docs/desk_system.md §8.4다. 병목처럼 지도 위에 다는 관찰 표시라 매수 추천·목표 비중·주문
수량을 만들지 않는다(§1). 가격·전망은 yfinance다. TradingView MCP 값은 넣지 않는다(AGENTS.md).

최근 30일, 같은 회계연도(다음 회계연도) 기준으로
  R = EPS 컨센서스 변화율, P = 주가 수익률(배당 조정), V = (1+P)/(1+R) - 1 = 선행 PER의 변화.
두 시점 모두 흑자이고 애널리스트가 3명 이상인 회사만 잰다(적자는 PER로 판단하지 않는다).
- 저평가 분야: 잴 수 있는 회사가 3곳 이상인 분야를 '분야 중앙값 V - 전체 중앙값 V'로 줄 세운 하위
  1/3 가운데, 분야 중앙값 R > 0이고 분야 중앙값 V < 0인 분야(순환매: 밀려난 분야).
- 저평가 회사: 대표 분야(지도의 main)가 저평가 분야이고, 자기 R > 0, V < 0인 회사.
미래 이익이나 PER 수준 하나로 줄 세우면 이익이 정점인 메모리가 늘 싸 보인다(사용자 2026-10-10).
그래서 회사마다 '자기 전망 대비 주가'의 30일 변화를 재고 분야끼리 비교한다. PER 수준·성장 대비
PER·전망 상향만으로는 2026년 두 달 검증에서 앞서지 못했다(data/derived/analysis/value_signal/report.md).
"""

from __future__ import annotations

import json
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from statistics import median

import pandas as pd

import ai_ath_gap

from . import config, tour
from .state import clean

CACHE_PATH = config.CACHE / "value.json"
MAX_AGE_MIN = 360        # 컨센서스는 하루 한두 번 바뀐다
DAYS = 30                # 비교 기간(달력일). yfinance eps_trend의 '30daysAgo' 열과 맞춘다
TREND_COL = "30daysAgo"
MIN_ANALYSTS = 3
MIN_MEMBERS = 3          # 분야 판정에 필요한 '잴 수 있는 회사' 수
SHARE = 1 / 3            # 분야를 줄 세운 하위 몫
# 메모리·저장은 이익이 정점일 때 PER이 가장 낮아 보이는 경기순환 업종이라 저평가 표시에 주의를 단다.
CYCLE_ZONES = {"hbm", "flash", "disk"}
# 2026년 두 달 전방 검증(신호 8/11·9/10, 각 다음 한 달, 지도 상장 265곳 중앙값 대비 %p). 이 규칙을 그대로
# 돌린 값이다(analyze.py --production). 화면이 이 값을 그대로 읽는다.
CHECK = {
    "period": "2026-08~10", "months": 2, "horizon": "다음 한 달",
    "zone_flag": [2.3, 2.1], "zone_hit": [62, 64], "zone_dear": [-1.8, -4.6],
    "company_mean": [2.4, 1.7], "company_median": [0.3, 1.3], "company_hit": [53, 55],
    "fades": "다만 두 달을 보면 저평가 분야도 AI 전체보다 2.5%p 뒤졌습니다(가장 비싸진 1/3은 6.4%p)",
    "report": "data/derived/analysis/value_signal/report.md",
}
_SUFFIXES = {"US": ("",), "KR": (".KS", ".KQ"), "JP": (".T",), "TW": (".TW", ".TWO"), "IN": (".NS",),
             "CN": (".SS", ".SZ"), "HK": (".HK",)}
_LOCK = threading.Lock()


# ----------------------------------------------------------------------------- the rule (pure)
def zone_index(m: dict) -> tuple[dict[str, list[str]], dict[str, str]]:
    """분야 → 회사 키, 회사 키 → 대표 분야(지도 main, 없으면 처음 나온 분야; tour.js와 같다)."""
    zones: dict[str, list[str]] = {}
    first: dict[str, str] = {}
    main: dict[str, str] = {}
    for _, z in tour.zones(m):
        if z.get("special"):
            continue
        for mem in z.get("members", []):
            zones.setdefault(z["id"], []).append(mem["t"])
            first.setdefault(mem["t"], z["id"])
            if mem.get("main"):
                main.setdefault(mem["t"], z["id"])
    return zones, {t: main.get(t, z) for t, z in first.items()}


def measure(est: dict, p_now: float | None, p_then: float | None) -> dict:
    """한 회사의 R·P·V. est = {'+1y': {'current':…, '30daysAgo':…}, 'analysts': n}."""
    y1 = (est or {}).get("+1y") or {}
    e1, e0 = y1.get("current"), y1.get(TREND_COL)
    row: dict = {"analysts": (est or {}).get("analysts")}
    if p_now and p_then:
        row["P"] = p_now / p_then - 1
    if e0 is not None and e1 is not None:
        if e0 > 0 and e1 > 0:
            row["R"] = e1 / e0 - 1
            if "P" in row:
                row["V"] = (1 + row["P"]) / (1 + row["R"]) - 1
        else:
            row["loss"] = True
    return row


def usable(row: dict) -> bool:
    return row.get("V") is not None and (row.get("analysts") or 0) >= MIN_ANALYSTS


def judge(rows: dict[str, dict], zones: dict[str, list[str]], main: dict[str, str]) -> dict:
    """회사 키 → measure() 결과로 분야·회사 저평가를 판정한다."""
    ok = {t: r for t, r in rows.items() if usable(r)}
    uni = {"n": len(ok), "V": median(r["V"] for r in ok.values()) if ok else None,
           "R": median(r["R"] for r in ok.values()) if ok else None}
    zrows: dict[str, dict] = {}
    for zid, members in zones.items():
        rs = [ok[t] for t in dict.fromkeys(members) if t in ok]
        if len(rs) < MIN_MEMBERS:
            continue
        v = median(r["V"] for r in rs)
        zrows[zid] = {"n": len(rs), "V": v, "R": median(r["R"] for r in rs), "P": median(r["P"] for r in rs),
                      "dV": v - uni["V"], "cycle": zid in CYCLE_ZONES}
    order = sorted(zrows, key=lambda z: zrows[z]["dV"])
    cut = max(1, int(len(order) * SHARE)) if order else 0
    for i, z in enumerate(order):
        zr = zrows[z]
        zr.update(rank=i + 1, of=len(order), flag=i < cut and zr["R"] > 0 and zr["V"] < 0)
    companies = {}
    for t, r in rows.items():
        z = main.get(t)
        c = {k: r[k] for k in ("R", "P", "V", "analysts") if r.get(k) is not None}
        c["zone"] = z
        if not usable(r):
            c["skip"] = "loss" if r.get("loss") else "thin" if r.get("V") is not None else "no_data"
        c["flag"] = bool(usable(r) and zrows.get(z, {}).get("flag") and r["R"] > 0 and r["V"] < 0)
        companies[t] = c
    return {"universe": uni, "zones": zrows, "companies": companies}


# ----------------------------------------------------------------------------- data
def map_symbols(m: dict) -> dict[str, list[str]]:
    """지도에 배치된 상장 회사 키 → Yahoo 심볼 후보. 유니버스 종목은 그 심볼, 나머지는 시장 접미사."""
    aliases = {a: t for t, c in m["companies"].items() for a in c.get("aliases", [])}
    known = {}
    for _, _, sym in ai_ath_gap.universe_rows():
        key = tour.universe_symbol(sym)
        known[aliases.get(key, key)] = sym
    out = {}
    for _, z in tour.zones(m):
        for mem in z.get("members", []):
            t, c = mem["t"], m["companies"].get(mem["t"], {})
            if t in out or c.get("market") in ("private", "fund"):
                continue
            if t in known:
                out[t] = [known[t]]
            else:
                base = t.zfill(4) if c.get("market") == "HK" else t
                out[t] = [base + sfx for sfx in _SUFFIXES.get(c.get("market"), ())]
    return {t: c for t, c in out.items() if c}


def fetch_estimates(symbol: str) -> tuple[str, dict]:
    import yfinance as yf

    t, out = yf.Ticker(symbol), {}
    try:
        et = t.eps_trend
        if et is not None and len(et) and "+1y" in et.index:
            out["+1y"] = {c: (None if pd.isna(et.loc["+1y", c]) else float(et.loc["+1y", c]))
                          for c in ("current", TREND_COL) if c in et.columns}
    except Exception:  # 전망이 없으면 비운다(가정으로 채우지 않음)
        pass
    try:
        ee = t.earnings_estimate
        if ee is not None and len(ee) and "+1y" in ee.index and not pd.isna(ee.loc["+1y", "numberOfAnalysts"]):
            out["analysts"] = int(ee.loc["+1y", "numberOfAnalysts"])
    except Exception:
        pass
    return symbol, out


def fetch_closes(symbols: list[str]) -> pd.DataFrame:
    import yfinance as yf

    px = yf.download(symbols, period="3mo", auto_adjust=True, progress=False, threads=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(symbols[0])
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    return px


def _on(px: pd.DataFrame, sym: str, day: pd.Timestamp) -> float | None:
    if sym not in px:
        return None
    s = px[sym].dropna()
    s = s[s.index <= day]
    return float(s.iloc[-1]) if len(s) else None


def compute(today: date | None = None) -> dict:
    m = tour.load_map()
    cands = map_symbols(m)
    px = fetch_closes(sorted({s for c in cands.values() for s in c}))
    sym_of = {t: next((s for s in c if s in px and px[s].notna().any()), None) for t, c in cands.items()}
    sym_of = {t: s for t, s in sym_of.items() if s}
    with ThreadPoolExecutor(max_workers=8) as pool:
        est = dict(pool.map(fetch_estimates, sorted(set(sym_of.values()))))
    now = pd.Timestamp(today or date.today())
    then = now - pd.Timedelta(days=DAYS)
    rows = {t: measure(est.get(s, {}), _on(px, s, now), _on(px, s, then)) for t, s in sym_of.items()}
    zones, main = zone_index(m)
    out = judge(rows, zones, main)
    for t, s in sym_of.items():
        out["companies"][t]["symbol"] = s
    last = px.index[px.notna().any(axis=1)].max()
    out.update(
        computed_at=datetime.now(config.KST).isoformat(timespec="seconds"),
        price_date=last.date().isoformat() if pd.notna(last) else None,
        window={"days": DAYS, "from": then.date().isoformat(), "to": now.date().isoformat()},
        map_as_of=m["as_of"], missing=sorted(set(cands) - set(sym_of)),
    )
    return out


def _rules(data: dict) -> dict:
    """규칙 상수와 검증 숫자는 캐시가 아니라 지금 코드에서 붙인다(규칙을 고치면 바로 화면에 반영)."""
    return {**data, "rules": {"min_analysts": MIN_ANALYSTS, "min_members": MIN_MEMBERS, "share": round(SHARE, 3)},
            "cycle_zones": sorted(CYCLE_ZONES), "check": CHECK}


def payload(force: bool = False, max_age_min: int = MAX_AGE_MIN) -> dict:
    """캐시가 max_age_min보다 새로우면 그대로 쓰고, 아니면 다시 계산한다(첫 계산 20~40초)."""
    with _LOCK:
        if not force and CACHE_PATH.exists():
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if _time.time() - cached.get("_epoch", 0) < max_age_min * 60:
                return _rules(cached)
        data = clean(compute())
        data["_epoch"] = _time.time()
        config.CACHE.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return _rules(data)
