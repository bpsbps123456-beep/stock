"""AI 탐방 — the research desk visits the AI supply chain in product order, from the most basic
inputs (power, materials, tools) to the finished product (the model) and the apps that sell it.

The route (stops, zones, companies, OpenAI/Anthropic links and their sources) is curated by
hand in desk/tour_map.json; this module validates it and adds the latest TIME/KoAct weights.
7종목 targets and my positions come from /api/state on the page, so the tour never waits for
prices. Rules: docs/desk_system.md §8.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import watchlist
from gbrc.etf.market import company_key, is_cash_holding
from gbrc.etf.storage import NORMALIZED_HOLDINGS_DIR, load_normalized_snapshot
from watchlist import resolve_sector

from .scenes import load as load_scenes

MAP_PATH = Path(__file__).with_name("tour_map.json")
FUNDS = (("456600", "TIME"), ("2ETFQ1", "KoAct"))
AXES = ("openai", "anthropic")
MARKETS = {"US", "KR", "JP", "HK", "CN", "TW", "EU", "IN", "private", "fund"}
POSITIONS = {"독점", "선두", "추격"}
KINDS = {"투자", "클라우드", "칩", "메모리", "데이터센터", "전력", "네트워크", "서버", "유통"}
STATUSES = {"확정", "의향", "보도", "확대", "축소", "종료"}
STOP_FIELDS = ("leg", "name", "site", "makes", "gets", "gives", "need")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CLAIM = re.compile(r"\d[\d,.]*[%위]?|최대|최고|유일")  # what a reason may only say with a source
RANK = re.compile(r"\d+위|최대|최고|유일")
# Yahoo suffixes used by watchlist.AI_UNIVERSE; longest first so ".TWO" is not read as ".T".
UNIVERSE_SUFFIXES = (".TWO", ".TW", ".KS", ".KQ", ".SZ", ".SS", ".HK", ".VI", ".DE", ".PA", ".T")


def symbol(code) -> str:
    """'NVDA US EQUITY' -> 'NVDA', '285A JP EQUITY' -> '285A', 'GOOGL …' -> 'GOOG', futures -> 'NQ'."""
    key = company_key(code)
    return key.split()[0] if key else ""


def universe_symbol(sym: str) -> str:
    """'005930.KS' -> '005930', '285A.T' -> '285A', '0100.HK' -> '100' (the funds' own HK code)."""
    for suffix in UNIVERSE_SUFFIXES:
        if sym.endswith(suffix):
            base = sym[: -len(suffix)]
            return (base.lstrip("0") or "0") if suffix == ".HK" else base
    return sym


def universe(m: dict) -> dict[str, str]:
    """Map symbol -> 'group · sub' for watchlist.AI_UNIVERSE, the AI universe the ATH-gap study
    owns. The route must visit every one of them; it may hold more (axis partners, private labs,
    the power chain)."""
    alias = {a: t for t, c in m["companies"].items() for a in c.get("aliases", [])}
    out: dict[str, str] = {}
    for group, subs in getattr(watchlist, "AI_UNIVERSE", {}).items():
        for sub, syms in subs.items():
            for sym in syms:
                t = universe_symbol(sym)
                out[alias.get(t, t)] = f"{group} · {sub}"
    return out


def load_map(path: Path = MAP_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def zones(m: dict):
    return [(s, z) for s in m.get("stops", []) for z in s.get("zones", [])]


def _iso(value) -> bool:
    if not isinstance(value, str) or not ISO.match(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def check_map(m: dict) -> list[str]:
    """Every problem that would make the tour lie or break. Empty list = publishable."""
    out: list[str] = []
    if not _iso(m.get("as_of")):
        out.append("as_of가 YYYY-MM-DD가 아닙니다")
    if not isinstance(m.get("review_days"), int) or m["review_days"] <= 0:
        out.append("review_days는 양의 정수여야 합니다")
    companies = m.get("companies", {})
    names: dict[str, str] = {}
    for t, c in companies.items():
        for field in ("name", "ko"):
            if not str(c.get(field, "")).strip():
                out.append(f"{t}: {field} 없음")
        if c.get("market") not in MARKETS:
            out.append(f"{t}: market {c.get('market')!r}")
        for alias in [t, *c.get("aliases", [])]:
            if alias in names and names[alias] != t:
                out.append(f"{alias}: {names[alias]}와 {t}가 같은 기호를 씁니다")
            names[alias] = t

    stop_ids = [s.get("id") for s in m.get("stops", [])]
    cells: dict[tuple, str] = {}
    if len(set(stop_ids)) != len(stop_ids):
        out.append("방문지 id 중복")
    for i, s in enumerate(m.get("stops", []), 1):
        if s.get("no") != i:
            out.append(f"{s.get('id')}: 순번 no가 {i}가 아닙니다 (탐방은 기초→완제품 순서)")
        for field in STOP_FIELDS:
            if not str(s.get(field, "")).strip():
                out.append(f"{s.get('id')}: {field} 없음")
        for ref in (*s.get("from", []), *s.get("to", [])):
            if ref not in stop_ids:
                out.append(f"{s.get('id')}: 없는 방문지 {ref!r}를 가리킵니다")
        at = s.get("at")
        if not (isinstance(at, list) and len(at) == 2 and all(isinstance(v, int) and v >= 0 for v in at)):
            out.append(f"{s.get('id')}: 흐름도 자리 at=[열, 줄]이 필요합니다")
        elif tuple(at) in cells:
            out.append(f"{s.get('id')}: 흐름도 자리 {at}가 {cells[tuple(at)]}와 겹칩니다")
        else:
            cells[tuple(at)] = s.get("id")

    placements: dict[str, list[tuple[str, dict]]] = {}
    zone_ids = set()
    linked = {ln.get("t") for ln in m.get("links", [])}
    for _, z in zones(m):
        zid = z.get("id")
        if zid in zone_ids:
            out.append(f"구역 id 중복: {zid}")
        zone_ids.add(zid)
        for field in ("name", "need"):
            if not str(z.get(field, "")).strip():
                out.append(f"{zid}: {field} 없음")
        if not z.get("members") and z.get("special") != "axes":
            out.append(f"{zid}: 회사가 없습니다")
        if "src" in z and not str(z["src"]).startswith("http"):
            out.append(f"{zid}: src는 URL이어야 합니다")
        if z.get("special") != "axes":
            if z.get("lead") not in {mem.get("t") for mem in z.get("members", [])}:
                out.append(f"{zid}: 대장 lead가 이 부서의 회사가 아닙니다 ({z.get('lead')!r})")
            if not str(z.get("lead_why", "")).strip():
                out.append(f"{zid}: 대장 근거 lead_why 없음")
            if "lead_src" in z and not str(z["lead_src"]).startswith("http"):
                out.append(f"{zid}: lead_src는 URL이어야 합니다")
            # a number or rank in the lead's reason must quote what the lead's card already sources
            lead = next((mem for mem in z.get("members", []) if mem.get("t") == z.get("lead")), {})
            backed = str(lead.get("role", "")) if lead.get("src") else ""
            loose = [c for c in CLAIM.findall(str(z.get("lead_why", ""))) if c not in backed]
            if loose and not z.get("lead_src"):
                out.append(f"{zid}: 대장 근거의 {', '.join(loose)}가 출처 있는 문장에 없습니다"
                           " — 명함 문장(role+src)을 인용하거나 lead_src를 넣으세요")
        seen = set()
        for mem in z.get("members", []):
            t = mem.get("t")
            if t not in companies:
                out.append(f"{zid}: 지도에 없는 회사 {t}")
            if t in seen:
                out.append(f"{zid}: {t} 중복")
            seen.add(t)
            if not str(mem.get("role", "")).strip():
                out.append(f"{zid}/{t}: role 없음")
            if "pos" in mem and mem["pos"] not in POSITIONS:
                out.append(f"{zid}/{t}: pos {mem['pos']!r}")
            if "src" in mem and not str(mem["src"]).startswith("http"):
                out.append(f"{zid}/{t}: src는 URL이어야 합니다")
            if RANK.search(str(mem.get("role", ""))) and not mem.get("src") and t not in linked:
                out.append(f"{zid}/{t}: 순위·최대 같은 주장에 출처가 없습니다 (src를 넣거나 표현을 바꾸세요)")
            placements.setdefault(t, []).append((zid, mem))

    outside = {o.get("t") for o in m.get("outside", [])}
    for o in m.get("outside", []):
        if o.get("t") in placements:
            out.append(f"{o.get('t')}: 경로 안과 밖에 동시에 있습니다")
        if not str(o.get("why", "")).strip():
            out.append(f"{o.get('t')}: 밖에 둔 이유 없음")
    for t in companies:
        spots = placements.get(t, [])
        mains = [z for z, mem in spots if mem.get("main")]
        if not spots and t not in outside and companies[t].get("market") != "private":
            out.append(f"{t}: 어느 구역에도 없습니다")
        if len(spots) > 1 and len(mains) != 1:
            out.append(f"{t}: 여러 구역에 있으면 main 하나를 정해야 합니다 ({', '.join(z for z, _ in spots)})")

    for axis in AXES:
        a = m.get("axes", {}).get(axis)
        if not a:
            out.append(f"axes.{axis} 없음")
            continue
        for fact in a.get("facts", []):
            if not str(fact.get("src", "")).startswith("http") or not _iso(fact.get("date")):
                out.append(f"axes.{axis}: '{fact.get('label')}' 출처·날짜 없음")
    as_of = m.get("as_of", "")
    for i, ln in enumerate(m.get("links", [])):
        tag = f"links[{i}] {ln.get('axis')}/{ln.get('t')}"
        if ln.get("axis") not in AXES:
            out.append(f"{tag}: axis")
        if ln.get("t") not in companies:
            out.append(f"{tag}: 지도에 없는 회사")
        if ln.get("kind") not in KINDS:
            out.append(f"{tag}: kind {ln.get('kind')!r}")
        if ln.get("status") not in STATUSES:
            out.append(f"{tag}: status {ln.get('status')!r}")
        if not _iso(ln.get("date")) or (_iso(as_of) and ln["date"] > as_of):
            out.append(f"{tag}: 날짜 {ln.get('date')!r}")
        if not str(ln.get("src", "")).startswith("http"):
            out.append(f"{tag}: 출처 URL 없음")
        if not str(ln.get("what", "")).strip():
            out.append(f"{tag}: 내용 없음")
        for num in ("gw", "usd_bn"):
            if num in ln and not (isinstance(ln[num], (int, float)) and ln[num] > 0):
                out.append(f"{tag}: {num}")
    return out


def latest_holdings(fund_id: str, root: Path = NORMALIZED_HOLDINGS_DIR) -> dict | None:
    """The newest verified snapshot as {symbol: NAV fraction}, all listings, cash dropped."""
    paths = sorted(Path(root).glob(f"*_{fund_id}.json"))
    if not paths:
        return None
    snap = load_normalized_snapshot(paths[-1])
    weights: dict[str, float] = {}
    names: dict[str, str] = {}
    for row in snap["holdings"]:
        if is_cash_holding(row):
            continue
        sym = symbol(row.get("code"))
        weights[sym] = weights.get(sym, 0.0) + float(row["weight"]) / 100.0
        names.setdefault(sym, str(row.get("name") or ""))
    return {"date": paths[-1].name[:10], "weights": weights, "names": names}


def payload(today: date | None = None, root: Path = NORMALIZED_HOLDINGS_DIR,
            path: Path = MAP_PATH) -> dict:
    m = load_map(path)
    problems = check_map(m)
    alias = {a: t for t, c in m["companies"].items() for a in c.get("aliases", [])}
    placed = {mem["t"] for _, z in zones(m) for mem in z.get("members", [])}
    outside = {o["t"]: o for o in m.get("outside", [])}

    funds, held = {}, {}
    for fund_id, label in FUNDS:
        snap = latest_holdings(fund_id, root)
        if not snap:
            continue
        weights: dict[str, float] = {}
        for sym, w in snap["weights"].items():
            t = alias.get(sym, sym)
            weights[t] = weights.get(t, 0.0) + w
            row = held.setdefault(t, {"t": t, "name": snap["names"][sym], "funds": {}})
            row["funds"][label] = weights[t]
        funds[label] = {"id": fund_id, "date": snap["date"], "weights": weights}

    def described(row: dict) -> dict:
        reason = outside.get(row["t"], {})
        return {**row, "sector": reason.get("group") or resolve_sector(row["t"]), "why": reason.get("why"),
                "ko": reason.get("ko") or m["companies"].get(row["t"], {}).get("ko")}

    by_weight = lambda r: -max(r["funds"].values())  # noqa: E731
    unplaced = sorted((described(r) for t, r in held.items() if t not in placed and t not in outside), key=by_weight)
    outside_held = sorted((described(r) for t, r in held.items() if t in outside), key=by_weight)
    uni = universe(m)
    today = today or date.today()
    age = (today - date.fromisoformat(m["as_of"])).days if _iso(m.get("as_of")) else None
    return {"map": m, "alias": alias, "funds": funds, "unplaced": unplaced, "outside_held": outside_held,
            "problems": problems, "age_days": age, "stale": age is not None and age > m.get("review_days", 45),
            "sectors": {t: resolve_sector(t) for t in m["companies"]},
            "universe": uni, "universe_missing": sorted(t for t in uni if t not in placed),
            "scenes": load_scenes()}
