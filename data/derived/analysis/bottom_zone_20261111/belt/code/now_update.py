"""인정된 바닥 매수 신호 BB-1 점검 (docs/desk_system.md §17).

1차 과제 A의 수집기·신호판 코드(`code/collect_A.py`, `code/panel_A.py`)를 그대로 부르되
결과는 새 폴더(belt/now)에만 쓴다. 동결된 1차 자료(bottom_zone_20261111/data)는 읽기만 한다.

순서: 동결 원응답·캐시 중 없는 것만 복사 → 최근 빈 응답 다시 받기 → 가격·CNN 새로 받기
→ 신호판·조건 49개 재생성 → BB-1 켜짐, 강한 조합 겹침 수, 바닥 감지기(§13.6) 상태 출력.

실행: py data/derived/analysis/bottom_zone_20261111/belt/code/now_update.py [--skip-fetch]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BELT = HERE.parent
ZONE = BELT.parent                          # bottom_zone_20261111
ANALYSIS = ZONE.parent
REPO = ZONE.parents[3]
NOW = BELT / "now"
RES = BELT / "results"
FIRST_NEW = pd.Timestamp("2026-11-09")      # 1차 동결 뒤 다시 받기 시작하는 SEIBRO 날짜
CAL_END = pd.Timestamp("2027-02-26")        # build.py·calendar_map.py 달력 끝(2027-03-01) 앞
REFETCH_EMPTY_DAYS = 14                     # 최근 이만큼의 빈 응답은 게시 전일 수 있어 다시 받는다
MAIN = ("낙폭20", "SOXL비중상위20")           # BB-1 정의
STACK_BINS = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 10**9, "11+")]


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def bucket(n: int) -> str:
    for lo, hi, label in STACK_BINS:
        if lo <= n <= hi:
            return label
    raise ValueError(n)


def main_on(row) -> bool:
    return bool(all(bool(row[k]) for k in MAIN))


def strong_count(row, combos: list[str]) -> int:
    return sum(all(bool(row[p]) for p in c.split(" + ")) for c in combos)


# ---------------------------------------------------------------- 수집
def prepare(today: pd.Timestamp):
    for sub in ("seibro", "cache"):
        src, dst = ZONE / "data" / sub, NOW / "data" / sub
        for f in src.rglob("*"):
            target = dst / f.relative_to(src)
            if f.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
    recent = max(FIRST_NEW, today - pd.tseries.offsets.BDay(REFETCH_EMPTY_DAYS))
    for f in (NOW / "data" / "cache").glob("*.json"):
        day = pd.Timestamp(f.stem.split("_")[-1])
        if day >= recent and json.loads(f.read_text(encoding="utf-8"))["n"] == 0:
            f.unlink()


def fetch(today_kst: pd.Timestamp, today_ny: pd.Timestamp) -> dict:
    col = module("now_collect", ZONE / "code" / "collect_A.py")
    col.ROOT = NOW
    col.market(end=str((today_ny + timedelta(days=1)).date()))
    got = {}
    for day in pd.bdate_range(FIRST_NEW, today_kst):
        for kind, dtype in [("daily", v) for v in "1234"] + [("custody", None)]:
            got[f"{kind}_{dtype or 'amt'}_{day:%Y%m%d}"] = col.fetch((day, kind, dtype))["n"]
    return got


def build(source_end: pd.Timestamp) -> pd.DataFrame:
    pa = module("now_panel", ZONE / "code" / "panel_A.py")
    original_load = pa.load

    def load(name, path):                    # NOW 기준 상대 경로를 원래 연구 위치로 돌린다
        p = Path(path)
        if not p.exists() and "seibro_flows" in p.parts:
            p = ANALYSIS / "seibro_flows" / "code" / "collect.py"
        return original_load(name, p)

    pa.load = load
    pa.ROOT = NOW
    pa.raw_panel(source_end=str(source_end.date()))
    census = REPO / "research" / "pb_flow_census_20261026" / "census.csv"
    panel = pa.assemble(*pa.inputs(str(census)))
    panel.to_csv(NOW / "data" / "signal_panel.csv", index_label="date")
    return panel


def atoms_now(panel: pd.DataFrame):
    bs = module("now_belt", HERE / "belt_search.py")
    px = pd.read_csv(NOW / "data" / "prices.csv", parse_dates=["date"])
    wide = {t: g.set_index("date").sort_index() for t, g in px.groupby("ticker")}
    cnn = json.loads((NOW / "data" / "cnn.json").read_text(encoding="utf-8"))
    news = pd.read_csv(ZONE / "news" / "v2" / "news_series_B.csv", parse_dates=["date"], encoding="utf-8-sig")
    atoms = bs.build_atoms(panel, wide, cnn, news)
    atoms.to_csv(NOW / "atoms_now.csv", index_label="date", encoding="utf-8-sig")
    return wide, atoms


# ---------------------------------------------------------------- 상태
def history_rates() -> dict:
    """겹침 수별 과거 벨트10 비율(2024~2026 하락 상태, 사후 선택 조합이라 설명용)."""
    d = pd.read_csv(RES / "daily_belt.csv", parse_dates=["date"]).set_index("date")
    st = pd.read_csv(RES / "strong_stack_daily.csv", parse_dates=["date"]).set_index("date")
    d = d.join(st)
    d = d[(d["down"] == True) & d["belt10"].notna()]  # noqa: E712
    d["bucket"] = d["n_strong"].astype(int).map(bucket)
    out = {}
    for label in [b[2] for b in STACK_BINS]:
        g = d[d["bucket"] == label]
        out[label] = dict(days=int(len(g)), belt10=float(g["belt10"].mean()) if len(g) else None,
                          lows=int(g["anchor"].nunique()))
    main = d[d["낙폭20"].astype(bool) & d["SOXL비중상위20"].astype(bool)]
    out["BB-1"] = dict(days=int(len(main)), belt10=float(main["belt10"].mean()), lows=int(main["anchor"].nunique()))
    return out


def report(panel, wide, atoms, got, now_kst, now_ny):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from desk import bottom as bt  # noqa: PLC0415  바닥 감지기 정본(§13.6)
    close = wide["SOXX"]["close"]
    fr = bt.frame(close)
    det = bt.status(fr, bt.signals(fr))
    combos = pd.read_csv(RES / "strong_combos.csv", encoding="utf-8-sig")["조합"].tolist()
    rates = history_rates()
    rows = []
    for d in atoms.index[-8:]:
        r, p = atoms.loc[d], panel.loc[d]
        n = strong_count(r, combos)
        rows.append(dict(
            관찰일=str(d.date()), 직전종가=round(float(close.loc[:d - timedelta(days=1)].iloc[-1]), 2),
            낙폭60=round(float(p["dd60"]), 4), SOXL비중백분위=None if pd.isna(p["T5_P"]) else round(float(p["T5_P"]), 1),
            공포탐욕=None if pd.isna(p["G"]) else round(float(p["G"]), 1),
            수급원거래일=str(p["source_date"])[:10], 수급공개=bool(p["source_published"] == True),  # noqa: E712
            BB1=main_on(r), 강한조합겹침=n, 겹침구간=bucket(n), 구간과거벨트10=rates[bucket(n)]["belt10"]))
    low = float(det["low"]) if det.get("low") is not None else np.nan
    status = dict(
        checked_at_kst=now_kst.strftime("%Y-%m-%d %H:%M"), checked_at_ny=now_ny.strftime("%Y-%m-%d %H:%M"),
        last_close=dict(date=str(close.index[-1].date()), close=float(close.iloc[-1])),
        bottom_detector=dict(kind=det.get("kind"), low=low, low_date=det.get("low_date"), up=det.get("up"),
                             dd=det.get("dd"), belt10_line=round(low * 1.10, 2) if np.isfinite(low) else None),
        history=rates, rows=rows, seibro_fetched_recent={k: v for k, v in got.items() if v == 0},
        note="BB-1 = SOXX 60세션 낙폭 20% 이상 + 한국인 SOXL 매수 비중 5세션 백분위 80 이상. 과거 비율은 사후 탐색 값",
    )
    (NOW / "now_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(NOW / "now_table.csv", index=False, encoding="utf-8-sig")
    last = rows[-1]
    print(f"점검 {status['checked_at_kst']} KST (뉴욕 {status['checked_at_ny']}), 완료 종가 {status['last_close']['date']} {status['last_close']['close']:.2f}")
    print(f"BB-1 {'켜짐' if last['BB1'] else '꺼짐'} (관찰일 {last['관찰일']}, 낙폭 {last['낙폭60']:.1%}, SOXL 비중 백분위 {last['SOXL비중백분위']}, 수급 원거래일 {last['수급원거래일']})")
    print(f"강한 조합 겹침 {last['강한조합겹침']}개 → 과거 같은 구간 벨트10 {rates[last['겹침구간']]['belt10']:.0%}")
    print(f"바닥 감지기 {det.get('kind')}, 저점 {det.get('low_date')} {low:.2f}, 저점 대비 {det.get('up', float('nan')):+.1%}, 벨트10 선 {low * 1.10:.2f}")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true", help="이미 받은 now/data로 다시 계산만 한다")
    a = ap.parse_args()
    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    today_kst = pd.Timestamp(now_kst.date())
    if today_kst > CAL_END:
        raise SystemExit("달력이 2027-03-01에서 끝난다. seibro_bottoms/code/build.py·seibro_flows/code/calendar_map.py의 end를 늘린 뒤 실행")
    NOW.mkdir(parents=True, exist_ok=True)
    got = {}
    if not a.skip_fetch:
        prepare(today_kst)
        got = fetch(today_kst, pd.Timestamp(now_ny.date()))
    px = pd.read_csv(NOW / "data" / "prices.csv", parse_dates=["date"])
    source_end = px[px.ticker == "SOXX"]["date"].max()
    panel = build(source_end)
    wide, atoms = atoms_now(panel)
    report(panel, wide, atoms, got, now_kst, now_ny)


if __name__ == "__main__":
    main()
