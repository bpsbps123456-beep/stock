"""지금 상태 갱신: 1차 과제 A의 수집기·신호판 코드를 그대로 쓰되 새 폴더(belt/now)에만 쓴다.

동결된 1차 자료(bottom_zone_20261111/data)는 읽기만 한다. SEIBRO 원응답·캐시를 복사한 뒤
최근 빈 응답만 다시 받고, 가격·CNN을 새로 받아 신호판과 조건 49개를 다시 만든다.

실행: py data/derived/analysis/bottom_zone_20261111/belt/code/now_update.py [--skip-fetch]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ZONE = HERE.parents[1]                       # bottom_zone_20261111
ANALYSIS = ZONE.parent
NOW = ZONE / "belt" / "now"
REFETCH_FROM = pd.Timestamp("2026-11-09")    # 이 날짜 이후 빈 응답은 다시 받는다
FETCH_TO = pd.Timestamp("2026-11-12")


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def prepare():
    for sub in ("seibro", "cache"):
        shutil.copytree(ZONE / "data" / sub, NOW / "data" / sub, dirs_exist_ok=True)
    for f in (NOW / "data" / "cache").glob("*.json"):
        day = pd.Timestamp(f.stem.split("_")[-1])
        if day >= REFETCH_FROM and json.loads(f.read_text(encoding="utf-8"))["n"] == 0:
            f.unlink()


def fetch():
    col = module("now_collect", ZONE / "code" / "collect_A.py")
    col.ROOT = NOW
    col.market()
    got = {}
    for day in pd.bdate_range(REFETCH_FROM, FETCH_TO):
        for kind, dtype in [("daily", v) for v in "1234"] + [("custody", None)]:
            r = col.fetch((day, kind, dtype))
            got[f"{kind}_{dtype or 'amt'}_{day:%Y%m%d}"] = r["n"]
    return got


def build():
    pa = module("now_panel", ZONE / "code" / "panel_A.py")
    original_load = pa.load

    def load(name, path):                    # NOW 기준 상대 경로를 원래 연구 위치로 돌린다
        p = Path(path)
        if not p.exists() and "seibro_flows" in p.parts:
            p = ANALYSIS / "seibro_flows" / "code" / "collect.py"
        return original_load(name, p)

    pa.load = load
    pa.ROOT = NOW
    pa.raw_panel()
    census = ZONE.parents[3] / "research" / "pb_flow_census_20261026" / "census.csv"
    panel = pa.assemble(*pa.inputs(str(census)))
    panel.to_csv(NOW / "data" / "signal_panel.csv", index_label="date")
    return panel


def atoms_now():
    bs = module("now_belt", HERE / "belt_search.py")
    panel = pd.read_csv(NOW / "data" / "signal_panel.csv", parse_dates=["date"]).set_index("date")
    px = pd.read_csv(NOW / "data" / "prices.csv", parse_dates=["date"])
    wide = {t: g.set_index("date").sort_index() for t, g in px.groupby("ticker")}
    cnn = json.loads((NOW / "data" / "cnn.json").read_text(encoding="utf-8"))
    news = pd.read_csv(ZONE / "news" / "v2" / "news_series_B.csv", parse_dates=["date"], encoding="utf-8-sig")
    extra = NOW / "news_extra.csv"            # 동결 계열 뒤에 확인한 의견 변경(잠정)
    if extra.exists():
        news = pd.concat([news, pd.read_csv(extra, parse_dates=["date"], encoding="utf-8-sig")], ignore_index=True)
    atoms = bs.build_atoms(panel, wide, cnn, news)
    atoms.to_csv(NOW / "atoms_now.csv", index_label="date", encoding="utf-8-sig")
    return panel, wide, atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    a = ap.parse_args()
    NOW.mkdir(parents=True, exist_ok=True)
    got = {}
    if not a.skip_fetch:
        prepare()
        got = fetch()
    panel = build()
    panel, wide, atoms = atoms_now()
    last = panel.tail(4)
    summary = dict(
        fetched=got,
        prices_retrieved=json.loads((NOW / "data" / "retrieved.json").read_text(encoding="utf-8")),
        last_close=str(wide["SOXX"].index[-1].date()),
        rows={str(d.date()): dict(open=r.open, close=r.close, dd60=r.dd60, T5_P=r.T5_P, S1_P=r.S1_P,
                                  G=r.G, source_date=str(r.source_date)[:10], published=r.source_published)
              for d, r in last.iterrows()},
        on={str(d.date()): [c for c in atoms.columns if atoms.loc[d, c]] for d in atoms.index[-4:]},
    )
    (NOW / "now_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
