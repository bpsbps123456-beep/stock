# -*- coding: utf-8 -*-
"""OFR Hedge Fund Monitor 공개 API에서 헤지펀드 레버리지 시계열을 받는다.

API 루트: https://data.financialresearch.gov/hf/v1/
  - metadata/mnemonics            : 전체 시리즈 이름 목록
  - metadata/query?mnemonic=X     : 시리즈 설명(출처·빈도·단위)
  - series/full?mnemonic=X        : 시계열 [[date, value], ...]
데이터셋: fpf(SEC Form PF, 분기), tff(CFTC 금융선물 포지션, 주간), ficc(스폰서드 레포, 월간), scoos(연준 딜러 설문, 분기)

결과: results/ofr_hfm_series.csv (mnemonic, date, value) + results/ofr_hfm_meta.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
BASE = "https://data.financialresearch.gov/hf/v1"
UA = {"User-Agent": "Mozilla/5.0"}

# 레버리지·규모 핵심 시리즈 (이름은 mnemonics 목록에서 골랐다)
WANT_SUBSTR = [
    "LEVERAGE",           # GAV/NAV·GNE/NAV 레버리지 비율, SCOOS 레버리지 사용
    "ALLQHF_GNE_SUM", "ALLQHF_GAV_SUM", "ALLQHF_NAV_SUM", "ALLQHF_COUNT",  # 규모
    "FPF-BORROW_",        # 프라임브로커·레포·기타 담보 차입 (분기)
    "FICC-SPONSORED",     # 스폰서드 레포 (월간)
    "SCOOS-NET_HF",       # 딜러 설문: 헤지펀드 신용 조건 (분기)
    "TFF-LF_", "TFF-AI_", # CFTC 레버리지 펀드·자산운용사 선물 포지션 (주간)
]


def get(path: str, **params):
    r = requests.get(f"{BASE}/{path}", params=params, headers=UA, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    mnemonics = get("metadata/mnemonics")
    print(f"총 시리즈 수: {len(mnemonics)}", file=sys.stderr)
    picked = [m for m in mnemonics if any(s in m for s in WANT_SUBSTR)]
    # TFF는 NET_POSITION만(롱·숏 개별은 생략), 전략별은 LEVERAGERATIO만
    picked = [m for m in picked if not (m.startswith("TFF-") and "NET" not in m)]
    print(f"받을 시리즈 수: {len(picked)}", file=sys.stderr)

    meta = {}
    rows = []
    for m in picked:
        try:
            md = get("metadata/query", mnemonic=m)
            meta[m] = md
        except Exception as e:  # noqa: BLE001
            meta[m] = {"error": str(e)}
        try:
            s = get("series/full", mnemonic=m)
        except Exception as e:  # noqa: BLE001
            print(m, "series ERR", e, file=sys.stderr)
            continue
        # 응답 형태: {mnemonic: {"timeseries": {"aggregation": [[date, value], ...]}}}
        data = []
        if isinstance(s, dict):
            inner = s.get(m) or next(iter(s.values()), {})
            data = (inner.get("timeseries") or {}).get("aggregation") or []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                rows.append({"mnemonic": m, "date": item[0], "value": item[1]})
    with (RESULTS / "ofr_hfm_series.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["mnemonic", "date", "value"])
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "ofr_hfm_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved rows={len(rows)} series={len(picked)}", file=sys.stderr)
    # 시리즈별 마지막 관측일 요약
    last = {}
    for r in rows:
        if r["date"] and (r["mnemonic"] not in last or r["date"] > last[r["mnemonic"]][0]):
            last[r["mnemonic"]] = (r["date"], r["value"])
    for m, (d, v) in sorted(last.items()):
        print(f"{m}\t{d}\t{v}")


if __name__ == "__main__":
    main()
