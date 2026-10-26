"""사용자가 브라우저로 받은 삼성액티브 PDF 엑셀(.xls)을 비교 코드 형식(pdfs/<펀드>/<날짜>.json)으로 바꾼다.

파일: manual_xls/KoAct <펀드명>_<YYYYMMDD>.xls (2026-10-12 사용자가 월말 기준으로 29개 받음).
형식: 1행 제목, 2행 기준일(YYYY/MM/DD), 3행 머리글(번호·종목명·ISIN·종목코드·수량·비중(%)·평가금액(원)), 그 아래 종목.
설정현금액(CASH00000001)은 구성종목이 아니라 빼고, 원화현금(KRD010010001)은 현금으로 둔다(운영 파서와 같은 규칙).
파일 이름의 날짜와 2행 기준일이 다르면 멈춘다.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from gbrc.etf.providers import SAMSUNG_CASH_ROW, SAMSUNG_KRW_DEPOSIT, normalize_samsung_code  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "manual_xls"
OUT = HERE.parent / "pdfs"
FUNDS = {"KoAct 미국바이오헬스케어액티브": "2ETFS9", "KoAct 미국치매&뇌질환치료제액티브": "2ETFO5",
         "KoAct 미국로봇피지컬AI액티브": "2ETFU7"}


def parse(path: Path) -> tuple[str, str, list[dict]]:
    m = re.match(r"(.+)_(\d{8})\.xls$", path.name)
    if not m or m.group(1) not in FUNDS:
        raise ValueError(f"모르는 파일 이름: {path.name}")
    fid, ymd = FUNDS[m.group(1)], m.group(2)
    df = pd.read_excel(path, header=None, engine="xlrd")
    stated = str(df.iat[1, 0]).strip().replace("/", "")
    if stated != ymd:
        raise ValueError(f"{path.name}: 파일 안 기준일 {df.iat[1, 0]}이 이름의 날짜와 다릅니다")
    head = [str(v).strip() for v in df.iloc[2]]
    rows = df.iloc[3:].copy()
    rows.columns = head
    rows = rows[rows["종목코드"].notna()]
    rows = rows[rows["종목코드"].astype(str).str.strip() != SAMSUNG_CASH_ROW]
    value = pd.to_numeric(rows["평가금액(원)"], errors="coerce").fillna(0.0)
    total = float(value.sum())
    holdings, cash = [], 0.0
    for (_, r), v in zip(rows.iterrows(), value):
        code = str(r["종목코드"]).strip()
        if code == SAMSUNG_KRW_DEPOSIT:
            cash += v
            continue
        holdings.append({"code": normalize_samsung_code(code), "name": str(r["종목명"]).strip(),
                         "isin": str(r["ISIN"]).strip(), "qty": float(pd.to_numeric(r["수량"], errors="coerce") or 0),
                         "value_krw": v, "weight": v / total * 100.0 if total else 0.0})
    if cash:
        holdings.append({"code": "", "name": "Cash", "value_krw": cash, "weight": cash / total * 100.0 if total else 0.0})
    return fid, f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}", sorted(holdings, key=lambda h: -h["weight"])


def main():
    summary = {}
    for path in sorted(SRC.glob("*.xls")):
        fid, day, holdings = parse(path)
        n_us = sum(1 for h in holdings if h["code"].endswith(" US EQUITY"))
        if not holdings:
            print(fid, day, "빈 파일 — 건너뜀")
            continue
        (OUT / fid).mkdir(parents=True, exist_ok=True)
        (OUT / fid / f"{day}.json").write_text(json.dumps({"date": day, "fund": fid, "source": path.name,
                                                           "holdings": holdings}, ensure_ascii=False), encoding="utf-8")
        summary.setdefault(fid, []).append((day, len(holdings), n_us))
    for fid, rows in summary.items():
        print(fid, " ".join(f"{d[5:]}:{n}/{u}" for d, n, u in rows))


if __name__ == "__main__":
    main()
