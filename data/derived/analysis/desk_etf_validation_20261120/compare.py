"""Compare the three frozen desk ledgers with each Korean issuer's actual market closes."""
import argparse
import json
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent / "finviz_recipes_20261120"
TIME_URL = "https://timeetf.co.kr/m11_view.php"
KOACT_URL = "https://www.samsungactive.co.kr/api/v1/product/etf-earning/2ETFQ1.do"
KOACT_FACTS = "https://www.samsungactive.co.kr/sheet/20260805/2ETFQ1_20260731.pdf"
AS_OF_KRX = pd.Timestamp("2026-11-19")
ANCHORS = {"ytd": "2025-12-31", "h2": "2026-06-30"}
LABELS = {
    "balance/monthly1": "재무 강화 FINVIZ",
    "original/monthly1": "기존 FINVIZ",
    "base": "기존 데스크",
    "time": "TIME",
    "koact": "KoAct",
}


def download_quotes():
    # The issuer pages label both fields as the market close. NAV is in a separate column.
    r = requests.get(TIME_URL, params={"idx": "6", "navStartDate": "2025-12-29",
                                      "navEndDate": AS_OF_KRX.date().isoformat()}, timeout=30)
    r.raise_for_status()
    doc = BeautifulSoup(r.content, "html.parser")
    table = next(t for t in doc.select("table") if "시장가격(원)" in t.get_text(strip=True)[:250])
    rows = {}
    for tr in table.select("tbody tr"):
        date, _, _, close, _, _ = [cell.get_text(strip=True) for cell in tr.select("td")]
        rows[pd.Timestamp(date.replace(".", "-"))] = float(close.replace(",", ""))
    r = requests.get(KOACT_URL, params={"graphTerm": 12}, timeout=30)
    r.raise_for_status()
    koact = pd.Series({pd.Timestamp(rec["GIJUN_YMD"]): float(rec["PRICE"]) for rec in r.json()})
    kr = pd.DataFrame({"time": pd.Series(rows), "koact": koact}).sort_index()
    kr = kr.loc["2025-12-29":AS_OF_KRX]
    # KRX closed on the 2025 year-end holiday. TIME's NAV table repeats the prior
    # market close on that date even though no transaction occurred.
    kr = kr.drop(index=pd.Timestamp("2025-12-31"), errors="ignore")
    if kr.empty or kr.index[-1] != AS_OF_KRX or kr.isna().any().any() or not (kr > 0).all().all():
        raise ValueError("Unmatched issuer dates or missing historical quotes; keep the frozen CSV")
    kr.index.name = "krx_date"
    kr.astype(int).to_csv(HERE / "issuer_closes.csv", lineterminator="\n")


def daily_ledgers():
    src = pd.read_csv(RESEARCH / "selected_daily.csv", parse_dates=["date"])
    strategies = src.pivot(index="date", columns="key", values="krw10")[list(LABELS)[:3]]
    initial = pd.DataFrame(1., index=pd.to_datetime([ANCHORS["ytd"]]), columns=strategies.columns)
    mark = pd.read_csv(RESEARCH / "forward_mark.csv").set_index("key").loc[strategies.columns]
    if not mark.scheduled_hold.all() or mark.mark_date.nunique() != 1:
        raise ValueError("Only the actual scheduled-hold extension can be marked")
    later = pd.DataFrame([1 + mark.marked_ytd], index=pd.to_datetime([mark.mark_date.iat[0]]))
    strategies = pd.concat([initial, strategies, later]).sort_index()
    if strategies.isna().any().any() or not strategies.index.is_unique:
        raise ValueError("Incomplete or ambiguous strategy observations")
    return strategies


def period_stats(series):
    return float(series.iat[-1] / series.iat[0] - 1), float((series / series.cummax() - 1).min())


def compare():
    kr = pd.read_csv(HERE / "issuer_closes.csv", index_col="krx_date", parse_dates=True)
    us = daily_ledgers()
    # US t closes in the Korean morning following US calendar day t, before the next KRX close.
    # Truncate to full pairs. An unclosed current KRX session must never price yesterday's US mark.
    found = kr.index.searchsorted(us.index, side="right")
    arrived = found < len(kr)
    paired = us.loc[arrived].copy()
    paired.index.name = "us_date"
    paired.insert(0, "krx_date", kr.index[found[arrived]])
    for key in kr:
        paired[key] = kr[key].iloc[found[arrived]].to_numpy()
    if not (paired.index < pd.DatetimeIndex(paired.krx_date)).all():
        raise ValueError("Same-day or earlier Korean close matched to a US session")
    paired.to_csv(HERE / "paired_daily.csv", float_format="%.12g", lineterminator="\n")

    end_us, end_kr = paired.index[-1], paired.krx_date.iat[-1]
    metrics = {"through_us": str(end_us.date()), "through_krx": str(end_kr.date()), "rows": []}
    records = []
    for key, label in LABELS.items():
        row = {"label": label}
        for period, anchor in ANCHORS.items():
            start_us = pd.Timestamp(anchor)
            start_kr = paired.at[start_us, "krx_date"]
            # MDD includes every published own-market close between the common endpoints.
            series = kr.loc[start_kr:end_kr, key] if key in kr else us.loc[start_us:end_us, key]
            ret, dd = period_stats(series)
            assert abs(ret - (paired[key].iat[-1] / paired.at[start_us, key] - 1)) < 1e-10
            row[period] = ret
            records.append(dict(key=key, period=period, return_krw=ret, max_drawdown=dd,
                                us_anchor=str(start_us.date()), krx_anchor=str(start_kr.date()),
                                initial=series.iat[0], final=series.iat[-1], observations=len(series)))
        metrics["rows"].append(row)
    summary = pd.DataFrame(records)
    summary.to_csv(HERE / "performance.csv", index=False, float_format="%.12g", lineterminator="\n")
    (HERE / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    native = {}
    for period, anchor in ANCHORS.items():
        native_start = kr.index[kr.index <= pd.Timestamp(anchor)][-1]
        native[period] = {k: period_stats(kr.loc[native_start:end_kr, k])[0] for k in kr}
    def pct(x):
        return f"{x:+.2%}".replace("-", "−")
    price_dates = [paired.at[pd.Timestamp(anchor), "krx_date"] for anchor in ANCHORS.values()] + [end_kr]
    endpoints = {k: "·".join(f"{v:,}원" for v in kr.loc[price_dates, k]) for k in kr}
    out = ["# 내 계좌와 두 ETF의 연중·하반기 비교", "",
           f'미국 {metrics["through_us"]} 종가와 그 후 첫 한국거래소 {metrics["through_krx"]} 종가까지 원화로 비교했다. 11월 20일 한국장 마감 전 확인한 자료다. 미국 11월 19일의 전략 평가액은 비교할 한국 종가가 아직 없어 제외했다.', "",
           "| 전략·ETF | 연초 이후 | 7월 이후 | 연중 최대낙폭 | 7월 이후 최대낙폭 |",
           "|---|---:|---:|---:|---:|"]
    detail = summary.set_index(["key", "period"])
    for key, label in LABELS.items():
        cells = [pct(detail.at[(key, p), c]) for c in ("return_krw", "max_drawdown") for p in ANCHORS]
        out.append("| " + " | ".join([label, *cells]) + " |")
    out += ["", "## 어떤 종가를 맞췄는가", "",
            f'미국 2025년 12월 31일·2026년 6월 30일 마감 뒤 처음 형성된 한국 종가는 각각 1월 2일·7월 1일이다. 미국 1월 2일과 7월 1일의 전략 수익부터 계산했다. 세 기준일의 ETF 종가는 TIME {endpoints["time"]}, KoAct {endpoints["koact"]}이다. 최대낙폭에는 이 시작·종료 사이 각 시장의 모든 정규장 종가를 넣었다. 한국 휴장으로 여러 미국 거래일에 동일한 다음 한국 종가가 대응할 수 있다. 날짜 대응은 각 기간을 확정하는 용도다.', "",
            f'한국 달력으로 전년 말(12월 30일)과 6월 말(6월 30일)부터 11월 19일까지 계산하면 TIME은 각각 {pct(native["ytd"]["time"])}, {pct(native["h2"]["time"])}이고 KoAct는 {pct(native["ytd"]["koact"])}, {pct(native["h2"]["koact"])}이다. 위 표는 미국 전략의 기준 시점에 맞춘 수익이다.', "",
            "ETF는 운용사가 게시한 거래소 시장 종가이며 펀드 보수는 가격에 반영돼 있다. 각 ETF를 기간 시작부터 보유한 가격 수익을 사용했다. ETF 거래비용은 따로 차감하지 않았다. 전략은 달러 주식과 현금을 매일 원화로 환산한 원장에 가상 체결비용 편도 0.10%를 적용했다. 전략과 ETF 모두 분배금·배당금 재투자, 세금을 제외했다. ETF는 이미 원화로 거래돼 추가 환산이 필요 없다. 이 모형은 170개 조합을 본 뒤 고른 것이며 실제 개인 계좌 수익과 구분한다.", "",
            "TIME 글로벌AI인공지능액티브의 거래코드는 456600이고 2023년 5월 16일 상장했다. [TIME 운용사 시장가격과 ETF 정보](https://timeetf.co.kr/m11_view.php?idx=6&navStartDate=2025-12-29&navEndDate=2026-11-19)", "",
            f"KoAct 미국나스닥성장기업액티브는 거래코드 0015B0, 운용사 ID 2ETFQ1이고 2025년 2월 25일 상장했다. 두 상품 모두 이번 비교 구간 전체에서 거래됐다. [KoAct 운용사 시장가격](https://www.samsungactive.co.kr/etf/view.do?id=2ETFQ1) · [KoAct 공식 팩트시트]({KOACT_FACTS})", "",
            "compare.py를 실행하면 저장한 운용사 종가와 이전 연구 원장에서 표를 재현한다. --refresh는 두 운용사에 공개된 원천의 같은 가격 구간을 다시 내려받는다. [시장 종가](issuer_closes.csv) · [날짜 대응](paired_daily.csv) · [수익·낙폭 원장](performance.csv)"]
    (HERE / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--refresh", action="store_true")
    if arg.parse_args().refresh:
        download_quotes()
    print(json.dumps(compare(), ensure_ascii=False, indent=2))
