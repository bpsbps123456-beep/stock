"""SEIBRO 종목별내역(주식 TOP50) 미국 수집기.

원천: https://seibro.or.kr (메뉴 921, BIP_CNTS10013V). 화면과 같은 요청을 보낸다.
- 결제금액(S_TYPE=2): 기간 [START_DT, END_DT] 합계 Top50. D_TYPE 1 매수, 2 매도, 3 매수+매도, 4 순매수 순위.
  응답에는 순위 기준과 무관하게 매수·매도·합계·순매수 네 금액이 다 온다. 단위 USD.
- 보관금액(S_TYPE=1): 하루 잔고 Top50. 단위 USD.
- 쪽 번호(PG_START/PG_END)는 서버가 무시한다. 언제나 1~50위만 온다(2026-10-26 확인).

날짜 기준(2026-10-26 확인, README 참조): SEIBRO 결제 날짜 D = 미국 결제일 + 1일(주말이면 월요일).
그래서 미국 T+1(2024-05-28 거래분부터) 기간에는 미국 월요일 거래가 SEIBRO 수요일에 찍힌다.
주간 창은 미국 거래 주(월~금)에 맞춰 SEIBRO 날짜로 바꿔 조회한다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
URL = "https://seibro.or.kr/websquare/engine/proworks/callServletService.jsp"
PAGE = "https://seibro.or.kr/websquare/control.jsp?w2xPath=/IPORTAL/user/ovsSec/BIP_CNTS10013V.xml&menuNo=921"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
TASK = "ksd.safe.bip.cnts.OvsSec.process.OvsSecIsinPTask"
BTN = "allview,allview,print,hwp,word,pdf,searchIcon,seach,xls,link,link,wide,wide,top,"
T1_START = dt.date(2024, 5, 28)  # 미국 T+1 결제 시작 거래일

ROW_RE = re.compile(r"<result>(.*?)</result>", re.S)
FIELD_RE = re.compile(r'<(\w+) value="([^"]*)"\s*/>')
WARN_RE = re.compile(r"<WARNING>.*?value='([^']*)'", re.S)


class SeibroError(RuntimeError):
    pass


def body(kind: str, start: str, end: str, dtype: str | None) -> str:
    if kind == "custody":
        action, s_type, extra = "getImptFrcurStkCusRemaList", "1", ""
    else:
        action, s_type, extra = "getImptFrcurStkSetlAmtList", "2", f'<D_TYPE value="{dtype}"/>'
    return (
        f'<reqParam action="{action}" task="{TASK}"><MENU_NO value="921"/>'
        f'<CMM_BTN_ABBR_NM value="{BTN}"/><W2XPATH value="/IPORTAL/user/ovsSec/BIP_CNTS10013V.xml"/>'
        f'<PG_START value="1"/><PG_END value="50"/><START_DT value="{start}"/><END_DT value="{end}"/>'
        f'<S_TYPE value="{s_type}"/><S_COUNTRY value="US"/>{extra}</reqParam>'
    )


def parse(text: str, kind: str) -> list[dict]:
    warn = WARN_RE.search(text)
    if warn:
        raise SeibroError(warn.group(1))
    if "<vector" not in text:
        raise SeibroError("응답 형식이 다름: " + text[:200])
    rows = []
    for chunk in ROW_RE.findall(text):
        f = {k: html.unescape(v) for k, v in FIELD_RE.findall(chunk)}
        row = {"rank": int(f["RNUM"]), "isin": f["ISIN"], "name": f["KOR_SECN_NM"]}
        if kind == "custody":
            row["amt"] = float(f["SUM_FRSEC_AMT"])
        else:
            row.update(
                buy=float(f["SUM_FRSEC_BUY_AMT"]),
                sell=float(f["SUM_FRSEC_SELL_AMT"]),
                tot=float(f["SUM_FRSEC_TOT_AMT"]),
                net=float(f["SUM_FRSEC_NET_BUY_AMT"]),
            )
        rows.append(row)
    return rows


class Client:
    def __init__(self, gap: float):
        self.gap = max(gap, 0.8)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Referer": PAGE})
        self.last = 0.0
        self.s.get(PAGE, timeout=30)

    def fetch(self, kind: str, start: str, end: str, dtype: str | None) -> list[dict]:
        wait = self.gap - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.s.post(
                URL,
                data=body(kind, start, end, dtype).encode("utf-8"),
                headers={"Content-Type": "application/xml; charset=UTF-8"},
                timeout=60,
            )
        finally:
            self.last = time.monotonic()
        r.raise_for_status()
        return parse(r.content.decode("utf-8", "replace"), kind)


def weekdays(since: dt.date, until: dt.date):
    d = since
    while d <= until:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def seibro_date(trade: dt.date, sessions: list[dt.date]) -> dt.date:
    """미국 거래일 → SEIBRO 결제 날짜(결제일 + 1일, 주말이면 월요일)."""
    i = sessions.index(trade)
    settle = sessions[i + (1 if trade >= T1_START else 2)]
    d = settle + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def us_weeks(since: dt.date, until: dt.date) -> list[dict]:
    """미국 거래 주마다 그 주 거래를 담는 SEIBRO 날짜 창. 창은 서로 겹치지 않고 빈틈없이 이어진다."""
    import exchange_calendars as xc

    cal = xc.get_calendar("XNYS", start="2022-12-01", end="2027-03-01")
    sessions = [s.date() for s in cal.sessions]
    weeks: dict[tuple, list[dt.date]] = {}
    for s in sessions:
        if since - dt.timedelta(days=7) <= s <= until + dt.timedelta(days=14):
            weeks.setdefault(tuple(s.isocalendar()[:2]), []).append(s)
    keys = sorted(weeks)
    out = []
    for k, nxt in zip(keys, keys[1:]):
        first = weeks[k][0]
        if first < since:
            continue
        start = seibro_date(first, sessions)
        end = seibro_date(weeks[nxt][0], sessions) - dt.timedelta(days=1)
        if end > until:
            break
        out.append({"week": f"{k[0]}-W{k[1]:02d}", "trade_first": first, "trade_last": weeks[k][-1],
                    "start": start, "end": end})
    return out


def path_for(kind: str, dtype: str | None, start: dt.date, end: dt.date) -> Path:
    sub = RAW / kind / (f"d{dtype}" if dtype else "amt")
    return sub / f"{start:%Y%m%d}_{end:%Y%m%d}.json"


def plan(jobs: list[str], since: dt.date, until_settle: dt.date, until_custody: dt.date):
    tasks = []
    if "weekly" in jobs:
        for w in us_weeks(since, until_settle):
            for t in "4123":
                tasks.append(("weekly", t, w["start"], w["end"], {k: str(v) for k, v in w.items()}))
    if "custody" in jobs:
        for d in weekdays(since, until_custody):
            tasks.append(("custody", None, d, d, {}))
    if "daily" in jobs:
        for d in weekdays(since, until_settle):
            for t in "4123":
                tasks.append(("daily", t, d, d, {}))
    return tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="weekly,custody,daily")
    ap.add_argument("--since", default="2023-01-02")
    ap.add_argument("--until-settle", default=None, help="결제 마지막 날짜(기본: 어제 이전 평일)")
    ap.add_argument("--until-custody", default=None, help="보관 마지막 날짜(기본: 2평일 전)")
    ap.add_argument("--gap", type=float, default=1.0)
    a = ap.parse_args(argv)

    today = dt.date.today()
    back = lambda n: [d for d in weekdays(today - dt.timedelta(days=10), today - dt.timedelta(days=1))][-n]
    since = dt.date.fromisoformat(a.since)
    until_s = dt.date.fromisoformat(a.until_settle) if a.until_settle else back(1)
    until_c = dt.date.fromisoformat(a.until_custody) if a.until_custody else back(2)
    tasks = plan(a.jobs.split(","), since, until_s, until_c)
    todo = [t for t in tasks if not path_for(t[0], t[1], t[2], t[3]).exists()]
    print(f"[plan] 전체 {len(tasks)}건, 남은 {len(todo)}건 (결제 ~{until_s}, 보관 ~{until_c})", flush=True)

    client = Client(a.gap)
    fails = 0
    for i, (kind, dtype, start, end, meta) in enumerate(todo, 1):
        out = path_for(kind, dtype, start, end)
        if out.exists():  # 다른 수집 프로세스가 먼저 받음
            continue
        for attempt in range(3):
            try:
                rows = client.fetch(kind, f"{start:%Y%m%d}", f"{end:%Y%m%d}", dtype)
                break
            except (requests.RequestException, SeibroError) as e:
                print(f"[retry] {kind} d{dtype} {start}~{end}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
                if attempt == 1:
                    client = Client(a.gap)  # 세션 새로 받기
        else:
            fails += 1
            if fails >= 3:
                print("[stop] 연속 3건 실패. 받은 것까지 저장하고 멈춘다.", flush=True)
                return 2
            continue
        fails = 0
        out.parent.mkdir(parents=True, exist_ok=True)
        rec = {"kind": kind, "dtype": dtype, "start": str(start), "end": str(end), "meta": meta,
               "fetched_at": dt.datetime.now().isoformat(timespec="seconds"), "n": len(rows), "rows": rows}
        out.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        if i % 50 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {kind} d{dtype} {start}~{end} n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
