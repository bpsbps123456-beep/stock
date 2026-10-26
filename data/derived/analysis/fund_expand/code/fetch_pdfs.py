"""KoAct 다른 펀드 3개(바이오헬스케어 2ETFS9, 치매&뇌질환 2ETFO5, 로봇피지컬AI 2ETFU7)의 2026 일별 PDF를 받는다.

연구 전용(2026-10-12 사용자 요청 '다른 펀드 종목도 넣고 돌렸을 때 수익률 향상?'). 운영 수집(gbrc)과 정규화 스냅샷
(data/normalized)은 건드리지 않고 이 폴더(git 제외)에만 저장한다. 날짜는 2ETFQ1 검증 스냅샷이 있는 KRX 영업일 중
주마다 마지막 날(주간 판단에 쓰는 보유 기준과 같다). 운용사 서버가 연속 요청을 429로 막으므로 요청 사이 4초,
막히면 90초·180초…를 기다렸다가 다시 시도한다.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from gbrc.etf.providers import parse_samsung_koact  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "pdfs"
FUNDS = {"2ETFS9": "KoAct 미국바이오헬스케어액티브", "2ETFO5": "KoAct 미국치매&뇌질환치료제액티브",
         "2ETFU7": "KoAct 미국로봇피지컬AI액티브"}
BASE = "https://www.samsungactive.co.kr"


def get(fid: str, ymd: str):
    url = f"{BASE}/api/v1/product/etf-pdf/{fid}.do?gijunYMD={ymd}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": f"{BASE}/etf/view.do?id={fid}"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            if text.strip():
                return url, json.loads(text)
        except Exception as exc:
            print(fid, ymd, "재시도", attempt + 1, type(exc).__name__, getattr(exc, "code", ""), flush=True)
        time.sleep(90 * (attempt + 1))
    return url, None


def main():
    import pandas as pd
    all_dates = sorted(p.name[:10] for p in (ROOT / "data" / "normalized" / "etf_holdings").glob("2026-*_2ETFQ1.json"))
    s = pd.Series(all_dates, index=pd.to_datetime(all_dates))
    dates = list(s.groupby(s.index.to_period("W-SUN")).max())  # 주마다 마지막 영업일
    for fid, name in FUNDS.items():
        out = OUT / fid
        out.mkdir(parents=True, exist_ok=True)
        fund = SimpleNamespace(provider="samsung_koact", provider_id=fid, source_url=BASE, legacy_idx=None,
                               snapshot_ticker=fid, display_name=name)
        for d in dates:
            path = out / f"{d}.json"
            if path.exists():
                continue
            url, data = get(fid, d.replace("-", ""))
            if data is None:
                print(fid, d, "빈 응답", flush=True)
                continue
            snap = parse_samsung_koact(data, fund, d, url)
            snap.pop("_raw_payload", None)
            path.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
            time.sleep(4.0)
        print(fid, "완료", len(list(out.glob("*.json"))), flush=True)


if __name__ == "__main__":
    main()
