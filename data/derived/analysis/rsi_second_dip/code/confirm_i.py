"""봉 안(I) 방식을 '5분봉 종가로 확인된 닿음'으로만 다시 재기(2026-10-10, 키 rsi_second_dip).

질문: 계좌 시험에서 'I K2 + 깨어 있을 때 + 매수 체결 확인' 판만 두 기간 모두 QQQ·단순 급락 매수를 이겼다. 그런데 매도(RSI 70 가격
  지정가)는 확인하지 않았다(확인 63%). 매수·매도 둘 다 실제 거래로 확인되는 경우만 인정하면 남는가.
규칙: study.py의 I와 같되, 40분봉 고가·저가 대신 그 봉 안 5분봉 종가의 최고·최저를 쓴다(5분봉 종가가 RSI 30 가격 이하로
  거래돼야 매수, RSI 70 가격 이상으로 거래돼야 매도). 체결가는 min(시가, p30)·max(시가, p70) 그대로.
  깨어 있을 때 판: 한국 00~08시 봉에서는 RSI 30 닿음을 보지 않는다(세지도 사지도 않음). 매도는 시각 제한 없음
  (자기 전에 그때의 RSI 70 가격에 지정가 매도를 건다고 봄; 그 가격은 봉마다 조금씩 바뀌므로 근사).
비교: 같은 묶음의 C(봉 마감 확정)와 원래 I. 급락주 표시·기간·종목 묶음은 study.py와 같다(40m만, 1h는 5분봉이 없어 못 함).
사용: py confirm_i.py → results/confirm_i.txt
"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study as S


def bars_40m_conf(s, folder):
    """40분봉 + 그 봉 안 5분봉 종가 최고·최저."""
    m5 = pd.read_parquet(os.path.join(folder, f"5m_{s}.parquet"))
    m5 = m5[m5.High > 0]
    g = m5.resample("40min", origin="start_day", offset="4h", label="left", closed="left")
    d = g.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    d["cmax"], d["cmin"] = g.Close.max(), g.Close.min()
    return d.dropna(subset=["Close"])


def run(syms, folder, crash_since):
    rows = []
    for s in syms:
        try:
            d = bars_40m_conf(s, folder)
        except (FileNotFoundError, OSError):
            continue
        if len(d) < 60 or s in ("SPY", "QQQ"):
            continue
        A = S.prep(d, 14, 40, crash_since)
        kst_h = A["idx"].tz_convert("Asia/Seoul").hour
        awake = np.asarray(kst_h >= 8)                     # 봉 시작이 한국 08~24시
        hc, lc = d.cmax.values.astype(float), d.cmin.values.astype(float)
        for K in (1, 2, 3):
            for nm, (hh, ll, ok) in {"I 원래": (A["h"], A["l"], None), "I 5분종가확인": (hc, lc, None),
                                      "I 5분종가확인+깨어매수": (hc, lc, awake)}.items():
                l_use = ll.copy()
                if ok is not None:
                    l_use = np.where(ok, ll, np.inf)             # 자는 시간 봉에서는 매수 이벤트가 안 일어난다고 봄
                S_, E_, EP_, X_, XP_, F1_ = S.machine(A["o"], hh, l_use, A["c"], A["r"], A["p30"], A["p70"], A["tday"],
                                                      K, 30.0, 70.0, 0, True)
                for j in range(len(S_)):
                    sb = S_[j]
                    rows.append((nm, K, s, A["day"][sb], XP_[j] / EP_[j] - 1, X_[j] < 0, bool(A["crash_I"][sb])))
            S_, E_, EP_, X_, XP_, F1_ = S.machine(A["o"], A["h"], A["l"], A["c"], A["r"], A["p30"], A["p70"], A["tday"],
                                                  K, 30.0, 70.0, 0, False)
            for j in range(len(S_)):
                sb = S_[j]
                rows.append(("C", K, s, A["day"][sb], XP_[j] / EP_[j] - 1, X_[j] < 0, bool(A["crash_C"][sb])))
    return pd.DataFrame(rows, columns=["규칙", "K", "sym", "day", "ret", "open", "crash"])


def main():
    t1 = run(S.universe("40m"), S.B5M, date(2026, 8, 11))
    t1 = t1[t1.crash & (t1.day >= date(2026, 8, 12))]
    t1["묶음"] = "급락987·급락주(8/12~)"
    t2 = run(S.ai137(), S.B5M_AI, None)
    t2["묶음"] = "AI137 전체"
    t = pd.concat([t1, t2])
    t["기간"] = np.where(t.day < date(2026, 9, 1), "앞 ~8/31", "뒤 9/1~10/9")
    t = t[t.day >= date(2026, 7, 24)]
    rows = []
    for (b, K, nm, p), x in t.groupby(["묶음", "K", "규칙", "기간"]):
        dm = x.groupby("day").ret.mean()
        se = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else np.nan
        rows.append(dict(묶음=b, K=K, 규칙=nm, 기간=p, 건수=len(x), 평균=f"{x.ret.mean():+.2%}", 중앙=f"{x.ret.median():+.2%}",
                         승률=f"{(x.ret > 0).mean():.0%}", 최악=f"{x.ret.min():+.1%}", t=f"{dm.mean() / se:+.2f}" if se else "",
                         미청산=f"{x.open.mean():.0%}"))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    txt = ("I 방식을 5분봉 종가로 확인된 닿음만 인정해 다시 잼 (40분봉 RSI14, 2026-10-10, confirm_i.py)\n"
           "I 원래 = 40분봉 고가·저가로 닿음 판정, I 5분종가확인 = 그 봉 안 5분봉 종가 최고·최저로 판정(매수·매도 둘 다),\n"
           "+깨어매수 = 매수 이벤트는 한국 08~24시 봉에서만. C = 봉 마감 확정(비교). 미청산은 10/9 마지막 종가 평가.\n\n" + out.to_string(index=False))
    with open(os.path.join(S.ROOT, "results", "confirm_i.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
