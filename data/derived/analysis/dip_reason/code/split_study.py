"""분할 매수(50%+50%) 연구(2026-10-09 사용자: 프리장에서 한 번 샀으면 본장에서도 한 번 더 살 기회를, 하루 최대 2번).
지금 지표 규칙(EMA x (1-2%) 지정가 매수, RSI14>70 다음 봉 시가 매도, 첫 매수 뒤 처음 오는 아침 8시 정리, 전날 +7% 다음 날 정규장 매수 안 함)에
매수 단위와 횟수 규칙만 바꾼다. 수익은 종목별 자금 대비(단위 비중 x 단위 수익)로 잰다.
  '하루' = 한국시간 아침 8시 ~ 다음 날 아침 8시(8시 정리 주기와 같음)
변형:
  V0 지금 지표: 100% 한 번, 들고 있으면 추가 없음, 하루 횟수 제한 없음
  V0h 50% 한 번(노출 절반 기준)
  V1 사용자안: 50%씩, 들고 있을 때 첫 매수와 다른 세션이면 50% 추가, 하루 최대 2번
  V2 50%씩, 같은 세션 추가도 허용(다음 봉부터 매수선에 다시 닿으면), 하루 최대 2번
  V3 50%씩, 다른 세션 추가, 하루 횟수 제한 없음(보유는 최대 2단위)
  V4 100% 한 번, 하루 최대 2번(횟수 제한만의 효과)
사용: py split_study.py"""
import numpy as np, pandas as pd
from stop_lab import load, PERIODS, PNAME, USER78

DAY = 86_400_000_000_000
H8 = 8 * 3_600_000_000_000
H9 = 9 * 3_600_000_000_000   # KST = UTC+9


def sim(A, unit, add_mode, max_day, gap=0.02):
    """add_mode: None(추가 없음) / 'other'(다른 세션만) / 'any'(같은 세션도). 반환 [(첫 매수 봉, 매도 봉, 자금 대비 수익, 단위 수, 추가 단위 수익)]"""
    o, l, r, ema, tk, dead_of, hot, ses = A["o"], A["l"], A["r"], A["ema"], A["tk"], A["dead"], A["hot7"], A["ses"]
    cyc = (tk + H9 - H8) // DAY                      # 한국 아침 8시 기준 하루 번호
    out, units, dead, ei0, last_i, fses = [], [], 0, 0, -1, ""
    cnt, cur = 0, -1
    for i in range(61, len(o)):
        if cyc[i] != cur:
            cur, cnt = cyc[i], 0
        sold = False
        if units:
            if (i - 1 >= ei0 and r[i - 1] > 70) or tk[i] >= dead:
                px = o[i]
                ret = sum(w * (px / e - 1) for w, e, _ in units)
                add = [w * (px / e - 1) for w, e, k in units if k > 0]
                out.append((ei0, i, ret, len(units), add[0] if add else np.nan))
                units, sold = [], True
        if sold or hot[i]:
            continue
        if max_day and cnt >= max_day:
            continue
        lvl = ema[i - 1] * (1 - gap)
        if l[i] > lvl:
            continue
        if not units:
            units = [(unit, min(o[i], lvl), 0)]
            ei0, last_i, fses, dead = i, i, ses[i], dead_of[i]
            cnt += 1
        elif add_mode and len(units) < 2 and i > last_i and (add_mode == "any" or ses[i] != fses):
            units.append((unit, min(o[i], lvl), 1))
            last_i = i
            cnt += 1
    return out


V = {"V0 지금(100% 한 번)": (1.0, None, 0), "V0h 50% 한 번": (0.5, None, 0),
     "V1 50%+50% 다른 세션, 하루 2번": (0.5, "other", 2), "V2 50%+50% 같은 세션도, 하루 2번": (0.5, "any", 2),
     "V3 50%+50% 다른 세션, 횟수 제한 없음": (0.5, "other", 0), "V4 100% 한 번, 하루 2번": (1.0, None, 2)}

if __name__ == "__main__":
    rows, addr = [], []
    for p in ("early", "late"):
        S = load(p)
        lo, hi = PERIODS[p][3], PERIODS[p][4]
        for name, (unit, mode, mx) in V.items():
            per = {}
            for s, A in S.items():
                if s not in USER78:
                    continue
                tr = [(A["day"][e], ret, n, a) for e, x, ret, n, a in sim(A, unit, mode, mx) if lo <= A["day"][e] <= hi]
                per[s] = tr
            allr = [t for v in per.values() for t in v]
            rets = np.array([t[1] for t in allr])
            comp = np.mean([np.prod([1 + t[1] for t in v]) - 1 for v in per.values()])
            mdds = []
            for v in per.values():
                eq = np.cumprod([1 + t[1] for t in v]) if v else np.array([1.0])
                mdds.append((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min() if v else 0)
            two = [t for t in allr if t[2] == 2]
            rows.append(dict(기간=PNAME[p], 변형=name, 포지션=len(rets), 두번산포지션=len(two), 합=rets.sum(), 평균=rets.mean(),
                             최악=rets.min(), 계좌복리=comp, 최악낙폭=min(mdds)))
            if two:
                addr.append(dict(기간=PNAME[p], 변형=name, 추가건수=len(two), 추가분평균=np.nanmean([t[3] for t in two]) / 0.5))
    r = pd.DataFrame(rows)
    for c, f in (("합", "{:+.2f}"), ("평균", "{:+.2%}"), ("최악", "{:+.1%}"), ("계좌복리", "{:+.1%}"), ("최악낙폭", "{:+.1%}")):
        r[c] = r[c].map(f.format)
    pd.set_option("display.width", 220)
    print("사용자78, 24시간 매매, 수익은 종목별 자금 대비(100% = 그 종목 몫 전부)\n" + r.to_string(index=False))
    a = pd.DataFrame(addr)
    a["추가분평균"] = a["추가분평균"].map("{:+.2%}".format)
    print("\n두 번째(추가) 매수분 자체의 수익(그 50%만의 매수가 → 같이 판 값):\n" + a.to_string(index=False))
