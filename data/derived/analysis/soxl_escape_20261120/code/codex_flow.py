"""SOXL flow study. Run with py code/codex_flow.py in the study directory."""
from itertools import product
from pathlib import Path
import json

import numpy as np
import pandas as pd
import harness as H

ROOT = Path(H.ROOT)
CLOCK = {"전체": (570, 960), "개장": (570, 600), "점심": (690, 810), "마감": (900, 960)}
FLOW = ["전체", "거래량급증", "보통거래량", "다중봉"]


def features():
    b = H.resample(1)
    c, v = b.Close, b.Volume
    vw = H.session_vwap(b)
    tp = (b.High + b.Low + c) / 3
    vol = v.groupby(b.date).cumsum()
    sigma = ((tp ** 2 * v).groupby(b.date).cumsum() / vol - vw ** 2).clip(lower=0) ** .5
    hist = v.shift(1).rolling(30)
    z = (v - hist.mean()) / hist.std(ddof=0).replace(0, np.nan)
    bear, bull = c < b.Open, c > b.Open
    for minutes in (3, 15):
        bars = H.resample(minutes)
        # At the OPEN of the signal minute, use the most recent completed HTF.
        bear = bear & H.to_1m_mask(bars.Close < bars.Open, bars)
        bull = bull & H.to_1m_mask(bars.Close > bars.Open, bars)
    decline = H.ctx_decline(b)
    rally = H.ctx_rally(b)
    mask = {"exit": H.to_1m_mask(rally, b), "bottom": H.to_1m_mask(decline, b)}
    signals = {}
    definitions = {}
    for kind in ("exit", "bottom"):
        for setup, level in product(("VWAP", "전저점"), (0.5, 1.5) if kind == "bottom" else (0, 1)):
            line = vw - level * sigma if setup == "VWAP" else b.Low.shift(1).rolling(
                (5 if level in (0, .5) else 15)).min()
            if kind == "exit":
                sig = rally & (c < line) & (c.shift() >= line.shift())
                desc = "현재 종가≥최근 60봉 최저가×1.015, 직전 종가≥직전 선, 현재 종가<현재 선"
            else:
                sig = decline & (b.Low < line) & (c >= line)
                desc = "현재 종가≤최근 120봉 최고가×0.97, 현재 저가<현재 선≤현재 종가"
            line_desc = (f"선=04시 ET부터 누적한 세션 VWAP−{level}×σ" if setup == "VWAP"
                         else f"선=직전 {5 if level in (0,.5) else 15}개 1분봉 저가의 최솟값")
            for flow, (clock, (start, end)) in product(FLOW, CLOCK.items()):
                condition = {"전체": True, "거래량급증": z >= 1, "보통거래량": z < 1,
                             "다중봉": bear if kind == "exit" else bull}[flow]
                key = f"{setup}/{level}/{flow}/{clock}"
                flag = sig & condition & (b["mod"] >= start) & (b["mod"] < end)
                pos = H.exec_index(flag, b)
                pos = pos[pos >= 0]
                signals[kind, key] = H.dedup(pos[b.reg.to_numpy()[pos]], 30)
                direction = "음봉" if kind == "exit" else "양봉"
                filter_desc = {"전체": "거래량·상위봉 조건 없음", "다중봉": f"현재 1분봉과 그 시가 시점에 완성된 최신 3·15분봉 모두 {direction}",
                               "거래량급증": "거래량 z≥1", "보통거래량": "거래량 z<1"}[flow]
                definitions[kind, key] = f"{desc}. {line_desc}. {filter_desc}. 신호봉 시작은 ET 자정 후 [{start},{end})분."
    return signals, definitions, mask


def event_table(rng):
    p = H.baseline_positions(rng)
    # event_metrics uses its input's maximum date for partial-session detection.
    # Add the real dataset endpoint, then restrict scoring back to requested rows.
    return H.event_metrics(np.append(p, len(H.load_1m()) - 1)).iloc[:-1].set_index("pos", drop=False)


def evaluate(kind, pos, rng, events, ctx):
    selected = events.loc[events.index.intersection(pos)]
    score = H.summarize(selected, rng, kind, ctx)
    if kind == "exit" and score["n"]:
        score.update(H.hold_sim(selected.pos.values, rng))
    return score


def ranking(kind, score):
    if not (score["days"] >= 6 and .3 <= score["per_day"] <= 6):
        return ()
    keys = (["x_rand_mean%", "x_rand_t_days", "r60_x_dm%"] if kind == "exit" else
            ["r60_x_ctx%", "rclose_x_ctx%", "r60_x_dm_t"])
    return tuple(score[k] if score[k] is not None else -1e9 for k in keys)


def main():
    signals, definitions, ctx = features()
    results = {"grid": {}, "final": {}}
    train = event_table(H.TRAIN)
    for kind in ctx:
        grid = {key: evaluate(kind, pos, H.TRAIN, train, ctx[kind])
                for (side, key), pos in signals.items() if kind == side}
        key = max(grid, key=lambda k: ranking(kind, grid[k]))
        results["grid"][kind] = grid
        results["final"][kind] = {"key": key, "definition": definitions[kind, key], "TRAIN": grid[key]}
    for label, rng in [("TEST", H.TEST), ("PRE", H.PRE)]:
        events = event_table(rng)
        for kind, item in results["final"].items():
            item[label] = evaluate(kind, signals[kind, item["key"]], rng, events, ctx[kind])
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k,v in o.items()}
        return None if isinstance(o, float) and not np.isfinite(o) else o
    (out / "codex.json").write_text(json.dumps(clean(results), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    final = results["final"]
    x, y = final["exit"], final["bottom"]
    lines = ["EXIT는 TEST에서 평균 손익이 소폭 개선됐지만 날짜별 편차가 크고 하위 5% 손실이 더 컸다. BOTTOM은 TEST에서 방향이 뒤집혀 바닥 신호로 검증되지 않았다.", f"TEST에서 EXIT의 무작위 대비 평균 손익차는 {x['TEST'].get('x_rand_mean%')}%p, BOTTOM의 하락 상황 대비 60분 수익차는 {y['TEST'].get('r60_x_ctx%')}%p였다.",
             "", "| 신호·구간 | 횟수(일평균) | 1순위 초과 %p | 날짜 t¹ | 종가 초과² %p | ±1.5% 먼저/기준 % | 꼬리³ % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for kind,item in final.items():
        for label in ("TRAIN","TEST"):
            s = item[label]
            is_exit = kind == "exit"
            tail = f"{s.get('rule_cvar5%')}/{s.get('rand_cvar5%')}" if is_exit else str(s.get('mae60_med%'))
            lines.append(f"| {kind.upper()} {label} | {s['n']} ({s.get('per_day')}) | {s.get('x_rand_mean%' if is_exit else 'r60_x_ctx%')} | {s.get('x_rand_t_days' if is_exit else 'r60_x_dm_t')} | {s.get('rclose_x_dm%' if is_exit else 'rclose_x_ctx%')} | {s.get('fp_good%')}/{s.get('fp_good_base%')} | {tail} |")
    lines += ["¹ EXIT는 날짜별 무작위 대비, BOTTOM은 같은 날 ±30분 대비 t. ² EXIT는 ±30분 대비, BOTTOM은 같은 하락 상황 대비. ³ EXIT는 규칙/무작위 CVaR5, BOTTOM은 60분 최대 역행폭 중앙값.",
              "EXIT·BOTTOM 각각 2종 선×2수준×4필터(무조건·z≥1·z<1·1/3/15분 합의)×4시간대(전체·09:30~10:00·11:30~13:30·15:00~16:00 ET)=64개, 총 128개. 선은 VWAP 아래 EXIT 0·1σ, BOTTOM 0.5·1.5σ 또는 직전 5·15봉 저점. EXIT는 종가 이탈을 지속 하락 후보, BOTTOM은 봉 안 이탈 후 회복을 반등 후보로 분류했다.",
              "TRAIN 15일만으로 신호일 ≥6, 하루 0.3~6회인 후보에서 표의 1순위 초과 최대값 선택. 동률 시 EXIT 날짜 t→60분 근접시각 초과, BOTTOM 종가 초과→날짜 t. 다중 탐색에 따른 선택편향이 있다.",
              f"PRE 장세 점검에서 EXIT 무작위 대비 {x['PRE'].get('x_rand_mean%')}%p, BOTTOM 하락 상황 대비 {y['PRE'].get('r60_x_ctx%')}%p. TEST 11월 19일은 미완성으로 보유 모의에서 제외. 공통 틀의 TEST 대조군은 해당 일자의 미완성 종가·60분 값을 일부 포함하며 CVaR 무작위는 진입별 20회 평균 경로를 사용한다.",
              "정규장 전체 체결 모의(수수료 0, 스프레드 미반영)로 한국 00~08시 수동 주문 제약이 있다. 개장은 뉴욕 09:30 고정, 한국은 서머타임 중 22:30·종료 후 23:30.",
              "", "Pine v6 재현 규칙", "EXIT — "+x["definition"], "BOTTOM — "+y["definition"],
              "연장시간 1분 차트에서 봉 확정 시 판정. 극값 창은 별도 표기가 없으면 현재 봉 포함. VWAP와 σ는 hlc3의 당일 거래량 가중 평균·모표준편차, z는 현재 거래량에서 직전 30봉 평균을 빼고 그 30봉 모표준편차로 나눈 값(0이면 결측). 같은 날짜 다음 가용 정규장 1분봉 시가에 체결(바로 다음 봉이 정규장 밖이면 폐기), 같은 방향 직전 채택 체결부터 30분 미만 억제."]
    (out / "codex_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()


