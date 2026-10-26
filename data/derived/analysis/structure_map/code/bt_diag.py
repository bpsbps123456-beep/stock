"""대응 백테스트 점검: (1) 무작위 진입 + 같은 청산 규칙, (2) 같은 신호를 반대 방향으로, (3) 수수료 0 기준.

무작위 진입: 같은 코인·시간봉·기간에서 셋업 수만큼 봉을 고르고, 방향은 셋업과 같은 비율로, 손절 폭(ATR 배수)은
그 셋업들의 분포에서 뽑는다. 진입은 다음 봉 시가, 목표는 같은 구조물 규칙(1차 절반+본전+2차). 씨앗을 바꿔 5번 평균.
반대 방향: 셋업과 같은 체결 봉·가격에서 방향만 뒤집고 손절 폭은 같게, 목표는 반대쪽 구조물.
"""
from __future__ import annotations

import pathlib
import random
import zlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
from bt_run import KINDS, SYMS, TFS, period  # noqa: E402

V0 = bt.Var(buf=0.0)
PATH = "tv" if "--tv" in sys.argv else "stop"     # --tv: 5분봉 없이 트레이딩뷰 전략 테스터 가정


def main():
    real = defaultdict(list)      # kind -> R (수수료 0, 2026)
    rnd = defaultdict(list)
    opp = defaultdict(list)
    for sym in SYMS:
        sub = bt.Sub(sym) if ("--nosub" not in sys.argv and "--tv" not in sys.argv) else None
        for tf in TFS:
            eng = bt.Engine(sym, tf)
            eng.sub = sub
            n = eng.n
            idx26 = [i for i in range(n - 2) if period(eng.b[i]["t"]) in ("26H1", "26H2")]
            per_kind = defaultdict(list)
            seen = set()
            for s in eng.setups:
                if ":" in s.kind or period(eng.b[s.i]["t"]) not in ("26H1", "26H2"):
                    continue
                key = (s.kind, s.i, s.d)
                if key in seen:
                    continue
                seen.add(key)
                f = bt.fill(eng, s)
                if f is None:
                    continue
                r = bt.simulate(eng, s, bt.Var(), 0.0, f, sub, PATH)
                if r is None:
                    continue
                fj, px, mkt = f
                a = eng.atr[fj - 1]
                ru = abs(px - (s.line.val(fj) - s.d * 0.6 * a if s.kind == "D1-BN" else s.line.sub(fj) - s.d * 1.1 * a if s.kind == "E-CH4" else s.stop - s.d * 0.1 * a))
                real[s.kind].append(r["R"])
                per_kind[s.kind].append((s.d, ru / a))
                # 반대 방향: 같은 체결, 같은 손절 폭
                so = bt.Setup(s.kind + "~", s.i, -s.d, px if not mkt else None, None, fj, fj, s.code, s.ref)
                so.stop = px + s.d * ru
                so.lim_dir = s.d
                ro = sim_at(eng, so, fj, px, mkt)
                if ro is not None:
                    opp[s.kind].append(ro["R"])
            # 무작위 진입
            for k, lst in per_kind.items():
                for seed in range(5):
                    rng = random.Random(zlib.crc32(f"{sym}|{tf}|{k}|{seed}".encode()))   # 실행마다 같은 무작위
                    for _ in range(len(lst)):
                        i = rng.choice(idx26)
                        d, katr = rng.choice(lst)
                        px = eng.b[i + 1]["o"]
                        sr_ = bt.Setup("rnd", i, d, None, px - d * katr * eng.atr[i], i + 1, i + 1, -1, px)
                        rr = sim_at(eng, sr_, i + 1, px, True)
                        if rr is not None:
                            rnd[k].append(rr["R"])
            print(sym, tf, "done", flush=True)
    print("\n셋업     실제(수수료0)        무작위 진입(같은 청산)   반대 방향")
    for k in KINDS:
        def s_(x):
            if len(x) < 2:
                return "—"
            m = statistics.mean(x)
            t = m / (statistics.pstdev(x) / len(x) ** 0.5)
            return f"{m:+.3f} (n{len(x)}, t{t:+.1f})"
        print(f"{k:7}  {s_(real[k]):22} {s_(rnd[k]):24} {s_(opp[k])}")


def sim_at(eng, s, fj, px, mkt):
    return bt.simulate(eng, s, V0, 0.0, (fj, px, mkt), eng.sub, PATH)


if __name__ == "__main__":
    main()
