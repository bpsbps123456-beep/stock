"""확인 단계(BRIEF2.md '확인' 절) — 얼린 후보 전부를 한 번 잰다. 모든 묶음이 얼린 뒤에만 실행한다.

무엇을 재나
- 후보: results/fam_F1_frozen.json ~ fam_F5_frozen.json 의 frozen 항목 전부. 전략은 각 항목의 factory 문자열
  'fam_Fk.make(<dict 리터럴>)'을 ast.literal_eval 로 풀어 그 모듈의 make(dict)를 그대로 부른다(얼린 코드·매개변수 그대로).
  파일이 없으면 그 묶음은 missing 으로 적는다. factory 가 오류를 내면 고치지 않고 factory_errors 에 적는다.
- 구간 sel·conf·aux × 마감 actual·std × 판(base = 비용 1배·종가 체결 / cost2x = 비용 2배·종가 / open = 비용 1배·시가 체결).
  conf 는 7~8월(h1)·9~10월(h2) 두 반을 base 로 따로 잰다.
- 대조(base 판): 늘 SOXL, 늘 SOXS(09:31 결정 = 09:30 봉 마감에 주문 → 09:31 봉 종가 체결 → 마감, fam_F1.Always 와 같은 정의),
  각 후보의 거울.

결과 보기 전에 정한 해석(바꾸지 않는다)
1. 거울 = MirrorWrap: 안쪽 전략에는 ctx.pos 를 뒤집어 보여 주고(자기 방향으로 보이게), 낸 Order.pos 의 부호를 바꾼다.
   손절·목표 비율은 그대로 둔다(fam_F3.Mirror 와 같은 꼴). 뒤집힌 종목이 아직 거래 전(nan)이면 주문하지 않는다.
   예외 F2: NoiseBand 는 따라가는 손절을 ctx.entry_px(자기 종목 체결가)로 다시 재는데, 래퍼 아래서는 그 값이 반대 종목
   가격이 되어 손절 비율이 뜻을 잃는다. 그래서 F2 는 묶음이 설계해 frozen 에 적어 둔 mirror_factory
   (fam_F2.make(p, mirror=True): 같은 청산 사건, 손절 → 지정가 목표)를 거울로 쓴다.
   묶음 고유 거울이 있는 F1·F4·F5 는 mirror_native 로도 재서 래퍼와 비교한다(F5 rng 손절만 정의가 달라 값이 다를 수 있다).
2. 판정(마감 방식마다): 지표는 모두 conf base 날별 수익(%) 기준.
   후보 = 평균 > 0 & 부트스트랩(10,000회, seed 20261029) 95% 하한 > 0 & Holm 보정 단측 p ≤ 0.05
          & h1·h2 평균 모두 > 0 & cost2x 평균 > 0.
   관찰 = 후보가 아니고 평균 > 0 & h1·h2 평균 모두 > 0.
          (BRIEF2 문구 '구간은 0 포함'은 후보 조건 중 하한만 못 넘은 경우를 적은 것으로 읽는다. 하한 > 0 인데 Holm·비용 2배에서
          떨어진 경우 문자 그대로면 어느 칸에도 안 맞으므로 관찰로 두고 literal_differs 로 표시한다.)
   탈락 = 나머지.
3. Holm: 마감 방식별로 얼린 후보 전체(m = 얼린 후보 수)의 conf base 단측 부트스트랩 p(lib.summarize p_one_sided =
   부트스트랩 평균 ≤ 0 비율)를 step-down 보정한다. 후보가 고른 마감 방식이 아니어도 모든 후보를 두 방식 모두에 넣는다.
4. 판정에 open(시가 체결)은 쓰지 않는다(감도). aux(2025)는 장세 전환 위험 설명용이다.

사용: cd code && py confirm.py → results/confirm.json, results/confirm_daily_<file_id>_<scheme>.csv
"""
import ast
import hashlib
import json
import os
import re
import sys
import time
import traceback

os.environ["SOS_UNLOCK"] = "frozen"          # 확인 단계만 conf·aux 를 연다(모든 묶음이 얼린 뒤)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                            # noqa: E402
import pandas as pd                           # noqa: E402

from lib import Data, Engine, Order, INST, RTH0, PERIODS, cutoff_idx, summarize   # noqa: E402
import fam_F1, fam_F2, fam_F3, fam_F4, fam_F5  # noqa: E402,E401

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
FAMS = ["F1", "F2", "F3", "F4", "F5"]
MODS = {"fam_F1": fam_F1, "fam_F2": fam_F2, "fam_F3": fam_F3, "fam_F4": fam_F4, "fam_F5": fam_F5}
SCHEMES = ["actual", "std"]
RUN_PERIODS = ["sel", "conf", "aux"]
VARIANTS = {"base": dict(cost_mult=1.0, fill="close"), "cost2x": dict(cost_mult=2.0, fill="close"),
            "open": dict(cost_mult=1.0, fill="open")}
HALVES = {"h1": ("2026-07-01", "2026-08-31"), "h2": ("2026-09-01", "2026-10-28")}
ALPHA = 0.05
NBOOT, SEED = 10000, 20261029
FACTORY_RE = re.compile(r"^\s*(fam_F[1-5])\.make\((.*)\)\s*$", re.S)


def build_factory(s):
    """'fam_Fk.make({...})' → 그 모듈의 make(dict). dict 는 ast.literal_eval(None·True·False·숫자·문자열만)."""
    m = FACTORY_RE.match(s)
    if not m:
        raise ValueError(f"factory 꼴이 아니다: {s!r}")
    params = ast.literal_eval(m.group(2))
    if not isinstance(params, dict):
        raise ValueError(f"make 인자가 dict 가 아니다: {s!r}")
    return MODS[m.group(1)].make(params), params


class MirrorWrap:
    """SOXL↔SOXS 뒤집기. 안쪽 전략에는 ctx.pos 를 뒤집어 보여 주고 Order.pos 부호를 바꾼다(손절·목표 비율 그대로)."""

    def __init__(self, inner):
        self.inner = inner
        self.start = getattr(inner, "start", "09:30")

    def on_day(self, ctx):
        if hasattr(self.inner, "on_day"):
            ctx.pos = -ctx.pos
            try:
                self.inner.on_day(ctx)
            finally:
                ctx.pos = -ctx.pos

    def on_bar(self, ctx, i):
        ctx.pos = -ctx.pos
        try:
            o = self.inner.on_bar(ctx, i)
        finally:
            ctx.pos = -ctx.pos
        if o is None:
            return None
        if o.pos != 0 and not np.isfinite(ctx.bars(INST[-o.pos], "Close")[i]):
            return None
        return Order(-o.pos, stop=o.stop, target=o.target, keep=o.keep)


def wrap_mirror(make):
    return lambda: MirrorWrap(make())


class Always:
    """대조: 09:31 결정(09:30 봉 마감) → 다음 봉 종가 체결 → 마감까지 한 종목. fam_F1.Always 와 같은 정의."""
    start = "09:30"

    def __init__(self, pos):
        self.pos = pos

    def on_bar(self, ctx, i):
        return Order(self.pos) if i == RTH0 else None


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def safe_id(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.replace("=", "-")).strip("_")


def picked_for(e):
    if e.get("picked_for"):
        return list(e["picked_for"])
    s = str(e.get("scheme", ""))
    return [x for x in SCHEMES if x in s] or [s]


def frozen_sel_mean(e, scheme):
    """얼린 파일에 적힌 sel base 평균(재현 점검용)."""
    if isinstance(e.get("sel"), dict) and scheme in e["sel"]:
        return e["sel"][scheme].get("mean_pct")
    v = e.get(f"sel_{scheme}")
    return v.get("mean_pct") if isinstance(v, dict) else None


def native_mirror(fam, p):
    if fam == "F1":
        return fam_F1.make({**p, "mirror": True}), "fam_F1.make({**p, 'mirror': True})"
    if fam == "F2":
        return fam_F2.make(p, mirror=True), "fam_F2.make(p, mirror=True)"
    if fam == "F4":
        return fam_F4.make(p, mirror=True), "fam_F4.make(p, mirror=True)"
    if fam == "F5":
        return fam_F5.make(p, mirror=True), "fam_F5.make(p, mirror=True)"
    return None, None


def load_candidates():
    cands, missing, errors = [], [], []
    for fam in FAMS:
        path = os.path.join(RES, f"fam_{fam}_frozen.json")
        if not os.path.exists(path):
            missing.append(fam)
            continue
        j = json.load(open(path, encoding="utf-8"))
        for e in j["frozen"]:
            try:
                make, fp = build_factory(e["factory"])           # 얼린 factory 그대로
                make()                                            # 객체 생성 점검
            except Exception as ex:                               # 고치지 않고 기록만 한다
                errors.append({"family": fam, "id": e.get("id"), "factory": e.get("factory"),
                               "error": repr(ex), "trace": traceback.format_exc(limit=3)})
                continue
            nm, nm_src = native_mirror(fam, fp)
            if fam == "F2":
                mir, mir_src = nm, f"native: {nm_src} (래퍼는 ctx.entry_px 를 반대 종목 값으로 읽어 쓸 수 없다)"
            else:
                mir, mir_src = wrap_mirror(make), "MirrorWrap(ctx.pos 뒤집어 보여 주기 + Order.pos 부호 뒤집기)"
            cands.append({"family": fam, "id": str(e["id"]), "file_id": safe_id(str(e["id"])),
                          "picked_for": picked_for(e), "params": e["params"], "factory": e["factory"],
                          "factory_params": fp, "make": make, "mirror": mir, "mirror_src": mir_src,
                          "native_mirror": nm if fam in ("F1", "F4", "F5") else None, "native_src": nm_src,
                          "frozen_sel_mean": {s: frozen_sel_mean(e, s) for s in SCHEMES},
                          "n_configs": j.get("n_configs", j.get("n_configs_total"))})
    return cands, missing, errors


def holm(ps):
    """ps: {id: p}. 반환 {id: Holm 보정 p}(step-down, 단조)."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p))
        out[k] = run
    return out


def exit_mix(tr):
    if not len(tr):
        return {"trades": 0}
    return {"trades": int(len(tr)), "why": {k: int(v) for k, v in tr["why"].value_counts().items()},
            "by_sym": {k: int(v) for k, v in tr["sym"].value_counts().items()},
            "trade_mean_pct": float(tr["ret"].mean() * 100),
            "trade_mean_by_sym_pct": {k: float(g["ret"].mean() * 100) for k, g in tr.groupby("sym")}}


def slim(s):
    keys = ["days", "active_days", "trades", "mean_pct", "sd_pct", "hit_days", "sum_pct", "compound_pct", "t",
            "ci95", "p_one_sided", "max_dd_pct"]
    return {k: s.get(k) for k in keys if k in s}


def in_half(dl, h):
    a, b = HALVES[h]
    return (dl["day"] >= pd.Timestamp(a).date()) & (dl["day"] <= pd.Timestamp(b).date())


def main():
    t0 = time.time()
    hashes = {f: sha(os.path.join(HERE, f)) for f in ["lib.py", "fam_F1.py", "fam_F2.py", "fam_F3.py", "fam_F4.py",
                                                      "fam_F5.py", "confirm.py"]}
    D = Data()
    print(f"자료 {time.time() - t0:.1f}s, 날 {len(D.days)}", flush=True)
    cands, missing, errors = load_candidates()
    print(f"후보 {len(cands)}개, 빠진 묶음 {missing}, factory 오류 {len(errors)}", flush=True)
    for c in cands:
        print("  ", c["family"], c["id"], c["picked_for"], c["factory_params"], flush=True)
    E = {(s, v): Engine(D, scheme=s, **kw) for s in SCHEMES for v, kw in VARIANTS.items()}
    pdays = {p: D.days_in(p) for p in RUN_PERIODS}
    for p in RUN_PERIODS:
        print(f"  {p}: {len(pdays[p])}일 {D.days[pdays[p][0]]}~{D.days[pdays[p][-1]]}", flush=True)

    def run(make, s, v, p):
        dl, tr = E[(s, v)].run(make, period=p)
        bad = int((~np.isfinite(dl["ret"].values)).sum())
        assert bad == 0, f"nan 수익 {bad}일"
        return dl, tr

    # ── 대조 ──
    base, bdaily = {}, {}
    for nm, pos in (("always_SOXL", 1), ("always_SOXS", -1)):
        for s in SCHEMES:
            for v in VARIANTS:
                for p in RUN_PERIODS:
                    dl, _ = run(lambda pos=pos: Always(pos), s, v, p)
                    base.setdefault(nm, {}).setdefault(s, {}).setdefault(p, {})[v] = slim(summarize(dl, NBOOT, SEED))
                    if v == "base":
                        bdaily[(nm, s, p)] = dl
            dl = bdaily[(nm, s, "conf")]
            base[nm][s]["conf_halves"] = {h: slim(summarize(dl[in_half(dl, h)].reset_index(drop=True), NBOOT, SEED))
                                          for h in HALVES}
    print(f"대조 끝 {time.time() - t0:.0f}s", flush=True)

    # ── 후보 ──
    out = []
    for c in cands:
        rec = {k: c[k] for k in ["family", "id", "file_id", "picked_for", "params", "factory", "factory_params",
                                 "mirror_src", "n_configs"]}
        rec.update(results={}, halves={}, mirror={}, mirror_native={}, exits_conf={}, sel_repro={})
        for s in SCHEMES:
            R = rec["results"][s] = {}
            keep = {}
            for p in RUN_PERIODS:
                R[p] = {}
                for v in VARIANTS:
                    dl, tr = run(c["make"], s, v, p)
                    R[p][v] = slim(summarize(dl, NBOOT, SEED))
                    keep[(p, v)] = dl
                    if p == "conf" and v == "base":
                        rec["exits_conf"][s] = exit_mix(tr)
            fm = c["frozen_sel_mean"][s]
            rec["sel_repro"][s] = {"frozen_mean_pct": fm, "rerun_mean_pct": R["sel"]["base"]["mean_pct"],
                                   "abs_diff": None if fm is None else abs(fm - R["sel"]["base"]["mean_pct"])}
            dc = keep[("conf", "base")]
            rec["halves"][s] = {h: slim(summarize(dc[in_half(dc, h)].reset_index(drop=True), NBOOT, SEED))
                                for h in HALVES}
            mir_d, nat_d = {}, {}
            rec["mirror"][s], rec["mirror_native"][s] = {}, {}
            for p in RUN_PERIODS:
                dm, _ = run(c["mirror"], s, "base", p)
                mir_d[p] = dm
                rec["mirror"][s][p] = slim(summarize(dm, NBOOT, SEED))
                if c["native_mirror"] is not None:
                    dn, _ = run(c["native_mirror"], s, "base", p)
                    nat_d[p] = dn
                    rec["mirror_native"][s][p] = {**slim(summarize(dn, NBOOT, SEED)),
                                                  "max_abs_diff_vs_wrapper_pct":
                                                      float(np.max(np.abs(dn["ret"].values - dm["ret"].values)) * 100)}
            # 날별 CSV(conf·aux)
            rows = []
            for p in ("conf", "aux"):
                b0 = keep[(p, "base")]
                assert list(b0["day"]) == list(bdaily[("always_SOXL", s, p)]["day"]) == list(mir_d[p]["day"])
                for k in range(len(b0)):
                    day = b0["day"].iloc[k]
                    ci = cutoff_idx(day, s)
                    row = {"day": day, "period": p,
                           "half": "" if p != "conf" else ("h1" if day <= pd.Timestamp(HALVES["h1"][1]).date() else "h2"),
                           "cutoff_et": f"{(ci + 240) // 60:02d}:{(ci + 240) % 60:02d}",
                           "ret_pct": b0["ret"].iloc[k] * 100, "n": int(b0["n"].iloc[k]),
                           "ret_cost2x_pct": keep[(p, "cost2x")]["ret"].iloc[k] * 100,
                           "ret_open_pct": keep[(p, "open")]["ret"].iloc[k] * 100,
                           "mirror_ret_pct": mir_d[p]["ret"].iloc[k] * 100,
                           "always_soxl_pct": bdaily[("always_SOXL", s, p)]["ret"].iloc[k] * 100,
                           "always_soxs_pct": bdaily[("always_SOXS", s, p)]["ret"].iloc[k] * 100}
                    if nat_d:
                        row["mirror_native_ret_pct"] = nat_d[p]["ret"].iloc[k] * 100
                    rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(RES, f"confirm_daily_{c['file_id']}_{s}.csv"), index=False,
                                      encoding="utf-8-sig")
        out.append(rec)
        print(f"  {c['id']} 끝 {time.time() - t0:.0f}s", flush=True)

    # ── Holm·판정 ──
    holm_tab, verdicts = {}, {}
    for s in SCHEMES:
        ps = {r["id"]: r["results"][s]["conf"]["base"]["p_one_sided"] for r in out}
        adj = holm(ps)
        holm_tab[s] = {k: {"p_one_sided": ps[k], "holm_p": adj[k]} for k in ps}
        for r in out:
            cb = r["results"][s]["conf"]["base"]
            chk = {"mean_gt0": cb["mean_pct"] > 0, "ci_lo_gt0": cb["ci95"][0] > 0, "holm_le_0.05": adj[r["id"]] <= ALPHA,
                   "h1_gt0": r["halves"][s]["h1"]["mean_pct"] > 0, "h2_gt0": r["halves"][s]["h2"]["mean_pct"] > 0,
                   "cost2x_gt0": r["results"][s]["conf"]["cost2x"]["mean_pct"] > 0}
            if all(chk.values()):
                v = "후보"
            elif chk["mean_gt0"] and chk["h1_gt0"] and chk["h2_gt0"]:
                v = "관찰"
            else:
                v = "탈락"
            r.setdefault("verdict", {})[s] = v
            r.setdefault("verdict_checks", {})[s] = {**chk, "holm_p": adj[r["id"]],
                                                     "literal_differs": v == "관찰" and chk["ci_lo_gt0"]}
            verdicts.setdefault(r["id"], {})[s] = v

    res = {"stage": "확인(BRIEF2 '확인' 절)", "brief": "BRIEF2.md", "unlock": "SOS_UNLOCK=frozen(이 스크립트 안에서만)",
           "periods": {p: PERIODS[p] for p in RUN_PERIODS}, "period_days": {p: len(pdays[p]) for p in RUN_PERIODS},
           "conf_halves": HALVES, "variants": VARIANTS, "bootstrap": {"n": NBOOT, "seed": SEED},
           "rules_fixed_before_results": __doc__, "code_sha256_16": hashes,
           "missing_families": missing, "factory_errors": errors, "n_frozen": len(out),
           "holm": holm_tab, "verdicts": verdicts, "baselines": base, "candidates": out,
           "daily_files": {r["id"]: {s: f"results/confirm_daily_{r['file_id']}_{s}.csv" for s in SCHEMES} for r in out},
           "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(RES, "confirm.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ── 화면 요약 ──
    print("\n대조(base, mean %·t):")
    for nm in base:
        for s in SCHEMES:
            print(f"  {nm} {s}: " + " | ".join(
                f"{p} {base[nm][s][p]['base']['mean_pct']:+.3f} t{base[nm][s][p]['base']['t']:+.2f}" for p in RUN_PERIODS)
                + f" | conf h1 {base[nm][s]['conf_halves']['h1']['mean_pct']:+.3f} h2 {base[nm][s]['conf_halves']['h2']['mean_pct']:+.3f}")
    for r in out:
        print(f"\n{r['family']} {r['id']} picked {r['picked_for']} sel 재현 "
              f"{ {s: r['sel_repro'][s]['abs_diff'] for s in SCHEMES} }")
        for s in SCHEMES:
            R = r["results"][s]
            cb = R["conf"]["base"]
            print(f"  [{s}] 판정 {r['verdict'][s]}  conf mean {cb['mean_pct']:+.3f} t {cb['t']:+.2f} "
                  f"ci [{cb['ci95'][0]:+.3f},{cb['ci95'][1]:+.3f}] p {cb['p_one_sided']:.4f} "
                  f"holm {holm_tab[s][r['id']]['holm_p']:.4f} act {cb['active_days']}/{cb['days']} "
                  f"| h1 {r['halves'][s]['h1']['mean_pct']:+.3f} h2 {r['halves'][s]['h2']['mean_pct']:+.3f} "
                  f"| 2x {R['conf']['cost2x']['mean_pct']:+.3f} open {R['conf']['open']['mean_pct']:+.3f} "
                  f"| mirror {r['mirror'][s]['conf']['mean_pct']:+.3f}")
            print(f"         sel {R['sel']['base']['mean_pct']:+.3f} t {R['sel']['base']['t']:+.2f} "
                  f"| aux mean {R['aux']['base']['mean_pct']:+.3f} t {R['aux']['base']['t']:+.2f} "
                  f"ci [{R['aux']['base']['ci95'][0]:+.3f},{R['aux']['base']['ci95'][1]:+.3f}] "
                  f"2x {R['aux']['cost2x']['mean_pct']:+.3f} mirror {r['mirror'][s]['aux']['mean_pct']:+.3f} "
                  f"| exits {r['exits_conf'][s]}")
            if r["mirror_native"].get(s):
                print(f"         native mirror conf {r['mirror_native'][s]['conf']['mean_pct']:+.3f} "
                      f"(래퍼와 최대 차 {r['mirror_native'][s]['conf']['max_abs_diff_vs_wrapper_pct']:.4f}%)")
    print(f"\n끝 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
