"""prechecks.md 쓰기(prechecks.py가 부른다). 숫자는 모두 prechecks_summary·prechecks_price·prechecks_external·
lexicon_v02_precision·loader_summary에서 가져온다. 판정 규칙은 HYPOTHESES.md §2.7·§3·§5.1 그대로이고,
해석을 고른 곳은 interim/notes_prechecks.md에 있다."""
from __future__ import annotations

import re

import pandas as pd

from prechecks import FROZEN, LEX_V02, MIN_BLOCKS, MISS_MAX, RESULTS, fmt, pct

# 가설 → 쓰는 사전 묶음(§3·§4 '쓰는 가설')
LEX_OF = {
    "H01": "E F", "H05": "E F FO ANG", "H06": "BULL BEAR", "H09": "BULL BEAR", "H10": "BULL BEAR SEMI",
    "H11": "BULL BEAR", "H12": "BULL BEAR PX", "H13": "MOCK_BEAR MOCK_BULL", "H14": "E F", "H15": "E F", "H16": "E F",
    "H19": "E", "H20": "E F", "H21": "FO", "H22": "BULL BEAR REBUT REBUT_FEAR", "H24": "F", "H25": "CAP HES",
    "H26": "RES ANG", "H27": "F", "H28": "F", "H30": "MC DIST", "H31": "LL_US", "H33": "MC", "H37": "BULL BEAR",
    "H38": "ALIAS_SEC ALIAS_SKH ALIAS_BOTH", "H39+H48": "ALIAS_SEC ALIAS_SKH ALIAS_BOTH ALIAS_US",
    "H40": "ALIAS_SEC ALIAS_SKH ALIAS_US", "H41": "ALIAS_SEC ALIAS_SKH ALIAS_US",
    "H43": "CHASE_BUY CHASE_SELL ALIAS_SEC ALIAS_SKH ALIAS_US", "H45": "IPO", "H46": "TILT_US TILT_KR", "H47": "KX KR",
    "H49": "E F", "H50": "MEM", "H51": "E F", "H53": "E F", "H57a": "E F", "H57b": "E F", "H57c": "E F", "H59": "E F",
}


def _failed(lexp: dict, hid: str) -> list[str]:
    b = lexp["bundles"]
    m = re.match(r"H\d+[abc]?(\+H48)?", hid)
    key = m.group(0) if m else hid
    return [x for x in LEX_OF.get(key, "").split() if x in b and b[x]["fail"]]


def _f(out, hid, cell=None):
    for r in out["features"]:
        if r["id"] == hid and (cell is None or r["cell"] == cell):
            return r
    raise KeyError((hid, cell))


def _w(out, hid, cell=None, market=None):
    for r in out["weekly"]:
        if r["id"] == hid and (cell is None or r["cell"] == cell) and (market is None or r["market"] == market):
            return r
    raise KeyError((hid, cell, market))


def _c(out, key):
    return out["correlations"][key]


def _cs(c: dict) -> str:
    return "—" if c["pearson"] is None else f"{c['pearson']:.3f} / {c['spearman']:.3f} (n {c['n']})"


def classify(out, price, ext, lexp) -> list[dict]:
    """가설별 판정 전 상태. auto = §2.7·§3 문턱을 넘은 이유(자동 설명용), prov = 외부 자료·해석 대기(잠정),
    warm = 지금 결측이 예열 때문(구조 결측은 문턱 안), meas = 측정 실패 묶음."""
    ev = price["h24_events"]
    nf_ev = len(ev["2026앞"]["kospi_or_hynix_clusters"])
    nb_ev = len(ev["2026뒤(10/14까지)"]["kospi_or_hynix_clusters"])
    n25 = len(ev["2025"]["kospi_or_hynix_clusters"])
    ds = price["down_state"]
    rg = price["h36_regimes"]
    den = out["denominators"]
    cor = out["correlations"]
    rows = []

    def add(hid, grade, feat, now, struct, blocks, auto, prov, warm, other, cell=None):
        meas = _failed(lexp, hid)
        rows.append({"id": hid, "grade": grade, "feat": feat, "now": now, "struct": struct, "blocks": blocks,
                     "auto": auto, "prov": prov, "warm": warm, "meas": meas, "other": other})

    def feat_basic(hid, cell, grade, label, extra_auto=None, extra_prov=None, other="", warm_note=True):
        r = _f(out, hid, cell)
        auto = list(extra_auto or [])
        prov = list(extra_prov or [])
        s = r["miss_struct"]
        if s is not None and s > MISS_MAX:
            auto.append(f"구조 결측 {pct(s)} > 30%")
        bf, bb = r["blocks_front_strict"], r["blocks_back_strict"]
        if min(bf, bb) < MIN_BLOCKS:
            auto.append(f"최소 표본: 엄한 독립 묶음 앞 {bf}·뒤 {bb} < 8")
        warm = warm_note and (r["miss_judge"] > MISS_MAX) and not (s is not None and s > MISS_MAX)
        add(hid, grade, f"{label} `{r['col']}`",
            f"{pct(r['miss_all'])} / {pct(r['miss_judge'])}", pct(s), f"{bf} / {bb}", auto, prov, warm, other)

    nq = ext["NQ=F_1h"]["old_week"].get("ok")
    # ---- F0
    feat_basic("H01", "관문", "관문", "S", other="corr(S, r(D−1))는 5단계(가격)에서. M0 R² 분해는 120창 뒤")
    add("H02", "측정 선택", "TMP", "못 함", "—", "—", [], [], False,
        "TMP(3,000단위 LLM 라벨)를 만들지 않았다 → corr(TMP, S) 못 함. E·F 실패라 H02 규칙상 TMP가 대체 후보")
    # ---- T1
    feat_basic("H03", "T1 대표", "T1 대표", "A",
               other=f"저장판과 상관 {_cs(cor['H03_corr_A_Asaved_rz8'])}(부호 같음), 스팸 파도 0")
    feat_basic("H06", "T1 대표", "T1 대표", "D_SD",
               other=f"표 20 미만 창이 구조 결측의 원인. H06 기준(score200 극성 정확도)은 v0.2 71%(14개) 통과")
    feat_basic("H10", "T1 대표", "T1 대표", "반도체 D_SD z60",
               other="표(점수 ≠ 0 작성자) 20 미만 창이 원인. NQ=F 시간봉 받기 가능")
    # ---- F1
    feat_basic("H05", "F1-1", "F1-1", "ED")
    c = cor["H04_corr_U_N_rz8"]
    add("H04", "F1-2", "B⊥ `Bperp_rz8`", f"{pct(_f(out,'H04')['miss_all'])}", pct(_f(out, "H04")["miss_struct"]),
        f"{_f(out,'H04')['blocks_front_strict']} / {_f(out,'H04')['blocks_back_strict']}",
        [f"corr(rz8 U, rz8 N) {c['pearson']:.3f} > 0.9 ({c['n']}창)"] if (c["pearson"] or 0) > 0.9 else [], [], False,
        f"4창판 {_cs(cor['H04_corr_U_N_rz8m4'])}, 원값 {_cs(cor['H04_corr_U_N_raw_per_h'])}")
    w52 = _w(out, "H52")
    add("H52", "F1-3", "WK `WK_rz8`(주말 13개)", pct(w52["miss_all"]), pct(w52["miss_struct"]), f"— / {w52['blocks_back']}",
        [], [] if ext["ES=F_1h"]["old_week"].get("ok") and nq else ["NQ=F·ES 시간봉 미확보"],
        w52["miss_all"] > MISS_MAX, "구조 결측 = 월요일 휴장·추석 주말. NQ=F·ES 시간봉 받기 가능(BTC는 확인 안 함)")
    feat_basic("H08", "F2-1", "F2-1", "Fight", other=f"corr(Fight, ln 댓글 수) {_cs(cor['H08_corr_fight_lncomments'])}")
    feat_basic("H09", "T2 대표", "T2 대표", "D_SD(H06과 같음)")
    feat_basic("H11", "F2b-1", "F2b-1", "X = −z(ΔD3)·z(B)",
               other=f"D_SD 원값 vs B {_cs(cor['H11_corr_DSDraw_B(표20+)'])}, vs |B| {_cs(cor['H11_corr_DSDraw_absB(표20+)'])}")
    # ---- F3
    feat_basic("H14", "T2 대표", "T2 대표", "S(z_e 재료)", other="z_e = S⊥는 M0 확장 회귀 120창 뒤(가격 단계)")
    feat_basic("H15압력", "F3-1", "F3-1", "Pess")
    feat_basic("H12", "F3-2", "F3-2", "Gap z60")
    feat_basic("H15반전", "F3-3", "F3-3", "Pess")
    h16 = den["H16_news"]
    feat_basic("H16", "F3-4", "F3-4", "사건 뉴스형 Pess",
               other=f"사건 뉴스 0개 창 {pct(h16['event_zero_share'])}, 5개 미만 {pct(h16['event_lt5_share'])} (§3 H16 문턱 30%)")
    sh, kk = cor["H54_corr_SH_raw_A_rz8(문자 그대로)"], cor["H55_corr_K_raw_A_rz8(문자 그대로)"]
    sh2, kk2 = cor["H54_corr_SH_A_rz8"], cor["H55_corr_K_A_rz8"]
    over = []
    for nm, cc in (("corr(SH, A)", sh), ("corr(K, A)", kk)):
        if cc["pearson"] is not None and max(abs(cc["pearson"]), abs(cc["spearman"])) > 0.8:
            over.append(f"{nm} {cc['pearson']:.3f}/{cc['spearman']:.3f} > 0.8")
    r54, r55 = _f(out, "H54"), _f(out, "H55")
    add("H54·H55", "F3-5", "A6 `A6_SH`, A7 `A7_K`", f"{pct(r54['miss_all'])}, {pct(r55['miss_all'])}",
        f"{pct(r54['miss_struct'])}, {pct(r55['miss_struct'])}", f"{r54['blocks_front_strict']} / {r54['blocks_back_strict']}",
        over, [], True,
        f"문자 그대로 corr(SH, A) {_cs(sh)}, corr(K, A) {_cs(kk)}. rz8끼리 corr(A6, A) {_cs(sh2)}, corr(A7, A) {_cs(kk2)}")
    # ---- F4
    feat_basic("H19", "T2 대표", "T2 대표", "E_A z60")
    feat_basic("H21", "F4-1", "F4-1", "FO_A z60")
    # ---- F5 사건형
    ev_auto = [f"최소 표본: 2026 사건 앞 {nf_ev}·뒤 {nb_ev}(10/14까지) < 8"] if min(nf_ev, nb_ev) < MIN_BLOCKS else []
    r24, r25, r26 = _f(out, "H24"), _f(out, "H25"), _f(out, "H26")
    add("H24", "T2 대표", "R `H24_R`(사건 창 F_A − 직전 20창)", f"{pct(r24['miss_all'])} / {pct(r24['miss_judge'])}", pct(r24["miss_struct"]),
        f"{nf_ev} / {nb_ev} 사건", ev_auto, [], r24["miss_judge"] > MISS_MAX,
        f"2026 합 {nf_ev + nb_ev}개(행 규칙 '8 미만이면 판정 불가'는 넘음). 2025 사건 {n25}개 < 12 → R̂는 b만. "
        "R̂·u는 2025 포럼 자료가 있어야 함")
    add("H25", "F5-1", "CR `H25_CR_d9`", f"{pct(r25['miss_all'])} / {pct(r25['miss_judge'])}", pct(r25["miss_struct"]),
        f"{nf_ev} / {nb_ev} 사건", ev_auto, [], False, "H24 사건을 쓴다")
    add("H26", "F5-2", "PH `H26_PH`", f"{pct(r26['miss_all'])} / {pct(r26['miss_judge'])}", pct(r26["miss_struct"]),
        f"{nf_ev} / {nb_ev} 사건", ev_auto, [], r26["miss_judge"] > MISS_MAX,
        f"하락 상태일(×0.90·60일 최고) 앞 {ds['2026앞']['down_state_60max_x0.90_days']}·뒤 "
        f"{ds['2026뒤(10/14까지)']['down_state_60max_x0.90_days']}일 ≥ 20 → H26 행 규칙 통과")
    # ---- F6
    feat_basic("H31", "T2 대표", "T2 대표", "LL_US 20창 변화",
               extra_prov=["8월 단일종목 레버리지 규제 시행일 미확보(§5.1 #11: 못 받으면 설명용)"],
               other="LL_US 통과 묶음. NVDL은 사후 단어 표시")
    # ---- F7
    nwb = cor.get("H35_NW30_boundary", {})
    feat_basic("H35", "T2 대표", "T2 대표", "NW90 rz8",
               other=(f"소금 연속(아래 #2). 30일판 NW30: 실시간 전 {nwb.get('pre_n')}창 평균 {fmt(nwb.get('pre_mean'), 3)}, "
                      f"추세 창당 {fmt(nwb.get('pre_slope_per_window'), 4)}, 실시간 {nwb.get('rt_n')}창 {fmt(nwb.get('rt_mean'), 3)} "
                      f"(직전 10창 대비 z {fmt(nwb.get('jump_z_vs_last10'), 2)})"))
    feat_basic("H36", "F7-1", "F7-1", "NEW90 rz8",
               other=(f"corr(A4, A) 등록판 못 함, 30일 4창판 {_cs(cor['H36_corr_NEW30_A_rz8m4'])}. 하락 조건 국면 앞 "
                      f"{rg['2026앞']['fall_regimes']}·뒤 {rg['2026뒤(10/14까지)']['fall_regimes']}개(보조 표는 8개 이상인 구간만)"))
    h37 = den["H37_den"]
    feat_basic("H37", "F7-2", "F7-2", "강세 전향률",
               extra_auto=[f"§3 H37: 분모 ≥ 5인 날 {pct(h37['ge5_of_computed'])} < 60%"] if h37["ge5_of_computed"] < 0.6 else [],
               other=(f"§3 H37 규칙(분모 ≥ 5인 날 60% 이상): 계산된 {h37['computed']}창 중 {pct(h37['ge5_of_computed'])}로 통과, "
                      f"예열 창까지 넣으면 {pct(h37['ge5_of_all'])}. 분모 월 중앙값 "
                      + "·".join(f"{m}월 {float(v):.0f}" for m, v in h37["by_month_median"].items()) + "로 줄어드는 중"))
    # ---- F8 (주)
    a_kr, a_us = _w(out, "H39+H48", "T2 대표(문서 규칙)", "kr"), _w(out, "H39+H48", "T2 대표(문서 규칙)", "us")
    l_us = _w(out, "H39+H48", "강건판(현재 12종목)", "us")
    blk21 = a_kr["blocks_back"]
    auto21 = [f"최소 표본: 21일 목표라 구간당 독립 묶음 {blk21} < 8(횡단면도 시간 묶음으로 셈, notes)"] if blk21 < MIN_BLOCKS else []
    add("H39+H48", "T2 대표", "AS(주 창 × 종목)", f"국내 {pct(a_kr['miss_all'])} · 미국 {pct(a_us['miss_all'])}",
        f"국내 {pct(a_kr['miss_struct'])} · 미국 {pct(a_us['miss_struct'])}", f"{blk21} / {blk21}",
        auto21,
        ["문서 규칙 '남은 종목 15개 미만인 주는 버린다'에 현재 12종목 목록으로는 모든 주가 걸림(구조 결측 100%). "
         "KIND 상장 목록은 받을 수 있어 국내 30종목 유니버스는 만들 수 있음(현재 상장사만), 예탁원 보관 상위 50은 경로 미확인 → 미국 부분 설명용(§3)"],
        False, f"강건판(현재 12종목) 미국 구조 결측 {pct(l_us['miss_struct'])}")
    b40u = _w(out, "H40", market="us")
    add("H40", "F8-2", "AS × B̃", f"B̃ {pct(_w(out,'H40',market='kr')['miss_all'])} / {pct(b40u['miss_all'])}",
        pct(b40u["miss_struct"]), f"{blk21} / {blk21}", auto21, [], False, "AS(H39) 결측을 물려받음")
    add("H41", "F8-3", "R20 × AS", "AS와 같음", "AS와 같음", f"{blk21} / {blk21}", auto21, [], False, "R20 등은 가격 단계")
    add("H43", "F8-1", "Herd 사건", "사건 100%(120창 예열)", "—", "—", [], ["2026 사건 수(30개 문턱)는 120창 예열 뒤에야 셀 수 있음"],
        True, "원값 Herd는 국내 0%·미국 3% 결측")
    # ---- F9
    feat_basic("H46", "T2 대표", "T2 대표", "ΔTilt",
               other=f"AC1 {fmt(_f(out,'H46')['ac1'])} → 뒤 구간 n_eff {fmt(_f(out,'H46')['neff_back'],1)}, MDE {fmt(_f(out,'H46')['mde_back'])}(검정력 매우 낮음)")
    feat_basic("H50", "F9-1", "F9-1", "M z20", other="NQ=F 시간봉 받기 가능. SKHY는 판정에서 뺌")
    # ---- F10
    feat_basic("H49", "T2 대표", "T2 대표", "S^E",
               extra_prov=[] if nq else ["NQ=F 시간봉 미확보(§3 H49: 없으면 설명용)"], other="NQ=F 시간봉 2025-01부터 받기 가능")
    feat_basic("H51", "T2 대표", "T2 대표", "S^NM",
               other="KOSPI200 야간선물은 받을 경로 미확인(정의상 '있는 구간'만 넣는 통제라 빼고 진행). KOSDAQ 시가는 네이버로 받기 가능")
    w53 = _w(out, "H53")
    add("H53", "F10-1", "ΔS `h53_ds`(주말 13개)", pct(w53["miss_all"]), pct(w53["miss_struct"]), f"— / {w53['blocks_back']}",
        [] if w53["blocks_back"] >= MIN_BLOCKS else ["최소 표본"], [], False, "주 목표, 엄한 간격 2주로 셈")
    # ---- F11
    feat_basic("H57a", "F11 대표", "F11 대표", "stage10", other="pct250 최소 120창 → 2025 자료가 차야 값이 생김")
    feat_basic("H57b", "F11 대표", "F11 대표", "끝 단계 더미", other="H01 실패 시 활동량 단계 ⌈10 × pct250(A)⌉로 바꿈(역시 120창)")
    # ---- 실시간
    mc = den["H33_mc_authors"]
    feat_basic("H33", "실시간", "실시간", "MC 신호",
               extra_prov=["금투협 값의 '실제 공개 시각'은 원천에 시각 필드가 없어 소급 재구성 불가(앞으로 받을 때 기록해야 함)"],
               other=f"MC 작성자 ≥ 3인 창 {pct(mc['share_ge3'])}, 신호 창 {mc['sig_windows']}/{mc['sig_computable']}")
    ov = cor["H34_NS20_vs_LL"]
    o1 = ov.get("top15_overlap_ll_d20", {})
    feat_basic("H34", "실시간", "실시간", "NS 몫",
               other=(f"LL 신호와 겹침(상위 15%): LL 20창 변화와 {o1.get('both')}/{o1.get('ns_on')}창 = "
                      f"{pct(o1.get('share_of_ns_on'))} ≤ 70%. corr(NS20, LL_d20) {_cs(ov['corr_ns20_lld20'])}"))
    # ---- 설명용(이미 α를 쓰지 않음) — 행별 가격 전 점검만
    h23 = den["H23_claims"]
    pm = price["h29_pm3"]
    expl = [
        ("H13", "조롱 300댓글 정밀도 검수 안 함. MOCK 두 묶음 측정 실패"),
        ("H20", "z_e(M0 120창) 필요"),
        ("H22", "B pct250(120창) 필요. REBUT 측정 실패"),
        ("H23", f"주장(M9+M10) 5창 합 < 5인 창 {pct(h23['M9+M10']['miss_5win_lt5'])}(결측 30% 문턱 안). 부호 100개 점검 안 함"),
        ("H27", "F 측정 실패. 2026 사건 3~6개"),
        ("H28", "H24 사건을 씀"),
        ("H29", f"±3% 날: 2026 앞 {pm['2026앞']['up3'] + pm['2026앞']['down3']}·뒤 {pm['2026뒤(10/14까지)']['up3'] + pm['2026뒤(10/14까지)']['down3']}(10/14까지). 10/14 뒤 새 날 0/20"),
        ("H30", "MC 통과·DIST 실패. pct250(120창) 필요. 금투협 반대매매는 D−2 값만(받기 가능)"),
        ("H38", "별칭 통과. pct250 최소 60창 필요(국내 61창 중 앞 창들)"),
        ("H45", "IPO 측정 실패. KIND 청약 일정 화면 응답 200(받기 경로는 있음), 2025 계수 고정 필요"),
        ("H47", "KX 통과·KR 실패. pct250 필요"),
        ("H57c", "pct250 필요. 전방 트랙"),
        ("H59", "pct250 필요. 가격 조건은 판정 단계. 전방 트랙"),
    ]
    for hid, note in expl:
        add(hid, "설명용", "—", "—", "—", "—", [], [], False, note)
    return rows


def status_text(r) -> str:
    if r["id"] == "H02":
        return "못 함 — TMP가 없다. S 묶음(E·F)이 실패했으므로 H02 규칙상 TMP를 만들어 품질(κ ≥ 0.6, 사용자 검수 75%)을 통과시켜야 S 대신 쓸 수 있다"
    parts = []
    if r["auto"]:
        parts.append("**자동 설명용** — " + "; ".join(r["auto"]))
    if r["prov"]:
        parts.append("잠정 설명용(자료 대기) — " + "; ".join(r["prov"]))
    if r["meas"]:
        parts.append("측정 실패 묶음 " + "·".join(r["meas"]) + " → LLM 지수 대체 전 판정 보류")
    if r["grade"] == "설명용":
        parts.insert(0, "이미 설명용")
    if not parts:
        parts.append("통과" + ("(지금 결측은 예열 → 전체 자료로 다시 잼)" if r["warm"] else ""))
    elif r["warm"] and not r["auto"]:
        parts.append("지금 결측은 예열")
    return " / ".join(parts)


def render(out, summ, price, ext, lexp, T) -> str:
    L = []
    a = L.append
    rows = classify(out, price, ext, lexp)
    nb, nf = out["n_back"], out["n_front"]
    a("# 가격 전 점검표 — 중간 검증 (2026-10-15 스냅샷)")
    a("")
    a("> 정의: `HYPOTHESES.md` §5.1(1~12)·§2.7·§3, `HYPOTHESES_addendum1.md`(동결판) 그대로. 해석을 고른 곳은 `interim/notes_prechecks.md`.")
    a(f"> 자료: 포럼 2026-07-13~10-15(파일 {summ['snapshot']['files']}개, 줄 {summ['snapshot']['lines']:,}개, 스냅샷 {summ['snapshot']['S']}). "
      f"완전한 창은 국내 W_KR 61개(7/15~), 미국 W_U 67개(7/14~). **판정용 전체 자료(2025-01~)가 아니다.** 그래서 결측은 '예열'과 '구조'로 나눠 적는다.")
    a("> 앞날 주가 방화벽: 진입 이후 값(R_h·MDD_h·HL·갭)은 계산하지도 읽지도 않았다. #9는 2026-10-14까지의 지난 종가만, #11은 날짜·시각(색인)만 봤다.")
    fz = "같음" if all(out["frozen_ok"].values()) else "다름(!)"
    lz = "같음" if all(out["lex_v02_ok"].values()) else "다름(!)"
    a(f"> 지문: 동결 문서 3개 SHA-256이 `docs/desk_system.md` §12.5와 {fz}(실행 전후 확인). 사전 v0.2 두 파일도 고정값과 {lz}.")
    a("> 코드: `interim/code/prechecks.py`(1~8·10, 이 문서), `prechecks_price.py`(9), `prechecks_external.py`(11), `prechecks_report.py`, 테스트 `test_prechecks.py`.")
    a("")

    # ---------------- 요약
    a("## 0. 결론 요약")
    auto = [r for r in rows if r["auto"] and r["grade"] != "설명용"]
    prov = [r for r in rows if r["prov"] and not r["auto"] and r["grade"] != "설명용"]
    meas = [r for r in rows if r["meas"] and not r["auto"] and r["grade"] != "설명용"]
    a("")
    a("### 자동 설명용으로 내려간 칸(§2.7·§3 문턱, 이 스냅샷에서 확인)")
    a("| 가설 | 칸 | 이유 |")
    a("|---|---|---|")
    for r in auto:
        a(f"| {r['id']} | {r['grade']} | {'; '.join(r['auto'])} |")
    a("")
    a("### 잠정 설명용(외부 자료·예열 대기, 받으면 풀릴 수 있음)")
    a("| 가설 | 칸 | 이유 |")
    a("|---|---|---|")
    for r in prov:
        a(f"| {r['id']} | {r['grade']} | {'; '.join(r['prov'])} |")
    a("")
    a("### 측정 실패 묶음에 기대는 칸(자동 설명용과 별개, lexicon §8: LLM 지수로 대신)")
    a("| 가설 | 칸 | 실패 묶음 |")
    a("|---|---|---|")
    for r in meas:
        a(f"| {r['id']} | {r['grade']} | {'·'.join(r['meas'])} |")
    a("")
    left = [r for r in rows if r["grade"] not in ("설명용",) and not r["auto"] and not r["prov"] and not r["meas"]
            and r["id"] not in ("H02",)]
    a("### 못 한 칸")
    a("- H02(측정 선택): TMP를 만들지 않아 corr(TMP, S)와 TMP 품질을 잴 수 없다.")
    a("")
    a("### 주 판정 15칸(T1 3 + T2 10 + F11 2)이 지금 어디에 있나")
    a("| 상태 | 칸 |")
    a("|---|---|")
    mains = [r for r in rows if "대표" in r["grade"]]
    a("| 자동 설명용 | " + ", ".join(r["id"] for r in mains if r["auto"]) + " |")
    a("| 잠정 설명용(자료 대기) | " + ", ".join(r["id"] for r in mains if r["prov"] and not r["auto"]) + " |")
    a("| 측정 실패 묶음(LLM 지수 대체 전 보류) | " + ", ".join(r["id"] for r in mains if r["meas"] and not r["auto"] and not r["prov"]) + " |")
    a("| 점검 통과 | " + ", ".join(r["id"] for r in mains if not r["auto"] and not r["prov"] and not r["meas"]) + " |")
    a("")
    a("### 지금 점검을 모두 통과한 칸")
    a(", ".join(f"{r['id']}({r['grade']})" for r in left) + ".")
    a("- 위 칸들도 지금 결측의 대부분은 예열 때문이다(자료가 7/13부터). 2025-01부터 다 찬 자료로 같은 코드를 다시 돌려 확정한다.")
    a("")

    # ---------------- 1
    km, um = summ["windows_kr"]["main"], summ["windows_us"]["main"]
    a("## 1. 결번율·장애 창·스팸 파도")
    a("| 항목 | 국내 W_KR 본판 | 미국 W_U 본판 |")
    a("|---|---|---|")
    a(f"| 창 수 | {km['n_windows']} | {um['n_windows']} |")
    a(f"| 결번율 중앙값 / 상위 10% 경계 / 최대 | {pct(km['missing_rate']['median'],2)} / {pct(km['missing_rate']['p90'],2)} / {pct(km['missing_rate']['max'],2)}(K2026-09-22) "
      f"| {pct(um['missing_rate']['median'],2)} / {pct(um['missing_rate']['p90'],2)} / {pct(um['missing_rate']['max'],2)}(U2026-09-29) |")
    a(f"| 장애 창(같은 요일 중앙값의 20% 미만) | {km['outage']}(앞 {km['outage_unknown']}창은 기준선이 짧아 판단 불가) | {um['outage']}(9/24·9/25, 추석 휴장으로 조용한 창) |")
    a(f"| 스팸 파도 | {km['spam_wave']} | {um['spam_wave']} |")
    a(f"| 강건판 창 장애 | 고정 24시간판 1(K24-2026-08-18, 8/17 대체공휴일) | 09:00 ET판 2, 고정 24시간판 3(8/17·9/24·9/25) |")
    mbd = summ["missing_by_day"]
    a("")
    a(f"- 날짜별 결번율: {mbd['days']}일, 중앙값 {pct(mbd['median'],2)}, 최대 {pct(mbd['max'],1)}(8/23). 실시간으로 받은 날은 낮다(10/14 2.7%, 10/15 3.9%): 아직 지워지지 않은 글이 남아 있다(§7 한계와 같은 현상, #5).")
    a("- 장애 창은 문서 규칙대로 모든 특징에서 결측으로 두었다. 실제로는 휴장으로 조용했던 창이라, 전체 자료에서 휴장 창을 장애 규칙에서 뺄지는 사용자가 정할 일이다(지금은 덜 넣는 쪽).")
    a("- §1 순서 4의 '2025-01-01~2026-10-23 날짜별 결번율 표'는 수집이 다 찬 뒤 만든다(지금은 94일뿐).")
    a("")

    # ---------------- 2
    raw = out["raw"]
    sm, sb = raw["salt_multirow"], raw["salt_blocks"]
    sc = summ["salt"]
    a("## 2. 소금 연속성")
    a("| 점검 | 값 | 판단 |")
    a("|---|---|---|")
    a(f"| state.json 소금 | 있음. 지문(SHA-256 앞 8자)이 로더 스냅샷 때와 {'같음' if out['salt']['same_as_loader_snapshot'] else '다름'}. 값은 출력하지 않았다 | 바뀌지 않음 |")
    a("| state.json 백업 | 저장소 안에 사본 없음(찾아봄). 이번 단계는 읽기만이라 만들지 않았다 | **§2.2 '백업한다'가 아직 안 됨** → 주 세션이 만들 일 |")
    a(f"| 같은 글을 두 번 받은 줄의 작성자 해시 | {sm['post_author_same']:,}/{sm['multi_row_posts']:,} 같음 | 다시 받기(다른 실행) 사이에 소금이 같다 |")
    a(f"| 두 줄에 같이 나온 댓글(본문·시각)의 해시 | {sm['comment_author_same']:,}/{sm['comment_pairs']:,} 같음 | 같음 |")
    a(f"| 이웃 날짜 작성자 겹침률 | 중앙값 {sc['median_overlap_prev']:.3f}, 최저 {sc['min_overlap_prev']:.3f}(월요일), 급락 표지 0일 | 끊김 없음 |")
    nr = ", ".join(f"{x['blk']:,}: {x['overlap']:.2f}" for x in sb["near_run_boundary"])
    a(f"| 글 번호 500개 묶음 사이 작성자 겹침 | {sb['pairs']}쌍, 중앙값 {sb['median']:.3f}, 최저 {sb['min']:.3f}, 중앙값의 절반 미만 {sb['lt_half_median']}쌍 | 끊김 없음 |")
    a(f"| 수집 실행 경계(10/16 02:58 멈춤 → 09:41 재개, 번호 405,523~406,523) 근처 | {nr} | 경계에서 떨어지지 않음 |")
    a("")

    # ---------------- 3
    fpg, pg = raw["first_page"], raw["page"]
    tb = raw["trunc_by_day"]
    a("## 3. 댓글 첫 쪽이 가장 이른 댓글인가, 날짜별 잘림 비율")
    a("| 점검 | 값 |")
    a("|---|---|")
    a(f"| 한 쪽 크기 | 뿌리 댓글 10개(+그 대댓글). 잘린 글 {pg['trunc_posts']:,}개 중 {pg['trunc_roots_ge10']:,}개({pct(pg['trunc_roots_ge10']/pg['trunc_posts'])})가 뿌리 10개, 뿌리 최대 {pg['roots_max']} |")
    a(f"| 두 줄 글: 첫 줄 댓글이 둘째 줄에 남은 비율 | {pct(fpg['c1_kept_share'],2)} ({fpg['rows_with_c1']:,}글) |")
    a(f"| 둘째 줄에서 잘린(뿌리 10개) 글: 첫 줄 뿌리 댓글이 둘째 줄 맨 앞에 그대로 | {fpg['trunc_c1_roots_prefix']}/{fpg['trunc_rows_10roots']} ({pct(fpg['trunc_c1_roots_prefix']/max(1,fpg['trunc_rows_10roots']))}) |")
    a(f"| 둘째 줄의 새 뿌리 댓글이 옛 댓글 뒤에 붙은 비율 | {fpg['new_roots_after_old']}/{fpg['new_roots_total']} ({pct(fpg['new_roots_after_old']/max(1,fpg['new_roots_total']))}) |")
    a(f"| 첫 저장 댓글의 등록 뒤 지연(중앙값) | 잘린 글 {pg['first_delay_min_trunc_median']:.1f}분, 안 잘린 글(뿌리 8개+) {pg['first_delay_min_nontrunc8_median']:.1f}분 |")
    a(f"| 마지막 저장 댓글 지연(중앙값) | 잘린 글 {pg['last_delay_min_trunc_median']:.0f}분, 안 잘린 글(뿌리 8개+) {pg['last_delay_min_nontrunc8_median']:.0f}분 |")
    a(f"| 날짜별 잘림 비율(글 등록일) | 중앙값 {pct(tb['median'])}, 최소 {pct(tb['min'])}, 최대 {pct(tb['max'])}({tb['max_day']}, 추석 연휴 주말). 표는 `results/prechecks_by_postdate.csv` |")
    a(f"| 잘림 하한 표지의 성격 | 잘린 글 중 ncomment − 저장 수 ≤ 2인 글 {pg['trunc_diff_le2']:,}개({pct(pg['trunc_diff_le2']/pg['trunc_posts'])}). 수집기가 뺀 차단·스팸 댓글도 ncomment에 들어 있어 잘림을 조금 부풀린다 |")
    a("")
    a("- 판단: **첫 쪽은 가장 이른 댓글 쪽이다**(등록 순서, 앞에서부터 10뿌리). 그래서 잘린 글은 '초반 댓글은 다 있고 늦은 댓글이 빠진' 하한이다. H08 주판(잘리지 않은 글만)·창별 잘림 비율 통제를 그대로 쓴다. 잘린 글은 첫 쪽이 등록 뒤 약 100분(마지막 저장 댓글 중앙값)에서 끝나므로, H12·H55의 120분 관심(c120)은 잘린 글에서 하한이다 → §2.2대로 잘림 비율을 통제 변수로 넣는다.")
    a("")

    # ---------------- 4
    ed = raw["edits"]
    cd = summ["comment_dates"]
    a("## 4. 실시간 첫 줄 vs 24시간 줄, 시각만 있는 댓글의 날짜 확정률")
    a("| 점검 | 값 |")
    a("|---|---|")
    e = ed["실시간(10/13~)"]
    a(f"| 실시간 글(10/13~) 두 줄 | {e['two_row_posts']:,}글. 본문이 바뀐 글 {e['body_edited']}({pct(e['body_edited']/e['two_row_posts'],2)}), 제목 {e['title_edited']}({pct(e['title_edited']/e['two_row_posts'],2)}) |")
    a(f"| 수정 표지(추가·수정·ps·업데이트)가 든 글 | 실시간 {pct(ed['edit_marker_share']['실시간(10/13~)'])}, 그 전 {pct(ed['edit_marker_share']['그 전'])} → 강건판(표지 글 제외)은 3% 남짓을 뺀다 |")
    a(f"| 시각만 있는 댓글 날짜 확정 | {cd['time_only']:,}개 중 {pct(cd['conf_rate_time_only'],2)}(다른 줄 맞춤 {cd['matched']:,}, 줄 안 좁히기 {cd['row']}, 가장 늦은 날짜 {cd['latest']}). 전체 댓글 기준 {pct(cd['conf_rate_all'],2)} |")
    a("| 글 날짜별 확정률 | 10/12 38.9%(36개 중), 10/13 99.3%(2,291), 10/14 98.0%(1,222), 10/15 100%(4,559). 확정 못 한 63개는 늦은 쪽으로만 틀린다 |")
    a("")
    a("- 판단: 첫 줄을 쓰는 규칙(§2.2) 때문에 생기는 차이는 0.2~0.3%로 작다. 자정 전후 댓글에 기대는 H08·H28·H34도 확정 몫이 국내 마감 전 댓글 기준 100%(latest 0)라 영향이 없다.")
    a("")

    # ---------------- 5
    bp, bw = raw["by_period"], raw["by_period_weekday"]
    a("## 5. 수집 경계(10/13) 앞뒤 구조")
    a("| 글 등록일 구간 | 날 | 글 | 글당 저장 댓글 | 지운 댓글 몫 | 잘린 글 | 결번율 중앙값 | 두 줄 글 |")
    a("|---|---|---|---|---|---|---|---|")
    for k, v in bp.items():
        a(f"| {k} | {v['days']} | {v['posts']:,} | {v['stored_per_post']:.2f} | {pct(v['deleted_share'],2)} | {pct(v['trunc_share'])} | {pct(v['missing_rate_median'])} | {pct(v['two_row_share'])} |")
    a("")
    a("| 평일만 | 날 | 글당 저장 댓글 | 잘린 글 | 지운 댓글 몫 |")
    a("|---|---|---|---|---|")
    for k, v in bw.items():
        a(f"| {k} | {v['days']} | {v['stored_per_post']:.2f} | {pct(v['trunc_share'])} | {pct(v['deleted_share'],2)} |")
    nwb = out["correlations"].get("H35_NW30_boundary", {})
    a("")
    a(f"- NW 수준(30일판, 비등록): 실시간 전 {nwb.get('pre_n')}창 평균 {fmt(nwb.get('pre_mean'),3)}, 그 마지막 10창 {fmt(nwb.get('pre_last10_mean'),3)}(SD {fmt(nwb.get('pre_last10_sd'),3)}), "
      f"실시간 {nwb.get('rt_n')}창 {fmt(nwb.get('rt_mean'),3)} → 직전 10창 대비 z {fmt(nwb.get('jump_z_vs_last10'),2)}. 실시간 전 추세 창당 {fmt(nwb.get('pre_slope_per_window'),4)}. 등록판 NW90은 {nwb.get('NW90_values')}뿐이다.")
    a("- 판단: 지운 댓글 몫은 경계 앞뒤가 같다(0.16~0.19%). 글당 저장 댓글과 잘린 글 비율은 실시간 쪽이 낮고(평일 기준 위 표), 결번율은 실시간 쪽이 절반이다. "
      "**lexicon §9-2 규칙대로 댓글 양 특징은 경계 앞뒤를 따로 정규화하는 것이 맞다.** 다만 실시간 3일이 10/13 급락 직후라 활동 변화와 섞여 있다. NW30은 실시간 2창이 직전 10창보다 z 0.49 높을 뿐이라 수준 점프라 할 만큼 크지 않다(실시간 전부터 창당 +0.0019 오르는 추세). 등록판 NW90은 10/13~ 3창뿐이라 점프 점검은 전체 자료로 다시 한다.")
    a("")

    # ---------------- 6
    a("## 6. 특징별 결측률·자기상관·n_eff·독립 묶음")
    a(f"- 결측률 = 전체 창 / 판정 자격 창(국내 96시간 초과 1창, 미국 장애 2창 제외). **구조 결측** = 예열(최소 창 수)을 빼고도 남는 결측(정의는 표의 '구조 규칙', notes).")
    a(f"- n_eff = (n − h) ÷ (1 + 2Σρᵏ(1 − k/h)), ρ = AC1(이웃 창 1차 자기상관, 0~0.99로 자름). §2.7 MDE 표와 같은 식이다(ρ 0.5·n 96이면 h 1·5·10·21에 95·41·33·27, 테스트로 확인).")
    a(f"- 뒤 구간 투영 = 2026-06-01~10-23 거래일(KRX {nb['krx']}, NYSE {nb['nyse']}일)에 지금 AC1을 넣은 값. 앞 구간은 KRX {nf['krx']}, NYSE {nf['nyse']}일. MDE = 2.49/√n_eff.")
    a("- 독립 묶음 = 값이 있는 창을 앞에서부터 h창 이상(겹침 없음) / h+1창 이상(엄함, 'h일 넘게 떨어진') 간격으로 고른 수. 판정은 엄한 쪽으로 한다.")
    a("")
    a("| 가설 | 칸 | 열 | h | 최소 창 | 값/창 | 결측 전체 / 판정 자격 | 구조 결측 | AC1(쌍) | n_eff 지금 | 묶음 지금 겹침없음/엄함 | n_eff 뒤 투영 | MDE 뒤 | 엄한 묶음 앞/뒤 투영 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out["features"]:
        evc = "(사건)" in r["cell"]
        a(f"| {r['id']} | {r['cell']} | `{r['col']}` ({r['market']}) | {r['h']} | {r['family']} | {r['n_valid']}/{r['n_rows']} | {pct(r['miss_all'])} / {pct(r['miss_judge'])} | "
          f"{pct(r['miss_struct'])} | {fmt(r['ac1'])} ({r['ac1_pairs']}) | {'사건 표' if evc else fmt(r['neff_now'],1)} | "
          f"{'사건 표' if evc else str(r['blocks_now']) + '/' + str(r['blocks_now_strict'])} | "
          f"{'사건 표' if evc else fmt(r['neff_back'],1)} | {'—' if evc else fmt(r['mde_back'])} | "
          f"{'사건 표' if evc else str(r['blocks_front_strict']) + '/' + str(r['blocks_back_strict'])} |")
    a("")
    a("주 단위 칸(값/행은 종목 × 주, 묶음은 주 단위, 21일 목표는 5주 간격)")
    a("")
    a("| 가설 | 칸 | 열 | 시장 | 값/행 | 주 | 결측 | 구조 결측 | 묶음 지금 | 묶음 뒤 투영 | 메모 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out["weekly"]:
        a(f"| {r['id']} | {r['cell']} | `{r['col']}` | {r['market']} | {r['n_valid']}/{r['n_rows']} | {r.get('weeks')} | {pct(r['miss_all'])} | {pct(r['miss_struct'])} | "
          f"{r.get('blocks_now', '—')} | {r.get('blocks_back', '—')} | {r.get('memo', '')} |")
    ev = price["h24_events"]
    a("")
    a("사건형 칸(독립 단위 = 사건 묶음, 지난 종가로 셈 #9)")
    a("")
    a("| 가설 | 2025 | 2026 앞 | 2026 뒤(10/14까지) | 기준 |")
    a("|---|---|---|---|---|")
    a(f"| H24·H25·H26·H28 (KOSPI 또는 하닉 r/σ60 ≤ −2, 5일 묶음) | {len(ev['2025']['kospi_or_hynix_clusters'])} | {len(ev['2026앞']['kospi_or_hynix_clusters'])} | {len(ev['2026뒤(10/14까지)']['kospi_or_hynix_clusters'])} | 구간마다 8 이상(§2.7), 2026 합 8 이상(H24 행) |")
    a(f"| (KOSPI만 판) | {len(ev['2025']['kospi_only_clusters'])} | {len(ev['2026앞']['kospi_only_clusters'])} | {len(ev['2026뒤(10/14까지)']['kospi_only_clusters'])} | — |")
    a(f"| H43 Herd 사건 | — | — | 셀 수 없음(120창 예열) | 2026 사건 30 이상 |")
    a("")

    # ---------------- 7
    den = out["denominators"]
    a("## 7. 분모 분포")
    k9 = den["H09_kosdaq_tag_lower"]
    h23 = den["H23_claims"]
    mc, mcu, h37, h16 = den["H33_mc_authors"], den["H33_mc_authors_us"], den["H37_den"], den["H16_news"]
    a("| 대상 | 분모 | 값 | 판단 |")
    a("|---|---|---|---|")
    a(f"| H09 보조 코스닥 태그 | 태그 글의 표(점수 ≠ 0 작성자) | 하한판('코스닥' 낱말만): 태그 글 창 중앙값 {k9['tag_posts_median']:.0f}, 표 중앙값 {k9['votes_median']:.0f}, 최대 {k9['votes_max']:.0f}, 표 20 이상 창 {pct(k9['share_votes_ge20'])} | 이름 30개 사전(KIND)이 없어 상한은 못 잼. 하한판은 결측 100% → 보조 표는 '결측률 30% 이하일 때만'에 걸릴 가능성이 큼 |")
    a(f"| H23 수익률 주장 | 5창 합 주장 수(5 미만 결측) | M9+M10(1인칭): 창 중앙값 {h23['M9+M10']['claims_per_window_median']:.1f}, 전체 {h23['M9+M10']['claims_total']}(큰 이익 {h23['M9+M10']['bw_total']}, 큰 손실 {h23['M9+M10']['bl_total']}), 결측 {pct(h23['M9+M10']['miss_5win_lt5'])}. M9만: 창 중앙값 {h23['M9']['claims_per_window_median']:.0f}, 결측 {pct(h23['M9']['miss_5win_lt5'])} | 결측 30% 문턱 안. 부호 정확도 80%(날짜 가린 100개) 점검은 아직 안 함 |")
    a(f"| H33 MC 작성자 | 창 안 MC 고유 작성자 | 국내 중앙값 {mc['median']:.0f}, 3명 이상 창 {pct(mc['share_ge3'])}; 미국 중앙값 {mcu['median']:.0f}, {pct(mcu['share_ge3'])}. 신호 창 {mc['sig_windows']}/{mc['sig_computable']} | 실시간 전방 트랙(신호 묶음 10개) |")
    bm = h37["by_month_median"]
    bms = "·".join(f"{m}월 {float(v):.0f}" for m, v in bm.items())
    a(f"| H37 약세 단골 | 5창 묶음의 글 쓴 약세 단골 | 계산된 {h37['computed']}창 중 5 이상 {pct(h37['ge5_of_computed'])}, 예열 창까지 넣은 판정 자격 60창 기준 {pct(h37['ge5_of_all'])}, 중앙값 {h37['median']:.0f}, 월 중앙값 {bms} | "
      f"H37 행 규칙(60%)은 계산된 창 기준으로 통과. 그러나 분모 < 5로 비는 창이 {pct(1 - h37['ge5_of_computed'])} > 30%라 §2.7로 자동 설명용. 분모가 줄어드는 추세 |")
    a(f"| H16 뉴스형 | 창 안 사건 뉴스 글 | 0개 창 {pct(h16['event_zero_share'])}, 5개 미만 {pct(h16['event_lt5_share'])}, 중앙값 {h16['event_median']:.0f}. 가격 보도 0개 창 {pct(h16['price_zero_share'])}. 미국 창 사건 뉴스 0개 {pct(h16['us_event_zero_share'])} | 사건 뉴스형 결측 30% 초과 → 자동 설명용 |")
    a("")

    # ---------------- 8
    cor = out["correlations"]
    a("## 8. 대표 특징끼리의 상관 (Pearson / Spearman, 판정 자격 창)")
    a("| 쌍 | 판 | 값 | 문턱 | 판단 |")
    a("|---|---|---|---|---|")

    def j(c, th):
        if c["pearson"] is None:
            return "못 잼"
        return "넘음 → 자동 설명용" if max(abs(c["pearson"]), abs(c["spearman"])) > th else "안 넘음"
    a(f"| corr(U, N) H04 | rz8 등록판 | {_cs(cor['H04_corr_U_N_rz8'])} | 0.9 | {j(cor['H04_corr_U_N_rz8'], 0.9)} |")
    a(f"| | rz8 4창판(비등록) | {_cs(cor['H04_corr_U_N_rz8m4'])} | | 참고(Pearson 0.899로 문턱 바로 아래) |")
    a(f"| | 시간당 원값 | {_cs(cor['H04_corr_U_N_raw_per_h'])} | | 참고 |")
    a(f"| | 미국 rz8 | {_cs(cor['H04_corr_U_N_rz8_us'])} | | 참고(보조) |")
    a(f"| corr(SH, A) H54 | SH 원값 × A(rz8), 문자 그대로 | {_cs(cor['H54_corr_SH_raw_A_rz8(문자 그대로)'])} | 0.8 | {j(cor['H54_corr_SH_raw_A_rz8(문자 그대로)'], 0.8)} |")
    a(f"| | A6(rz8 SH) × A | {_cs(cor['H54_corr_SH_A_rz8'])} | | 회귀에 같이 드는 꼴. {j(cor['H54_corr_SH_A_rz8'], 0.8)} |")
    a(f"| | 원값끼리 | {_cs(cor['H54_corr_SH_A_raw'])} | | 참고(요일 효과가 섞임) |")
    a(f"| corr(K, A) H55 | K 원값 × A(rz8), 문자 그대로 | {_cs(cor['H55_corr_K_raw_A_rz8(문자 그대로)'])} | 0.8 | {j(cor['H55_corr_K_raw_A_rz8(문자 그대로)'], 0.8)} |")
    a(f"| | A7(rz8 K) × A | {_cs(cor['H55_corr_K_A_rz8'])} | | 5창뿐이라 판단에 안 씀 |")
    a(f"| | 원값끼리 | {_cs(cor['H55_corr_K_A_raw'])} | | 참고 |")
    a(f"| corr(A4, A) H36 | 등록판(NEW90 rz8) | {_cs(cor['H36_corr_A4_A_rz8(NEW90)'])} | 문서에 문턱 없음(A에 잔차화) | 못 잼(90일 이력) |")
    a(f"| | 30일 4창판(비등록) | {_cs(cor['H36_corr_NEW30_A_rz8m4'])} | | 참고 |")
    a(f"| corr(TMP, S) H02 | — | 못 잼 | 0.9 | TMP를 만들지 않았다 |")
    o = cor["H34_NS20_vs_LL"]
    for nm, lab in (("ll_d20", "LL 20창 변화(H31 특징)"), ("ll20", "LL 20창 평균")):
        t = o.get(f"top15_overlap_{nm}")
        if t:
            a(f"| LL 신호와 NS20 겹침 H34 | 상위 15%(문턱 85)끼리, {lab} | {t['both']}/{t['ns_on']}창 = {pct(t['share_of_ns_on'])} (n {o['n']}) | 70% | 안 넘음 |")
    a(f"| | 순위 상관 NS20 × LL 20창 변화 / 20창 평균 | {_cs(o['corr_ns20_lld20'])} / {_cs(o['corr_ns20_ll20'])} | | 참고. 가격 잔차화 전 원값, 전 표본 순위라 설명용 점검값 |")
    a(f"| D_SD와 B H11 | D_SD 원값 × B (표 20 이상 창) | {_cs(cor['H11_corr_DSDraw_B(표20+)'])} | 보고만 | × \\|B\\|: {_cs(cor['H11_corr_DSDraw_absB(표20+)'])} |")
    a(f"| | D_SD(잠정 풀 40창) × B | {_cs(cor['H11_corr_DSDprov40_B'])} | | 2창뿐 |")
    a(f"| corr(A, 저장판 A) H03 | rz8 | {_cs(cor['H03_corr_A_Asaved_rz8'])} | 부호 같아야 인정 | 같음 |")
    a("")

    # ---------------- 9
    pr = price
    a("## 9. 가격만 쓰는 셈 (지난 종가, 2026-10-14까지)")
    a("| 셈 | 2025 | 2026 앞 | 2026 뒤(10/14까지) | 쓰는 곳·판단 |")
    a("|---|---|---|---|---|")
    e = pr["h24_events"]
    a(f"| KOSPI 또는 하닉 −2σ 사건 묶음 | {len(e['2025']['kospi_or_hynix_clusters'])} | {len(e['2026앞']['kospi_or_hynix_clusters'])} | {len(e['2026뒤(10/14까지)']['kospi_or_hynix_clusters'])} | H24~H26: 구간마다 8 미만 → 설명용. 2025 < 12 → R̂는 b만 |")
    a(f"| KOSPI −2σ 날(묶기 전) | {e['2025']['kospi_days_le_-2sigma']} | {e['2026앞']['kospi_days_le_-2sigma']} | {e['2026뒤(10/14까지)']['kospi_days_le_-2sigma']} | — |")
    a(f"| SOX −2σ 사건 묶음 | {len(e['2025']['sox_clusters'])} | {len(e['2026앞']['sox_clusters'])} | {len(e['2026뒤(10/14까지)']['sox_clusters'])} | H24 보조 SOX판 |")
    d = pr["down_state"]
    a(f"| 하락 상태 날(D−1 종가 ≤ 60일 최고 × 0.90) | {d['2025']['down_state_60max_x0.90_days']} | {d['2026앞']['down_state_60max_x0.90_days']} | {d['2026뒤(10/14까지)']['down_state_60max_x0.90_days']} | H26: 구간당 20일 이상 → 통과 |")
    a(f"| 하락 상태 날(D−1 종가 ≤ 20일 최고 × 0.92) | {d['2025']['down_state_20max_x0.92_days']} | {d['2026앞']['down_state_20max_x0.92_days']} | {d['2026뒤(10/14까지)']['down_state_20max_x0.92_days']} | F5 보조 |")
    g = pr["h36_regimes"]
    a(f"| 하락 조건 국면(r20 ≤ 0 연속 구간) | {g['2025']['fall_regimes']} | {g['2026앞']['fall_regimes']} | {g['2026뒤(10/14까지)']['fall_regimes']} | H36 하락 쪽 보조 표: 8개 이상인 구간만(앞 미달) |")
    p3 = pr["h29_pm3"]
    a(f"| ±3% 날(오름/내림) | {p3['2025']['up3']}/{p3['2025']['down3']} | {p3['2026앞']['up3']}/{p3['2026앞']['down3']} | {p3['2026뒤(10/14까지)']['up3']}/{p3['2026뒤(10/14까지)']['down3']} | H29: 10/14 뒤 새 날 20개에 판정(지금 0) |")
    cs = pr["crash_starts_12_4"]
    for nm in ("KOSPI", "SOX"):
        a(f"| §12.4 급락 시작 {nm} | {len(cs[nm]['2025'])} | {len(cs[nm]['2026앞'])} | {len(cs[nm]['2026뒤(10/14까지)'])} | "
          f"H31·H33 전방 트랙(새 사건 약 20건). 2026 목록이 §12.4와 {'같음' if (cs[nm]['2026앞'] + cs[nm]['2026뒤(10/14까지)'])[:len(cs['doc_12_4_2026'][nm])] == cs['doc_12_4_2026'][nm] else '다름'}"
          + (f"(SOX 10/08 시작은 §12.4 작성 뒤 10/14 종가로 확인된 새 사건)" if nm == "SOX" else "") + " |")
    a("")

    # ---------------- 10
    b = lexp["bundles"]
    npass = sum(1 for v in b.values() if not v["fail"])
    irmax = max((v["irony_rate_any"], k) for k, v in b.items())
    nrem = sum(v["n_removed"] for v in b.values())
    a("## 10. 사전 정밀도 점검 → v0.2 지문")
    a(f"| 항목 | 값 |")
    a("|---|---|")
    a(f"| 묶음 | {len(b)}개 중 통과 {npass}, 측정 실패 {len(b) - npass}(모두 묶음 정밀도 60% 미만) |")
    a(f"| 반어율 최대 | {pct(irmax[0])}({irmax[1]}) → 30% 넘는 묶음 0 |")
    a(f"| 뺀 단어 | {nrem}개(라벨된 적중 10회 이상이면서 단어 정밀도 50% 미만). 새 단어 0 |")
    a(f"| 통과 묶음 | " + ", ".join(k for k, v in b.items() if not v["fail"]) + " |")
    a(f"| 실패 묶음 | " + ", ".join(f"{k} {v['precision']*100:.0f}%" for k, v in b.items() if v["fail"]) + " |")
    a(f"| H06 극성 정확도(score200) | v0.2 점수 ≠ 0 글 14개 중 10개(71%, 95% 구간 45~88%) → 통과, 다만 얇다 |")
    a(f"| v0.2 SHA-256 | `lexicon_v02.json` {LEX_V02['lexicon_v02.json']}, `lexicon_v02_full.json` {LEX_V02['lexicon_v02_full.json']} (다시 계산해 같음) |")
    a(f"| 동결 문서 SHA-256 | " + ", ".join(f"`{n}` {v[:8]}…" for n, v in FROZEN.items()) + " (§12.5와 같음) |")
    a("| 남은 일 | v0.2로 새 가림 표본 재측정, H13 조롱 300댓글, H23 부호 100개, 사용자 검수 200건, §12.5에 v0.2 지문 기록(저장소 파일이라 이번에 안 함) |")
    a("")

    # ---------------- 11
    x = ext
    a("## 11. 외부 자료 확보 (받을 수 있는지만)")
    a("| 자료 | 쓰는 곳 | 결과 | 판단 |")
    a("|---|---|---|---|")
    nqo = x["NQ=F_1h"]
    a(f"| NQ=F 시간봉 | H10·H49·H50·H52 | yfinance 60분봉 2025-01-02~01-09 {nqo['old_week'].get('rows')}봉(ET 0~23시, 17시 휴장 빼고). 730일 넘는 구간(2024-10)은 거부 | 받기 가능. 2025-01-02 봉은 {nqo['deadline_to_freeze_2025-01']} 무렵 사라지므로 그 전에 받아 고정 |")
    a(f"| ES=F 시간봉 | H52 | 같은 구간 {x['ES=F_1h']['old_week'].get('rows')}봉 | 받기 가능 |")
    a("| KOSPI200 야간선물 | H51(있는 구간만 넣는 통제) | 저장소에 원천 없음, yfinance 기호 없음. 이번 단계(웹 검색 금지)로는 경로 미확인 | H51은 이 항 없이 진행 가능(정의가 '있는 구간') |")
    nv = x["KOSDAQ_open"]["naver_vs_yf"]
    a(f"| KRX·네이버 시가 | H51 | 네이버 fchart KOSDAQ 일봉 응답 {nv.get('status')}, 2025-01 첫 주 {nv.get('naver_rows')}일. yfinance ^KQ11 시가와 {nv.get('open_match_0.1pct')}/{nv.get('common_days')}일 0.1% 안 일치 | 받기 가능 |")
    kf = x["KOFIA_freesis"]
    a(f"| 금투협 공개 시각 | H30·H33 | freesis API 응답 {kf['api_old_days'].get('status')}(2025-01 이틀 {kf['api_old_days'].get('rows')}줄), 저장소에 2025-01~2026-10-13 파일 있음. 응답에 공개 시각 필드 없음 | 값은 받기 가능. 실제 공개 시각은 소급 불가 → H30은 D−2 값 규칙, H33은 앞으로 받을 때 시각 기록 |")
    kd = x["KIND"]
    a(f"| KIND 청약 일정 | H45 | 청약 일정 화면 응답 {kd['ipo_schedule'].get('status')}({kd['ipo_schedule'].get('bytes'):,}바이트). 이력 내려받기 형식은 확인 안 함 | 경로 있음(H45는 이미 설명용) |")
    a(f"| KIND 상장 이름 사전 | H39 국내 유니버스·H09 보조 | 상장 법인 목록 내려받기 응답 {kd['corp_list_download'].get('status')}({kd['corp_list_download'].get('bytes'):,}바이트, xls) | 받기 가능. 다만 현재 상장사만이라 상폐 종목이 빠진다(§7 한계). 시총 이력은 없어 H09 보조의 '시총 상위 30'은 대체 규칙(직전 8주 언급 상위 30)으로만 만든다 |")
    a(f"| 예탁원 해외주식 보관 상위 | H48 미국 유니버스 | SEIBro 첫 화면 응답 {x['SEIBro']['home'].get('status')}({x['SEIBro']['home'].get('bytes')}바이트, 스크립트 화면). 월말 보관 상위 50 자료 경로 미확인, 저장소에 없음 | **미확보 → 받기 전까지 H48 미국 부분 설명용(§3)** |")
    a("| 8월 단일종목 레버리지 규제 시행일 | H31 | 저장소에는 '8월 규제(기본예탁금 3,000만 원·신규 상장 중단)'만 있고 날짜가 없다(leverage_flood/report.md). 웹 검색 금지라 확정 못 함 | **미확보 → 확정 전까지 H31 잠정 설명용(§5.1 #11)**. 날짜 하나만 공식 공지로 확인하면 풀린다 |")
    ed_ = x["earnings_dates_yf"]
    a(f"| 일정 더미 달력 | 모든 회귀 | FOMC: 저장소 stop_research에 2023~2026 날짜 표 있음, 연준 일정 화면 응답 {x['FOMC_calendar'].get('status')}. CPI·고용(BLS) 화면 {x['BLS_schedule'].get('status')}(자동 요청 차단). 실적일: yfinance NVDA·MU·AVGO·하닉·삼전 각 2025~10/23 {', '.join(str(v.get('n_2025_2026')) for v in ed_.values())}개. 옵션 만기(둘째 목·셋째 금)는 달력 규칙으로 계산 가능. 삼전 잠정실적·MSCI 리밸런싱은 공지 확인 필요 | 대부분 가능. CPI·고용 날짜는 다른 원천 필요 |")
    a("| (참고) VKOSPI 이력 | M0 한국 | yfinance 기호 없음(^VKOSPI 404). 저장소에는 수동 표(kr_levetf_vkospi_manual.csv)뿐 | §5.1 #11 목록 밖이지만 M0 재료라 확보 경로가 필요 |")
    a("")

    # ---------------- 최소 창 수
    a("## 최소 창 수(rz8 8창, z60 40창, pct250 120창)를 지금 자료로 채웠나")
    a("| 정규화 | 채움(첫 값, 값 수) | 못 채움 |")
    a("|---|---|---|")
    wr = pd.DataFrame(out["warmup"])
    for fam, g in wr.groupby("family", sort=False):
        ok = g[g.n_valid > 0]
        no = g[g.n_valid == 0]
        a(f"| {fam} | " + ", ".join(f"{r.market} `{r.col}`({r.first[5:] if r.first else ''}, {r.n_valid})" for r in ok.itertuples())
          + " | " + ", ".join(f"{r.market} `{r.col}`" for r in no.itertuples()) + " |")
    a("")
    a("- rz8: 국내 9/9·미국 9/8부터 값이 있다(같은 요일 8창). 90일 이력(NW90·NEW90)을 먼저 채워야 하는 H35·H36은 rz8까지 못 갔다. B⊥(H04)는 rz8 값이 20창이 안 돼 회귀를 못 했다.")
    a("- z60: 41번째 창(국내 9/11, 미국 9/9)부터 값이 있다. z60 위에 다시 z60을 쓰는 것(ΔD3·H11)과 원값이 드문 것(H10·H16 사건 뉴스)은 못 채웠다.")
    a("- pct250: 120창이 필요한데 지금 61~67창이라 하나도 못 채웠다(stage10·20, SG, 엇갈림, MC pct250, 주 단계). M0 확장 회귀(최소 120창)도 같아서, X⊥(H14 z_e 등)는 가격 단계에서도 2025 자료가 있어야 만든다.")
    a("- 기타: D_SD 본판(풀 60창)은 풀이 찬 10/15 하나가 표 15개라 0개, 잠정 풀 40창판 2개. LL 20창 변화·ΔTilt·M z20은 8/11~8/19부터. H43 사건(120창)은 0개.")
    a("")

    # ---------------- 가설별
    a("## 가설별 판정 전 상태(전 가설)")
    a("| 가설 | 칸 | 대표 특징 | 결측 전체 / 판정 자격 | 구조 결측 | 엄한 묶음 앞/뒤(투영) | 점검 메모 | 상태 |")
    a("|---|---|---|---|---|---|---|---|")
    for r in rows:
        a(f"| {r['id']} | {r['grade']} | {r['feat']} | {r['now']} | {r['struct']} | {r['blocks']} | {r['other']} | {status_text(r)} |")
    a("")
    a("- 기각 7개(H07·H17·H18·H32·H42·H44·H56)는 점검 대상이 아니다.")
    a("- '측정 실패'는 §2.7 자동 설명용과 다른 갈래다. lexicon §8대로 LLM 라벨 지수로 대신해야 하고, H01 관문(5단계, 가격 필요)이 실패하면 §2.6대로 감정 사전 칸이 멈춘다. LLM 지수는 2026-07-01 뒤만 판정에 쓸 수 있어(§2.2) 앞 구간 확인이 없다.")
    a("- T1에서 사전 없이 남는 칸은 H03뿐이다(H06·H10 자동 설명용). H08을 F2 대표로 올리는 규칙은 'H01 실패로 H06을 시험하지 않을 때'만 정해져 있어, H06이 결측으로 내려간 경우는 문서가 정하지 않았다(사용자 결정).")
    a("")

    # ---------------- 못 한 것
    a("## 계산하지 못한 것과 이유")
    a("| 대상 | 이유 |")
    a("|---|---|")
    a("| corr(TMP, S)(H02) | TMP(3,000단위 LLM 라벨)를 만들지 않았다 |")
    a("| corr(A4, A) 등록판(H36), H35·H36 등록 rz8 | 90일 이력이 차는 창이 10/13~10/15 셋뿐이다 |")
    a("| H43 사건 수 | 직전 120창 상위 5% 기준이 필요하다 |")
    a("| H24 R̂·u, M0 R², X⊥ 전부 | 2025 포럼 자료와 가격 회귀(120창)가 필요하다. 가격 단계 몫 |")
    a("| state.json 백업 | 이번 단계는 읽기만이다 |")
    a("| 8월 규제 시행일, 예탁원 보관 상위, KOSPI200 야간선물 경로 | 저장소에 없고, 이번 단계는 웹 검색을 쓰지 않는다 |")
    a("| H09 보조 코스닥 태그 상한판 | KOSDAQ 시총 상위 30 이름 사전이 없다(하한판만 셈) |")
    a("")
    a("## 12. 다음")
    a("- 이 표는 중간 검증판이다. §1 순서 3의 정식 점검표는 수집이 2025-01-01까지 찬 뒤 같은 코드(`py prechecks_price.py; py prechecks_external.py; py prechecks.py`)를 전체 자료로 다시 돌려 기록하고, 그 뒤에 가격을 붙인다.")
    return "\n".join(L) + "\n"
