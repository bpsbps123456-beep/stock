"""사전 정밀도 점검용 가림 표본 (HYPOTHESES §5.1 #10, lexicon_draft §8·§9, HYPOTHESES §2.7 측정 품질 기준).

- 표본 틀: interim/results/units.parquet(로더 산출, 10/15 스냅샷). 급락일(7/29~7/31, 10/13)과 열어 본 8일(8/20·9/2·9/3·
  9/8·9/16·10/12·10/14·10/15)은 단위 자신의 날짜와 부모 글 날짜 모두에서 뺀다. 날짜를 확정하지 못한 댓글(latest)도 뺀다.
- 묶음(층)마다 사전 v0.2 후보(lexicon_v02.py)가 적중한 단위를 무작위 60개(적으면 전부), 강세·약세 글 점수용 무작위 글 200개.
- 가림: 날짜·시각·글 번호·작성자를 빼고, 본문 안의 날짜 → [날짜], 시각 → [시각], 가격·지수·수익률 숫자 → [숫자],
  사건 이름 → [사건], 주소 → [링크]. 적중 단어는 ⟦k:단어⟧로 표시한다(점수 200개는 사전 판독을 숨긴다).
- 산출: results/blind_samples.json(라벨러용), results/blind_key.parquet(정답 대응표, 라벨러에게 주지 않음),
  results/blind_summary.json(모집단 적중 수·표본 수·지문), results/lexicon_v02_candidate.json(점검한 사전 판).
가격은 읽지 않는다(units.parquet에는 가격 열이 없다).

실행: py blind_sample.py   (이 폴더에서)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v02 as L  # noqa: E402

INTERIM = Path(__file__).resolve().parents[1]
RESULTS = INTERIM / "results"
SEED = 20261016
N_PER_STRATUM = 60
N_SCORE = 200
CUTOFF = date(2026, 10, 15)
CRASH_DAYS = {date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31), date(2026, 10, 13)}
OPENED_DAYS = {date(2026, 8, 20), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 8), date(2026, 9, 16),
               date(2026, 10, 12), date(2026, 10, 14), date(2026, 10, 15)}
EXCLUDED_DAYS = CRASH_DAYS | OPENED_DAYS
BODY_MAX = 1500      # 본문이 길면 적중 주변 ±300자 발췌
EXCERPT_PAD = 300

STRATUM_INFO = {
    "E": "환호(태도어): 자랑·풀매수·떡상 등 들뜬 상승 기대나 이익 자랑",
    "F": "공포: 겁·손실 고통·폭락 공포",
    "PX": "가격 서술: 상한가·신고가·파란불 등 가격 움직임 자체를 말함(심리 셈에서 빼는 용도)",
    "CAP": "항복 완료: 글쓴이 자신이 이미 (손실을 보고) 팔았다·접었다는 1인칭 과거형",
    "HES": "매도 고민: 팔지·손절할지·버틸지 묻거나 고민함",
    "LL_KR": "레버리지 롱(국내): 신용·미수·빚투·국내 레버리지 ETF 등 빚이나 레버리지로 상승에 거는 것",
    "LL_US": "레버리지 롱(미국): SOXL·TQQQ·NVDL 등 미국 레버리지 ETF·콜옵션으로 상승에 거는 것",
    "LS": "레버리지 숏: 인버스·곱버스·풋·숏으로 하락에 거는 것",
    "MC": "반대매매·마진콜: 담보 부족·강제 청산",
    "NOV": "초보 유입: 주식 초보·처음 시작·무엇을 살지 묻는 신규 투자자",
    "FO": "소외(FOMO): 못 탔다·부럽다·놓쳤다·살걸 같은 상승 소외감",
    "RES": "체념·존버: 팔지 않고 버팀·장기 보유·본전만·체념",
    "ANG": "분노·음모: 세력·공매도·외인·기관 탓, 조작·설거지 음모",
    "KX": "국장 이탈: 한국 시장을 떠나 미국 시장으로",
    "KR": "국장 회귀·미장 비관: 한국 시장으로 돌아옴, 미국 시장 비관",
    "BUY": "저가 매수 의지: 줍줍·저점 매수·물타기·담았다",
    "BULL": "강세 의견 단어(글 점수 +1, 부정이면 −1)",
    "BEAR": "약세 의견 단어(글 점수 −1, 부정이면 +1)",
    "MOCK_BEAR": "약세론 조롱: 비관론자·인버스 투자자를 비웃음",
    "MOCK_BULL": "강세론 조롱: 낙관론자를 비웃음(행복회로·상투·흑우 등)",
    "REBUT": "반박: 앞 글·의견에 반대하거나 반박함",
    "REBUT_FEAR": "반박(공포 쪽): 공포·비관에 맞서 '오히려 기회·바닥'이라고 반박함",
    "CHASE_BUY": "추격 매수 고백: 글쓴이 자신이 샀다·탑승했다·물탔다는 1인칭 매수 고백",
    "CHASE_SELL": "매도 고백(대조): 글쓴이 자신이 팔았다·정리했다는 1인칭 매도 고백",
    "CONV_BULL": "강세 전향: 참다가 결국 샀다·탑승했다(회의에서 매수로 돌아섬)",
    "CONV_BEAR": "약세 전향: 버티다 다 던졌다·존버 포기(보유에서 매도로 돌아섬)",
    "DIST": "부실: 상장폐지·거래정지·횡령·법정관리 등 기업 부실 사건",
    "IPO": "공모주: 공모주 청약·상장 첫날 등 주식 공모(부동산 청약 아님)",
    "MEM": "메모리 주제: 메모리 반도체·관련 종목(하닉·삼전·마이크론 등) 이야기",
    "SEMI": "반도체 주제: 반도체 업종·종목 이야기",
    "TILT_US": "미국 시장 쪽 관심: 미장·나스닥·미국 종목 이야기(국내 쪽 단어 없는 글)",
    "TILT_KR": "국내 시장 쪽 관심: 국장·코스피·국내 종목 이야기(미국 쪽 단어 없는 글)",
    "ALIAS_SEC": "종목 별칭: 삼성전자를 가리킴",
    "ALIAS_SKH": "종목 별칭: SK하이닉스를 가리킴",
    "ALIAS_BOTH": "종목 별칭: 삼성전자와 SK하이닉스 둘 다를 가리킴",
    "ALIAS_US": "종목 별칭: 미국 AI 종목(엔비디아·마이크론·샌디스크·브로드컴·TSMC·AMD·WDC·델·팔란티어·테슬라)을 가리킴",
}

# ---------------------------------------------------------------- 가림

URL_RX = re.compile(r"https?://\S+|www\.\S+|\b[\w.-]+\.(?:com|co\.kr|kr|net|org|io)(?:/\S*)?", re.I)
DATE_RXS = [
    re.compile(r"(?<!\d)(?:19|20)\d{2}\s?[.\-/]\s?\d{1,2}(?:\s?[.\-/]\s?\d{1,2})?(?!\d)"),
    re.compile(r"(?<!\d)(?:(?:19|20)\d{2}|'?2\d)\s?년"),
    re.compile(r"(?<!\d)\d{1,2}\s?월\s?\d{1,2}\s?일"),
    re.compile(r"(?<!\d)\d{1,2}\s?월"),
    re.compile(r"(?<!\d)\d{1,2}\s?일(?!간|째|치)"),
    re.compile(r"(?<![\d.])\d{1,2}/\d{1,2}(?![\d/])"),
    re.compile(r"(?<!\d)\d{1,2}\s?분기"),
]
TIME_RXS = [
    re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)"),
    re.compile(r"(?:(?:오전|오후|새벽|아침|저녁|밤|낮)\s?)?(?<!\d)\d{1,2}\s?시(?:\s?\d{1,2}\s?분)?"
               r"(?!간|가|장|세|즌|청|점|총|초|작|황|켜|키|도|험|위|민|대)"),
]
EVENT_RXS = [
    re.compile(r"(?:[가-힣A-Za-z0-9\[\]]+\s?)?(?:쇼크|사태|파동|게이트)"),
    re.compile(r"ADR\s?상장|ADR|에이디알", re.I),
    re.compile(r"(?:단일\s?종목\s?)?레버리지\s?(?:ETF\s?)?(?:규제|제한)", re.I),
    re.compile(r"공매도\s?(?:전면\s?)?(?:재개|금지)"),
    re.compile(r"NXT|넥스트\s?레이드|대체거래소", re.I),
    re.compile(r"(?:지방|대통령|국회의원|중간)\s?선거|(?<![나어])선거|대선|총선|지선"),
    re.compile(r"추석|설날|설\s?연휴|한가위|광복절|한글날|개천절|제헌절|현충일|어린이날|대체\s?공휴일|성탄절|크리스마스|"
               r"블랙\s?프라이데이|추수감사절"),
    re.compile(r"잭슨\s?홀|셧다운"),
    re.compile(r"상법\s?개정"),
    re.compile(r"MSCI\s?(?:선진국|편입|관찰)[가-힣]*", re.I),
    re.compile(r"(?<![가-힣])이란(?:\s?(?:전쟁|사태|공습|제재|혁명수비대|본토))?|이스라엘|혁명수비대|IRGC|휴전|호르무즈"),
    re.compile(r"(?:케빈|캐빈)\s?워시|(?<![가-힣])워시(?![가-힣]{2})"),
]
NUM_RXS = [
    re.compile(r"[이삼사오육칠팔구]천\s?(?:피|포인트|선)"),
    re.compile(r"[+\-−±]?\s?\d[\d,]*(?:\.[\dXx]+)?\s?(?:%|％|퍼센트|퍼|프로)"),
    re.compile(r"\d[\d,]*(?:\.\d+)?\s?(?:포인트|pt|p)(?![a-zA-Z])"),
    re.compile(r"[$＄]\s?\d[\d,]*(?:\.\d+)?"),
    re.compile(r"\d[\d,]*(?:\.\d+)?(?:\s?(?:십|백|천|만|억|조))+\s?(?:원|달러|불)?|\d[\d,]*(?:\.\d+)?\s?(?:원|달러|불|USD)"),
    re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"),
    re.compile(r"(?<!\d)\d+\.\d+(?!\d)"),
    re.compile(r"(?<!\d)\d{3,}(?!\d)"),
]


def mask_text(s: str) -> str:
    """적중 단어가 아닌 구간에 쓰는 가림. 순서: 주소 → 날짜 → 시각 → 사건 → 숫자."""
    s = URL_RX.sub("[링크]", s)
    for rx in DATE_RXS:
        s = rx.sub("[날짜]", s)
    for rx in TIME_RXS:
        s = rx.sub("[시각]", s)
    for rx in EVENT_RXS:
        s = rx.sub("[사건]", s)
    for rx in NUM_RXS:
        s = rx.sub("[숫자]", s)
    return s


def blind_segment(text: str, spans: list[tuple[int, int, int]]) -> str:
    """text 안 spans [(start, end, k)]는 ⟦k:원문⟧로 두고 나머지를 가린다."""
    out, pos = [], 0
    for s, e, k in sorted(spans):
        if s < pos:
            continue
        out.append(mask_text(text[pos:s]))
        out.append(f"⟦{k}:{text[s:e]}⟧")
        pos = e
    out.append(mask_text(text[pos:]))
    return "".join(out)


def excerpt(s: str, limit: int = BODY_MAX, pad: int = EXCERPT_PAD) -> tuple[str, bool]:
    if len(s) <= limit:
        return s, False
    marks = [m.start() for m in re.finditer("⟦", s)]
    if not marks:
        return s[:limit] + " …(뒤 생략)", True
    ivs = []
    for m in marks:
        e = s.find("⟧", m)
        a, b = max(0, m - pad), min(len(s), (e if e >= 0 else m) + pad)
        if ivs and a <= ivs[-1][1]:
            ivs[-1][1] = max(ivs[-1][1], b)
        else:
            ivs.append([a, b])
    parts = [("… " if a > 0 else "") + s[a:b] + (" …" if b < len(s) else "") for a, b in ivs]
    return " ".join(parts), True


# ---------------------------------------------------------------- 표본

def _seed(name: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:8], 16)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frame(results: Path = RESULTS) -> tuple[pd.DataFrame, dict]:
    cols = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text"]
    u = pd.read_parquet(results / "units.parquet", columns=cols)
    u = u.reset_index().rename(columns={"index": "unit_idx"})
    d = u["dt_kst"].dt.date
    pdd = u["post_dt_kst"].dt.date
    stats = {"units_total": int(len(u))}
    keep = (d <= CUTOFF) & (pdd <= CUTOFF)
    stats["drop_after_cutoff"] = int((~keep).sum())
    lat = u["dt_conf"].eq("latest")
    stats["drop_latest_date"] = int((keep & lat).sum())
    keep &= ~lat
    exd = d.isin(EXCLUDED_DAYS) | pdd.isin(EXCLUDED_DAYS)
    stats["drop_excluded_days"] = int((keep & exd).sum())
    keep &= ~exd
    titles = u.loc[u["utype"].eq("post"), ["no", "title"]].set_index("no")["title"]
    e = u.loc[keep].copy()
    e["parent_title"] = e["no"].map(titles)
    stats["eligible_units"] = int(len(e))
    stats["eligible_posts"] = int(e["utype"].eq("post").sum())
    stats["eligible_comments"] = int(e["utype"].eq("comment").sum())
    days = sorted(set(e["dt_kst"].dt.date))
    stats["eligible_days"] = len(days)
    stats["eligible_first"], stats["eligible_last"] = str(days[0]), str(days[-1])
    return e, stats


def run_matcher(e: pd.DataFrame) -> tuple[list[L.UnitResult], dict]:
    results = []
    excl = {"news": 0, "deleted": 0, "politics": 0}
    for t, b, ut in zip(e["title"].tolist(), e["text"].tolist(), e["utype"].tolist()):
        r = L.analyze(t if ut == "post" else None, b, ut)
        if r.excluded:
            excl[r.excluded] += 1
        results.append(r)
    return results, excl


def _scope_ok(stratum: str, utype: str) -> bool:
    sc = L.STRATUM_SCOPE[stratum]
    return sc == "all" or sc == utype


def build_samples(e: pd.DataFrame, res: list[L.UnitResult]):
    utypes = e["utype"].tolist()
    by_stratum: dict[str, list[int]] = {s: [] for s in L.STRATA}
    word_pop: dict[str, dict[str, int]] = {s: {} for s in L.STRATA}
    for pos, (r, ut) in enumerate(zip(res, utypes)):
        if r.excluded or not r.hits:
            continue
        seen = set()
        for h in r.hits:
            if not _scope_ok(h.stratum, ut):
                continue
            if h.stratum not in seen:
                by_stratum[h.stratum].append(pos)
                seen.add(h.stratum)
            word_pop[h.stratum][h.word] = word_pop[h.stratum].get(h.word, 0) + 1
    picks: dict[str, list[int]] = {}
    for s in L.STRATA:
        cand = by_stratum[s]
        rng = np.random.default_rng(_seed(s))
        n = min(N_PER_STRATUM, len(cand))
        picks[s] = [cand[i] for i in rng.choice(len(cand), size=n, replace=False)] if n else []
    score_pool = [p for p, (r, ut) in enumerate(zip(res, utypes)) if ut == "post" and not r.excluded]
    rng = np.random.default_rng(_seed("SCORE200"))
    score_pick = [score_pool[i] for i in rng.choice(len(score_pool), size=min(N_SCORE, len(score_pool)), replace=False)]
    pop = {s: len(by_stratum[s]) for s in L.STRATA}
    return picks, score_pick, pop, word_pop, len(score_pool)


def _unit_texts(row) -> tuple[str, str, int]:
    if row["utype"] == "post":
        title = row["title"] or ""
        body = row["text"] or ""
        return title + "\n" + body, title, len(title)
    return row["text"] or "", "", 0


def make_outputs(e: pd.DataFrame, res: list[L.UnitResult], picks, score_pick):
    id_rng = np.random.default_rng(_seed("IDS"))
    used: set[str] = set()

    def new_id(prefix: str) -> str:
        while True:
            x = f"{prefix}{int(id_rng.integers(0, 2**32)):08x}"
            if x not in used:
                used.add(x)
                return x

    samples, score_samples, key_rows = [], [], []
    for s in L.STRATA:
        for pos in picks[s]:
            row = e.iloc[pos]
            r = res[pos]
            text, title, tlen = _unit_texts(row)
            hs = sorted([h for h in r.hits if h.stratum == s], key=lambda h: h.start)
            spans_t, spans_b, hit_out, hit_key = [], [], [], []
            for k, h in enumerate(hs, start=1):
                where = "제목" if h.in_title else ("본문" if row["utype"] == "post" else "댓글")
                if h.in_title:
                    spans_t.append((h.start, h.end, k))
                else:
                    spans_b.append((h.start - (tlen + 1 if row["utype"] == "post" else 0),
                                    h.end - (tlen + 1 if row["utype"] == "post" else 0), k))
                d = {"k": k, "word": h.word, "where": where}
                if h.reading:
                    d["dict_reading"] = h.reading
                if h.origin:
                    d["note"] = h.origin
                hit_out.append(d)
                hit_key.append({**d, "start": h.start, "end": h.end, "surface": text[h.start:h.end], "sign": h.sign})
            body_src = text[tlen + 1:] if row["utype"] == "post" else text
            body_bl, trunc = excerpt(blind_segment(body_src, spans_b))
            sid = new_id("P")
            item = {"id": sid, "set": "precision", "bundle": s, "unit": "글" if row["utype"] == "post" else "댓글",
                    "hits": hit_out}
            if row["utype"] == "post":
                item["title"] = blind_segment(title, spans_t)
                item["body"] = body_bl
            else:
                item["text"] = body_bl
                pt = row["parent_title"]
                item["parent_title"] = mask_text(pt) if isinstance(pt, str) else ""
            if trunc:
                item["excerpted"] = True
            samples.append(item)
            key_rows.append(_key_row(sid, "precision", s, row, r, hit_key, trunc))
    for pos in score_pick:
        row = e.iloc[pos]
        r = res[pos]
        text, title, tlen = _unit_texts(row)
        body_bl, trunc = excerpt(blind_segment(text[tlen + 1:], []))
        sid = new_id("S")
        item = {"id": sid, "set": "score200", "unit": "글", "title": blind_segment(title, []), "body": body_bl}
        if trunc:
            item["excerpted"] = True
        score_samples.append(item)
        hk = [{"word": h.word, "stratum": h.stratum, "reading": h.reading, "sign": h.sign, "start": h.start,
               "end": h.end, "surface": text[h.start:h.end], "in_title": h.in_title}
              for h in r.hits if h.stratum in ("BULL", "BEAR")]
        key_rows.append(_key_row(sid, "score200", "BULLBEAR", row, r, hk, trunc))
    return samples, score_samples, pd.DataFrame(key_rows)


def _key_row(sid, set_, stratum, row, r, hits, trunc) -> dict:
    return {"id": sid, "set": set_, "stratum": stratum,
            "bundle": L.STRATUM_BUNDLE.get(stratum, stratum), "unit_idx": int(row["unit_idx"]),
            "utype": row["utype"], "no": int(row["no"]), "cid": int(row["cid"]), "a": row["a"],
            "dt_kst": row["dt_kst"], "post_dt_kst": row["post_dt_kst"], "dt_conf": row["dt_conf"],
            "src_line": row["src_line"] if isinstance(row["src_line"], str) else None,
            "n_hits": len(hits), "hits_json": json.dumps(hits, ensure_ascii=False),
            "dict_score": r.score if row["utype"] == "post" else None, "m11_irony_flag": bool(r.m11),
            "excerpted": bool(trunc)}


LEAK_CHECKS = {
    "3자리 이상 숫자": re.compile(r"(?<!⟦\d)(?<![\d.,])\d{3,}"),
    "% 기호": re.compile(r"%"),
    "N월": re.compile(r"\d\s?월"),
    "시각 hh:mm": re.compile(r"\d:\d\d"),
    "명절·선거": re.compile(r"추석|설날|선거|한글날|광복절|개천절"),
    "ADR": re.compile(r"ADR", re.I),
}


def leak_scan(items: list[dict]) -> dict:
    """가림 뒤 남은 위험 패턴 수(적중 표시 ⟦…⟧ 안은 뺀다)."""
    out = {k: 0 for k in LEAK_CHECKS}
    for it in items:
        s = " ".join(str(it.get(f, "")) for f in ("title", "body", "text", "parent_title"))
        s = re.sub(r"⟦[^⟧]*⟧", " ", s)
        for k, rx in LEAK_CHECKS.items():
            if rx.search(s):
                out[k] += 1
    return out


def main() -> int:
    t0 = datetime.now()
    e, stats = load_frame()
    res, excl = run_matcher(e)
    picks, score_pick, pop, word_pop, n_score_pool = build_samples(e, res)
    samples, score_samples, key = make_outputs(e, res, picks, score_pick)

    lex_path = RESULTS / "lexicon_v02_candidate.json"
    lex_path.write_bytes(L.lexicon_json().encode("utf-8"))   # LF 그대로(지문 = 파일 SHA-256)
    meta = {
        "made": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "purpose": "사전 v0.2 후보의 정밀도 점검(HYPOTHESES §2.7: 묶음 정밀도 60% 이상, 적중 10회 이상 단어 50% 이상, "
                   "반어율 30% 이하). score200은 강세·약세 글 점수 정확도(H06: 60% 미만이면 측정 실패) 점검용",
        "lexicon": "v0.2 후보(정밀도 점검 전). results/lexicon_v02_candidate.json",
        "lexicon_sha256": L.lexicon_sha256(),
        "seed": SEED,
        "frame": "급락일 4일·열어 본 8일을 뺀 2026-07-13~10-11 단위(날짜 미확정 댓글 제외)",
        "marker": "⟦k:단어⟧ = 이 묶음 사전이 적중한 k번째 단어(hits의 k와 같다)",
        "blinding": "날짜·시각·글 번호·작성자 없음. 본문 안 날짜 [날짜], 시각 [시각], 가격·지수·수익률·금액 숫자 [숫자], "
                    "사건 이름 [사건], 주소 [링크]. 1~2자리 정수(2배·3번 등)는 남겼다",
        "fields": {"hits[].dict_reading": "BULL·BEAR에서 사전이 읽은 방향(부정이면 뒤집음)",
                   "hits[].note": "다른 묶음에서 규칙으로 옮겨 온 적중(예: 항복 완료 표현이 부정되어 체념으로)",
                   "parent_title": "댓글이 달린 글의 제목(가림, 맥락용)",
                   "excerpted": "본문이 길어 적중 주변만 보였거나(정밀도) 앞부분만 보였다(score200)"},
        "score200_note": "score200은 사전 판독을 숨겼다(라벨러가 독립으로 강세/약세/중립을 매기게). 사전 점수는 blind_key에만 있다",
    }
    out = {"meta": meta, "bundles": {s: {"meaning": STRATUM_INFO[s], "unit_scope":
                                             {"all": "글·댓글", "post": "글", "comment": "댓글"}[L.STRATUM_SCOPE[s]],
                                         "n": len(picks[s])} for s in L.STRATA},
           "samples": samples, "score_samples": score_samples}
    js_path = RESULTS / "blind_samples.json"
    js_path.write_bytes(json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8"))
    key_path = RESULTS / "blind_key.parquet"
    key.to_parquet(key_path, index=False)

    summary = {
        "frame": {**stats, "excluded_days": sorted(str(d) for d in EXCLUDED_DAYS),
                  "noise_filtered": excl, "score_pool_posts": n_score_pool},
        "strata": {s: {"population_units_hit": pop[s], "sampled": len(picks[s]), "scope": L.STRATUM_SCOPE[s]}
                   for s in L.STRATA},
        "score200": len(score_samples),
        "score200_dict_score_counts": {str(k): int(v) for k, v in
                                       key.loc[key["set"].eq("score200"), "dict_score"].value_counts().sort_index().items()},
        "word_population_hits": {s: dict(sorted(word_pop[s].items(), key=lambda kv: -kv[1])) for s in L.STRATA},
        "leak_scan_after_blinding": {"precision": leak_scan(samples), "score200": leak_scan(score_samples)},
        "seed": SEED, "lexicon_sha256": L.lexicon_sha256(),
        "files_sha256": {p.name: _sha(p) for p in (js_path, key_path, lex_path)},
        "code_sha256": {p.name: _sha(p) for p in (Path(__file__), Path(L.__file__))},
        "inputs_sha256": {"units.parquet": _sha(RESULTS / "units.parquet")},
        "runtime_sec": round((datetime.now() - t0).total_seconds(), 1),
    }
    (RESULTS / "blind_summary.json").write_bytes(json.dumps(summary, ensure_ascii=False, indent=1).encode("utf-8"))
    print(json.dumps({k: summary[k] for k in ("frame", "strata", "score200", "leak_scan_after_blinding",
                                              "runtime_sec")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
