"""사전 v0.2 매처: 후보 매처(lexicon_v02.py)에 '뺀 단어 막기'만 더한 판.

뺀 단어는 가장 긴 표현 먼저 규칙에서 자리를 그대로 차지한 뒤 버려진다. 그래서 그 안의 짧은 단어
(예: 뺀 '레버리지' 안의 '레버')가 자리를 다시 잡지 않고, 남은 단어의 적중은 후보판과 똑같다.
뺀 CAP 단어가 규칙으로 E·HES·RES로 옮겨 가던 적중도 함께 사라진다. 질문형·부정 보조 셈(eq·fq·neg_dropped)에도
들어가지 않는다. 새 단어나 새 규칙은 없다.

사용: from lexicon_v02_final import load_v02; analyze = load_v02(); r = analyze(title, body, utype)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v02 as L  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def make_analyzer(removed: dict[str, set[str]], removed_rule_items: dict[str, set[str]]):
    """removed: 층 → 뺀 단어의 공백 지운 소문자 키. removed_rule_items: 층 → 뺀 특수 규칙 이름."""
    blocked = {(st, k) for st, ks in removed.items() for k in ks}
    orig = L._context_excluded

    def _ctx_v02(e, *args, **kwargs):
        if (e.stratum, e.key) in blocked:
            return True                      # 자리는 이미 차지했고, 세지 않는다
        return orig(e, *args, **kwargs)

    def analyze_v02(title, body, utype):
        L._context_excluded = _ctx_v02
        try:
            r = L.analyze(title, body, utype)
        finally:
            L._context_excluded = orig
        if removed_rule_items:
            r.hits = [h for h in r.hits if h.word not in removed_rule_items.get(h.stratum, set())]
        return r

    return analyze_v02


def load_v02(path: Path = RESULTS / "lexicon_v02_full.json"):
    full = json.loads(path.read_text(encoding="utf-8"))
    if full["base"]["sha256"] != L.lexicon_sha256():
        raise RuntimeError("lexicon_v02.py의 후보 사전 지문이 v0.2의 기반(base.sha256)과 다르다")
    rule_names = {w for ws in full["rule_items_candidate"].values() for w in ws}
    removed: dict[str, set[str]] = {}
    removed_rules: dict[str, set[str]] = {}
    for st, words in full["removed"].items():
        for w in words:
            if w in rule_names:
                removed_rules.setdefault(st, set()).add(w)
            else:
                removed.setdefault(st, set()).add(L._compact(w))
    return make_analyzer(removed, removed_rules)
