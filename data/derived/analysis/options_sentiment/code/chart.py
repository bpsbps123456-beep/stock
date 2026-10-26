"""results/chart.html — QQQ와 옵션 심리 지표를 같은 날짜축의 작은 차트 4개로 그린다(외부 라이브러리 없음, SVG).

읽는 것: results/daily.parquet, results/eval.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
START = "2025-07-01"
W, PH, GAP, L, R, TOP = 960, 150, 34, 64, 16, 40

PANELS = [  # (열, 제목, 소수 자리)
    ("QQQ", "QQQ 종가 — 주황 띠: 콜 과열 종합이 지난 250일 상위 20%인 날", 0),
    ("CALLHOT_w", "콜 과열 종합 백분위(1 = 지난 250일 중 가장 콜 쏠림) — 가는 선: 0.8", 1),
    ("QQQ_LSKEW", "QQQ 1년 만기 풋(0.9) − 콜(1.1) 내재변동성 차, %p — 낮을수록 먼 풋이 콜보다 상대적으로 쌈", 1),
    ("HEDGE1Y", "QQQ 1년 만기 10% 외가격 풋 값, 현물 대비 % — 낮을수록 보험이 쌈", 2),
]


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """깔끔한 눈금(1·2·2.5·5 × 10^k 간격)."""
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step) + 1)]


def main() -> None:
    df = pd.read_parquet(RES / "daily.parquet")
    df = df[df.index >= START]
    ev = json.loads((RES / "eval.json").read_text(encoding="utf-8"))
    tops = [(t["peak"], f"QQQ 고점 {t['peak'][5:].replace('-', '/')}") for t in ev["tops"]["QQQ"] if t["peak"] >= START]
    dates = list(df.index)
    n = len(dates)
    x = lambda i: L + (W - L - R) * i / (n - 1)  # noqa: E731
    sig = (df["CALLHOT_w"] >= 0.8).to_numpy()
    height = TOP + len(PANELS) * (PH + GAP)
    svg = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="옵션 심리 지표와 QQQ">']
    for p, (col, title, nd) in enumerate(PANELS):
        y0 = TOP + p * (PH + GAP)
        v = df[col]
        if col == "CALLHOT_w":
            lo, hi = 0.0, 1.0
        else:
            pad = (v.max() - v.min()) * 0.08 or 1
            lo, hi = v.min() - pad, v.max() + pad
        y = lambda val: y0 + PH - (val - lo) / (hi - lo) * PH  # noqa: E731
        svg.append(f'<text class="t" x="{L}" y="{y0 - 10}">{title}</text>')
        for gv in ([0, 0.2, 0.4, 0.6, 0.8, 1.0] if col == "CALLHOT_w" else nice_ticks(lo, hi)):
            svg.append(f'<line class="g" x1="{L}" x2="{W - R}" y1="{y(gv):.1f}" y2="{y(gv):.1f}"/>'
                       f'<text class="a" x="{L - 6}" y="{y(gv) + 4:.1f}" text-anchor="end">{gv:,.{nd}f}</text>')
        if p == 0:  # 신호 띠
            i = 0
            while i < n:
                if sig[i]:
                    j = i
                    while j + 1 < n and sig[j + 1]:
                        j += 1
                    svg.append(f'<rect class="s" x="{x(i) - 1.5:.1f}" y="{y0}" width="{x(j) - x(i) + 3:.1f}" height="{PH}"/>')
                    i = j + 1
                else:
                    i += 1
        if col == "CALLHOT_w":
            svg.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{y(0.8):.1f}" y2="{y(0.8):.1f}"/>')
        for d, lab in tops:
            if d in df.index:
                xi = x(dates.index(d))
                svg.append(f'<line class="tp" x1="{xi:.1f}" x2="{xi:.1f}" y1="{y0}" y2="{y0 + PH}"/>')
                if p == 0:
                    svg.append(f'<text class="a" x="{xi + 3:.1f}" y="{y0 + 11}">{lab}</text>')
        segs, seg = [], []
        for i, val in enumerate(v):
            if pd.isna(val):
                if seg:
                    segs.append(seg)
                seg = []
            else:
                seg.append(f"{x(i):.1f},{y(val):.1f}")
        if seg:
            segs.append(seg)
        for s in segs:
            svg.append(f'<polyline class="l" points="{" ".join(s)}"/>')
    for k, d in enumerate(dates):
        if (k == 0 or dates[k - 1][5:7] != d[5:7]) and d[5:7] in ("01", "04", "07", "10"):
            svg.append(f'<text class="a" x="{x(k):.1f}" y="{height - GAP + 16}" text-anchor="middle">{d[:7]}</text>')
    svg.append(f'<line id="cross" class="cx" x1="0" x2="0" y1="{TOP}" y2="{height - GAP}"/></svg>')
    data = {"d": dates, **{c: [None if pd.isna(z) else round(float(z), 3) for z in df[c]] for c, _, _ in PANELS}}
    names = {c: t.split(" — ")[0][:24] for c, t, _ in PANELS}
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>옵션 심리와 QQQ</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;--grid:#e1e0d9;--line:#2a78d6;--sig:rgba(235,104,52,.16);--tp:#c3c2b7}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--line:#3987e5;--sig:rgba(217,89,38,.15);--tp:#383835}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif}}
.wrap{{max-width:{W}px;margin:auto;position:relative}} svg{{width:100%;height:auto}}
.t{{fill:var(--ink);font-size:13px;font-weight:600}} .a{{fill:var(--mut);font-size:11px;font-variant-numeric:tabular-nums}}
.g{{stroke:var(--grid);stroke-width:1}} .l{{fill:none;stroke:var(--line);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.s{{fill:var(--sig)}} .ref{{stroke:var(--mut);stroke-width:1}} .tp{{stroke:var(--tp);stroke-width:1}} .cx{{stroke:var(--mut);stroke-width:1;visibility:hidden}}
#tip{{position:absolute;pointer-events:none;background:var(--bg);border:1px solid var(--grid);border-radius:6px;padding:6px 8px;font-size:12px;color:var(--ink2);display:none;white-space:nowrap}}
#tip div:first-child{{color:var(--ink);font-weight:600}}
p{{color:var(--ink2);max-width:{W}px;margin:4px auto}}</style></head><body><div class="wrap">
<p>세로 가는 선: QQQ 종가 기준 7% 이상 하락 직전 고점. 지표는 그날 장 마감 뒤 값이다. 아래 두 줄은 5일 중앙값이고 2025-12부터 있다.</p>
{''.join(svg)}<div id="tip"></div></div><script>
const D={json.dumps(data, ensure_ascii=False)};const names={json.dumps(names, ensure_ascii=False)};
const svg=document.querySelector('svg'),tip=document.getElementById('tip'),cx=document.getElementById('cross');
const n=D.d.length,L={L},R={R},W={W};
function row(text){{const el=document.createElement('div');el.textContent=text;return el}}
svg.addEventListener('mousemove',e=>{{const b=svg.getBoundingClientRect(),sx=(e.clientX-b.left)*W/b.width;
const i=Math.round((sx-L)/(W-L-R)*(n-1));if(i<0||i>=n){{tip.style.display='none';cx.style.visibility='hidden';return}}
const xx=L+(W-L-R)*i/(n-1);cx.setAttribute('x1',xx);cx.setAttribute('x2',xx);cx.style.visibility='visible';
tip.replaceChildren(row(D.d[i]),...Object.keys(names).map(k=>row(names[k]+': '+(D[k][i]==null?'—':D[k][i]))));
tip.style.display='block';tip.style.left=Math.min((e.clientX-b.left)+14,b.width-tip.offsetWidth-4)+'px';tip.style.top=(e.clientY-b.top+10)+'px'}});
svg.addEventListener('mouseleave',()=>{{tip.style.display='none';cx.style.visibility='hidden'}});
</script></body></html>"""
    (RES / "chart.html").write_text(html, encoding="utf-8")
    print("저장", RES / "chart.html", n, "일")


if __name__ == "__main__":
    main()
