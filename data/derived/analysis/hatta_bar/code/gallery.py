"""원안 신호를 무작위로 뽑아 1분봉 그림(SVG)으로 모은다 — '내가 말한 봉이 맞는지' 눈으로 확인하는 용도.
사용: py gallery.py [장수=12]   → results/gallery.html"""
import os, sys, html
import numpy as np
import pandas as pd
import engine as E
from build import SYMS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
W, H, PAD = 360, 210, 6
BEFORE, AFTER = 45, 32


def panel(sym, row, pat, src="alp1m"):
    d = E.load(sym, src)
    a = E.prep(d)
    t = int(row.t)
    lo, hi = max(0, t - BEFORE), min(len(d) - 1, t + AFTER)
    seg = slice(lo, hi + 1)
    o, h, l, c = a["o"][seg], a["h"][seg], a["l"][seg], a["c"][seg]
    e10, e25, e60 = a["e10"][seg], a["e25"][seg], a["e60"][seg]
    entry = a["c"][t + 1]
    ymin = min(l.min(), row.S * 0.999)
    ymax = max(h.max(), entry * 1.012, np.nanmax(e60))
    n = hi - lo + 1
    bw = (W - 2 * PAD) / n
    Y = lambda p: PAD + (ymax - p) / (ymax - ymin) * (H - 2 * PAD)
    X = lambda i: PAD + (i - lo + 0.5) * bw
    parts = []
    for i in range(lo, hi + 1):
        j = i - lo
        up = c[j] >= o[j]
        col = "#26a69a" if up else "#ef5350"
        parts.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{Y(h[j]):.1f}" y2="{Y(l[j]):.1f}" stroke="{col}" stroke-width="1"/>')
        y1, y2 = sorted((Y(o[j]), Y(c[j])))
        parts.append(f'<rect x="{X(i) - bw * 0.35:.1f}" y="{y1:.1f}" width="{bw * 0.7:.1f}" height="{max(y2 - y1, 0.8):.1f}" fill="{col}"/>')
    for ser, col in ((e10, "#e53935"), (e25, "#00acc1"), (e60, "#7e57c2")):
        pts = " ".join(f"{X(i):.1f},{Y(ser[i - lo]):.1f}" for i in range(lo, hi + 1))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1" stroke-dasharray="2,2"/>')
    k = t if pat == "A" else t - 1
    parts.append(f'<line x1="{X(max(lo, k - 40)):.1f}" x2="{X(k):.1f}" y1="{Y(row.S):.1f}" y2="{Y(row.S):.1f}" stroke="#222" stroke-width="1.4"/>')
    tg = entry * 1.01
    parts.append(f'<line x1="{X(t + 1):.1f}" x2="{X(min(hi, t + 31)):.1f}" y1="{Y(tg):.1f}" y2="{Y(tg):.1f}" stroke="#2e7d32" stroke-width="1" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{X(t):.1f}" y="{H - 1}" font-size="11" text-anchor="middle" fill="#c2185b">{"①" if pat == "A" else "②"}</text>')
    ts = pd.Timestamp(int(row.ts) * 60, unit="s", tz="UTC").tz_convert(E.ET)
    res = "+1% 도달" if row.hit_e2 else f"못 닿음 {row.r1_e2 * 100:+.2f}%"
    title = f"{sym} {ts:%m/%d %H:%M} 뉴욕 · {res}"
    return f'<figure><svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">{"".join(parts)}</svg><figcaption>{html.escape(title)}</figcaption></figure>'


def main(k=12):
    rng = np.random.default_rng(7)
    blocks = []
    for name, pat in (("orig_A", "A"), ("orig_B", "B")):
        s = pd.read_csv(os.path.join(RES, f"signals_{name}.csv"))
        pick = s.iloc[np.sort(rng.choice(len(s), min(k, len(s)), replace=False))]
        figs = "".join(panel(SYMS[int(r.sid)], r, pat) for r in pick.itertuples())
        label = "① 꼬리 반전 (원안)" if pat == "A" else "② 장악 양봉 (원안)"
        blocks.append(f"<h2>{label} — 무작위 {len(pick)}개 / 전체 {len(s)}개</h2><div class='g'>{figs}</div>")
    page = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>하따봉 신호 예시</title>
<style>:root{{--bg:#fff;--fg:#222}}@media (prefers-color-scheme: dark){{:root{{--bg:#fff;--fg:#222}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,'Malgun Gothic',sans-serif;margin:16px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:10px}}
figure{{margin:0;border:1px solid #ddd;border-radius:6px;padding:4px}}figcaption{{font-size:12px;color:#555}}
svg{{max-width:100%;height:auto}}</style>
<p>검은 실선 = 깨진 지지선, 초록 점선 = 진입가 +1%(30분), 점선 = EMA 10(빨강)·25(하늘)·60(보라). 진입 = 신호 다음 봉 종가(E2).</p>
{''.join(blocks)}</html>"""
    open(os.path.join(RES, "gallery.html"), "w", encoding="utf-8").write(page)
    print("ok")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
