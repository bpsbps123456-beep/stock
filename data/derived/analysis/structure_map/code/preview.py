"""structure_ref 규칙으로 BTCUSDT 차트 미리 보기(SVG HTML)를 만든다. 작가 차트와 눈으로 대조하는 용도."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from structure_ref import atr_series, kst, load, run_liquidity, run_structure, run_trendlines, sma  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "results"


def htf_map(b, iv, n=20):
    """상위 봉 20SMA를 차트 봉 시각에 맞춘다(그 시각까지 완성된 상위 봉 값)."""
    h = load(iv)
    ms = {"12h": 12 * 3600_000, "1d": 24 * 3600_000}[iv]
    s = sma([r["c"] for r in h], n)
    out, j = [], 0
    for r in b:
        while j + 1 < len(h) and h[j + 1]["t"] + ms <= r["t"] + 1:
            j += 1
        out.append(s[j] if h[j]["t"] + ms <= r["t"] + 1 else (s[j - 1] if j else None))
    return out


def main(iv="2h", since="2026-10-03 00:00", w=1500, hgt=760):
    b = load(iv)
    atr = atr_series(b)
    i0 = next(i for i, r in enumerate(b) if kst(r["t"], True) >= since)
    ev = run_trendlines(b, atr, i0)
    liq_ev, liq_left = run_liquidity(b, atr, i0)
    h12 = load("12h")
    run_structure(h12, 0)
    flip = h12[-1]["flip"]
    s12, sd = htf_map(b, "12h"), htf_map(b, "1d")
    seg = b[i0:]
    lo = min(r["l"] for r in seg) - 600
    hi = max(r["h"] for r in seg) + 600
    padl, padr, padt, padb = 20, 110, 40, 40
    cw = (w - padl - padr) / len(seg)
    X = lambda i: padl + (i - i0 + 0.5) * cw
    Y = lambda p: padt + (hi - p) / (hi - lo) * (hgt - padt - padb)
    el = []
    # 가격 눈금
    step = 1000
    p = (int(lo) // step + 1) * step
    while p < hi:
        el.append(f'<line x1="{padl}" x2="{w - padr}" y1="{Y(p):.1f}" y2="{Y(p):.1f}" stroke="#e5e5e5"/>')
        el.append(f'<text x="{w - padr + 6}" y="{Y(p) + 4:.1f}" font-size="11" fill="#888">{p:,}</text>')
        p += step
    # 날짜
    for i in range(i0, len(b)):
        if kst(b[i]["t"]).endswith("01:00") or kst(b[i]["t"]).endswith("00:00") and iv != "2h":
            el.append(f'<text x="{X(i):.1f}" y="{hgt - 14}" font-size="11" fill="#888" text-anchor="middle">{kst(b[i]["t"])[3:5]}</text>')
    # 이동평균
    for ser, col, name in ((s12, "#E040FB", "12H 20"), (sd, "#42A5F5", "D 20")):
        pts = " ".join(f"{X(i):.1f},{Y(ser[i]):.1f}" for i in range(i0, len(b)) if ser[i])
        el.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        el.append(f'<text x="{w - padr + 6}" y="{Y(ser[-1]) + 4:.1f}" font-size="11" fill="{col}">{name}</text>')
    # 유동성(남은 것 중 가까운 위·아래 3개)
    last = b[-1]["c"]
    ups = sorted([L for L in liq_left if L["p"] >= last], key=lambda L: L["p"])[:3]
    dns = sorted([L for L in liq_left if L["p"] < last], key=lambda L: -L["p"])[:3]
    for L in ups + dns:
        if lo < L["p"] < hi:
            col = "#F23645" if L["p"] >= last else "#089981"
            x1 = X(max(L["i"], i0))
            el.append(f'<line x1="{x1:.1f}" x2="{w - padr}" y1="{Y(L["p"]):.1f}" y2="{Y(L["p"]):.1f}" stroke="{col}" stroke-dasharray="2,3" opacity="0.7"/>')
            el.append(f'<text x="{w - padr + 6}" y="{Y(L["p"]) + 4:.1f}" font-size="10" fill="{col}">유동성 {L["p"]:,.0f}</text>')
    # 추세 전환 유동성(12H)
    if flip:
        el.append(f'<line x1="{padl}" x2="{w - padr}" y1="{Y(flip[0]):.1f}" y2="{Y(flip[0]):.1f}" stroke="#F23645" stroke-width="2"/>')
        el.append(f'<text x="{padl + 6}" y="{Y(flip[0]) - 6:.1f}" font-size="12" fill="#F23645">추세 전환 유동성 {flip[0]:,.0f} (12H)</text>')
    # 캔들
    for i in range(i0, len(b)):
        r = b[i]
        up = r["c"] >= r["o"]
        col = "#26a69a" if up else "#ef5350"
        el.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{Y(r["h"]):.1f}" y2="{Y(r["l"]):.1f}" stroke="{col}"/>')
        y1, y2 = Y(max(r["o"], r["c"])), Y(min(r["o"], r["c"]))
        el.append(f'<rect x="{X(i) - cw * 0.35:.1f}" y="{y1:.1f}" width="{cw * 0.7:.1f}" height="{max(1, y2 - y1):.1f}" fill="{col}"/>')
    # 추세선: 지금 그려져 있을 선(하락선 = 작가 선, 마지막 사건 기준)
    seen = {}
    for i, who, what, t, exi in ev:
        if what in ("새 추세선", "다시 긋기", "오래된 선 교체"):
            seen[t.d] = t
    tl = [t for _, who, what, t, _ in ev if what.startswith("돌파 확정") and who == "하락선"]
    for t in tl[-1:] + [seen.get(1)]:
        if t is None:
            continue
        col = "#F23645" if t.d < 0 else "#089981"
        xa, xb = X(max(t.ia, i0)), X(len(b) - 1)
        dash = ' stroke-dasharray="6,4"' if t.broken else ""
        el.append(f'<line x1="{xa:.1f}" y1="{Y(t.val(max(t.ia, i0))):.1f}" x2="{xb:.1f}" y2="{Y(t.val(len(b) - 1)):.1f}" stroke="{col}" stroke-width="2"{dash}/>')
    # 사건 라벨(하락선만, 작가 차트와 대조)
    for i, who, what, t, exi in ev:
        if who != "하락선" or what in ("새 추세선", "다시 긋기", "오래된 선 교체", "확립", "확립 전 이탈 → 버림"):
            continue
        if i < i0 + 40:
            continue
        j, py = (exi[0], exi[1]) if exi else (i, b[i]["h"])
        txt = what
        el.append(f'<text x="{X(j):.1f}" y="{Y(py) - 10:.1f}" font-size="12" font-weight="bold" fill="#333" text-anchor="middle">{txt}</text>')
    html = f'<html><body style="margin:0;background:#fff"><svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{hgt}" font-family="Malgun Gothic, sans-serif">' \
           f'<text x="{padl}" y="24" font-size="16" font-weight="bold">BTCUSDT {iv} · 구조물 지도 기준 구현 미리 보기 ({kst(b[i0]["t"])} ~ {kst(b[-1]["t"])} KST)</text>' + "".join(el) + "</svg></body></html>"
    OUT.mkdir(exist_ok=True)
    f = OUT / f"preview_{iv}.html"
    f.write_text(html, encoding="utf-8")
    print(f)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["2h"]))
