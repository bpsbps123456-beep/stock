"""Build the AI 산업 지도 explainer into aimap/dist (static, deployed to Vercel).

Reads desk/tour_map.json (stops, zones, companies, the two axes' deals) and the Blender cutaways
(desk/static/scenes/*.webp + scenes.json), joins the plain-language text in aimap/guide.json, and
bakes one "spotlight" picture per zone and panel: the zone's own pixels (its hover mask from the
Blender export) stay sharp, everything else fades toward the paper colour. Fund weights, the
7-stock picks and the trade journal are never read, so nothing personal can reach the public site.
Rules: docs/desk_system.md §8.3.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from desk import scenes as desk_scenes
from desk import tour

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"
GUIDE = HERE / "guide.json"
DIST = HERE / "dist"
PROJECT = "ai-industry-map"                     # Vercel project name
SITE = "https://ai-industry-map-five.vercel.app"  # its production domain (Vercel's pick: the bare name was taken);
                                                # link previews need an absolute image URL

PLATE = (239, 233, 222)    # picture plate colour; style.css --plate is the same
FADE = 0.64                # how far everything outside the lit zone fades toward the plate
GLOW = (236, 160, 22)      # halo around lit parts, so white machines still stand out
SMALL = 60                 # zones drawn smaller than this (CSS px) also get a ring around them
ZONE_CROP = {"aspect": 3 / 2, "min_w": 300, "pad": 0.16, "out_w": 960}
THUMB_CROP = {"aspect": 4 / 3, "min_w": 360, "pad": 0.06, "out_w": 360}
HERO_W = 1440
RECT = re.compile(r"M([\d.]+) ([\d.]+)h([\d.]+)v([\d.]+)h-[\d.]+z")
HIDE_MARKETS = {"fund"}    # an ETF sits in the map for the desk's holdings lens; it is not a company
STOP_KEYS = ("id", "no", "leg", "name", "site", "makes", "gets", "gives", "from", "to", "need")
ZONE_KEYS = ("id", "name", "need", "race", "bottleneck", "src", "lead", "lead_why", "lead_src", "special")
MEMBER_KEYS = ("t", "role", "pos", "src", "main")
COMPANY_KEYS = ("name", "ko", "market", "short", "aliases")
PRIVATE_KEYS = {"funds", "weights", "holdings", "journal", "account", "positions", "targets", "rankings"}


def load_guide(path: Path = GUIDE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(m: dict, sc: dict, guide: dict) -> list[str]:
    """Everything that would leave the site half-explained or pointing at nothing."""
    out = [f"지도: {p}" for p in tour.check_map(m)]
    out += [f"단면도: {p}" for p in sc.get("problems", [])]
    stops = {s["id"]: s for s in m["stops"]}
    zones = {z["id"] for s in m["stops"] for z in s["zones"]}
    legs = {s["leg"] for s in m["stops"]}
    for sid in stops:
        g = guide["stops"].get(sid, {})
        out += [f"guide.stops.{sid}: '{k}' 없음" for k in ("easy", "analogy", "money") if not g.get(k)]
    out += [f"guide.stops.{sid}: 지도에 없는 방문지" for sid in guide["stops"] if sid not in stops]
    out += [f"guide.zones.{zid}: 쉬운 설명 없음" for zid in sorted(zones - set(guide["zones"]))]
    out += [f"guide.zones.{zid}: 지도에 없는 분야" for zid in sorted(set(guide["zones"]) - zones)]
    out += [f"guide.legs.{leg}: 없음" for leg in sorted(legs - set(guide["legs"]))]
    kinds = {ln["kind"] for ln in m["links"]}
    out += [f"guide.kinds.{k}: 없음" for k in sorted(kinds - set(guide["kinds"]))]
    statuses = {ln["status"] for ln in m["links"]}
    out += [f"guide.statuses.{k}: 없음" for k in sorted(statuses - set(guide["statuses"]))]
    seen = {}
    for g in guide["glossary"]:
        if not (g.get("term") and g.get("short") and g.get("match")):
            out.append(f"glossary {g.get('term')}: term·short·match 필요")
        for pat in g.get("match", []):
            if pat in seen:
                out.append(f"glossary: '{pat}'가 {seen[pat]}와 {g['term']}에 겹침")
            seen[pat] = g.get("term")
        out += [f"glossary {g['term']}: 없는 방문지 {sid}" for sid in g.get("stops", []) if sid not in stops]
    return out


def _grow(mask: Image.Image, px: float) -> Image.Image:
    """Dilate a 0/255 mask by about px pixels (blur + threshold: fast at any radius)."""
    return mask.filter(ImageFilter.GaussianBlur(px / 2)).point(lambda v: 255 if v > 20 else 0)


def crop_box(bbox, W: float, H: float, aspect: float, min_w: float, pad: float):
    """A crop of the panel (CSS px) around a zone's bbox: padded, at least min_w wide, at the
    given aspect, kept inside the picture."""
    x0, y0, x1, y1 = bbox
    cw = max((x1 - x0) * (1 + 2 * pad), min_w)
    ch = max((y1 - y0) * (1 + 2 * pad), min_w / aspect)
    if cw / ch < aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    if cw > W:
        cw, ch = W, W / aspect
    if ch > H:
        cw, ch = H * aspect, H
    x = min(max((x0 + x1) / 2 - cw / 2, 0), W - cw)
    y = min(max((y0 + y1) / 2 - ch / 2, 0), H - ch)
    return x, y, cw, ch


class Panel:
    """One Blender panel on its plate, plus a faded copy to spotlight zones against."""

    def __init__(self, pid: str, meta: dict):
        self.id, self.meta = pid, meta
        rgba = Image.open(desk_scenes.STATIC / meta["img"]).convert("RGBA")
        self.scale = rgba.width / meta["w"]
        plate = Image.new("RGBA", rgba.size, PLATE + (255,))
        self.base = Image.alpha_composite(plate, rgba).convert("RGB")
        self.faded = Image.blend(self.base, Image.new("RGB", rgba.size, PLATE), FADE)
        self.spots = {sp["zone"]: sp for sp in meta["spots"]}

    def lit(self, zones, ring: bool = True) -> Image.Image:
        hard = Image.new("L", self.base.size, 0)
        d, s = ImageDraw.Draw(hard), self.scale
        for z in zones:
            for x, y, w, h in RECT.findall(self.spots[z]["path"]):
                x, y, w, h = float(x), float(y), float(w), float(h)
                d.rectangle([x * s, y * s, (x + w) * s - 1, (y + h) * s - 1], fill=255)
        core = _grow(hard, 2 * s)                     # take in the outlines around the mask
        out = Image.composite(self.base, self.faded, core.filter(ImageFilter.GaussianBlur(2.5 * s)))
        halo = ImageChops.subtract(_grow(core, 4 * s), core).filter(ImageFilter.GaussianBlur(1.5 * s))
        out = Image.composite(Image.new("RGB", out.size, GLOW), out, halo.point(lambda v: v * 0.6))
        if ring:
            pen = ImageDraw.Draw(out)
            for z in zones:
                x0, y0, x1, y1 = self.spots[z]["bbox"]
                if max(x1 - x0, y1 - y0) < SMALL:
                    r = (max(x1 - x0, y1 - y0) / 2 + 16) * s
                    cx, cy = (x0 + x1) / 2 * s, (y0 + y1) / 2 * s
                    pen.ellipse([cx - r, cy - r, cx + r, cy + r], outline=GLOW, width=round(2.5 * s))
        return out

    def union_bbox(self, zones):
        boxes = [self.spots[z]["bbox"] for z in zones]
        return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]

    def save_crop(self, img: Image.Image, box, out_w: int, path: Path) -> dict:
        x, y, cw, ch = box
        s = self.scale
        part = img.crop((round(x * s), round(y * s), round((x + cw) * s), round((y + ch) * s)))
        out_h = round(out_w * ch / cw)
        part.resize((out_w, out_h), Image.LANCZOS).save(path, "WEBP", quality=80, method=4)
        return {"w": out_w, "h": out_h}


def hero_panels(stop: dict, sc: dict) -> list[str]:
    """The stop's own-scene panels that show at least one of its zones, most zones first."""
    scene = sc.get("scenes", {}).get(stop.get("scene") or "")
    if not scene:
        return []
    own = {z["id"] for z in stop["zones"]}
    ranked = []
    for pid in scene["panels"]:
        spots = [sp for sp in sc["panels"][pid]["spots"] if sp["zone"] in own]
        if spots:
            area = sum((sp["bbox"][2] - sp["bbox"][0]) * (sp["bbox"][3] - sp["bbox"][1]) for sp in spots)
            ranked.append((-len(spots), -area, pid))
    return [pid for *_, pid in sorted(ranked)]


def zone_panels(zid: str, stop: dict, sc: dict) -> list[str]:
    """Every panel that draws the zone: the stop's own scene first, larger drawings first."""
    own = set(sc.get("scenes", {}).get(stop.get("scene") or "", {}).get("panels", []))
    ranked = []
    for pid, p in sc.get("panels", {}).items():
        for sp in p["spots"]:
            if sp["zone"] == zid:
                b = sp["bbox"]
                ranked.append((pid not in own, -(b[2] - b[0]) * (b[3] - b[1]), pid))
    return [pid for *_, pid in sorted(ranked)]


def _pick(d: dict, keys) -> dict:
    return {k: d[k] for k in keys if d.get(k) not in (None, "")}


def payload(m: dict, sc: dict, guide: dict, pictures: dict) -> dict:
    companies = {t: _pick(c, COMPANY_KEYS) for t, c in m["companies"].items() if c.get("market") not in HIDE_MARKETS}
    stops = []
    for s in m["stops"]:
        zones = []
        for i, z in enumerate(s["zones"]):
            zz = _pick(z, ZONE_KEYS)
            zz["n"] = f"{s['no']}-{i + 1}"
            zz["easy"] = guide["zones"].get(z["id"], "")
            zz["members"] = [_pick(mem, MEMBER_KEYS) for mem in z.get("members", []) if mem["t"] in companies]
            zz["crops"] = pictures["zones"].get(z["id"], [])
            zones.append(zz)
        st = _pick(s, STOP_KEYS)
        st.update(guide["stops"][s["id"]])
        st["zones"] = zones
        st["heroes"] = pictures["heroes"].get(s["id"], [])
        if s["id"] in pictures["thumbs"]:
            st["thumb"] = pictures["thumbs"][s["id"]]
        stops.append(st)
    panels = {}
    for pid, p in sc.get("panels", {}).items():
        scene_title = next((v["title"] for v in sc["scenes"].values() if pid in v["panels"]), "")
        panels[pid] = {
            "img": f"scenes/{p['img']}", "w": p["w"], "h": p["h"], "title": p["title"] or scene_title,
            "scene": scene_title, "caption": p["caption"], "legend": p["legend"],
            "spots": [{k: sp[k] for k in ("zone", "anchor", "box", "end", "text")} for sp in p["spots"]],
        }
    placed = {mem["t"] for st in stops for z in st["zones"] for mem in z["members"]}
    linked = {ln["t"] for ln in m["links"]}
    return {
        "as_of": m["as_of"], "built": date.today().isoformat(), "title": guide["title"],
        "intro": guide["intro"], "legs": guide["legs"], "kinds": guide["kinds"], "statuses": guide["statuses"],
        "glossary": guide["glossary"], "axes": m["axes"], "links": m["links"],
        "companies": companies, "stops": stops, "panels": panels,
        "counts": {"stops": len(stops), "zones": sum(len(st["zones"]) for st in stops),
                   "companies": len((placed | linked) & set(companies))},
    }


def bake(m: dict, sc: dict, out: Path) -> dict:
    """Spotlight crops (zones), lit full panels (stop heroes) and route thumbnails."""
    img_dir = out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Panel] = {}
    panel = lambda pid: cache.setdefault(pid, Panel(pid, sc["panels"][pid]))  # noqa: E731
    pics = {"zones": {}, "heroes": {}, "thumbs": {}}
    for s in m["stops"]:
        for z in s["zones"]:
            for pid in zone_panels(z["id"], s, sc):
                P = panel(pid)
                b = P.spots[z["id"]]["bbox"]
                min_w = ZONE_CROP["min_w"] * (0.8 if max(b[2] - b[0], b[3] - b[1]) < SMALL else 1)
                box = crop_box(b, P.meta["w"], P.meta["h"], ZONE_CROP["aspect"], min_w, ZONE_CROP["pad"])
                name = f"z-{z['id']}-{pid}.webp"
                size = P.save_crop(P.lit([z["id"]]), box, ZONE_CROP["out_w"], img_dir / name)
                pics["zones"].setdefault(z["id"], []).append({"panel": pid, "img": f"img/{name}", **size})
        own = [z["id"] for z in s["zones"]]
        for k, pid in enumerate(hero_panels(s, sc)):
            P = panel(pid)
            mine = [zid for zid in own if zid in P.spots]
            lit = P.lit(mine)
            name = f"s-{s['id']}-{pid}.webp"
            size = P.save_crop(lit, (0, 0, P.meta["w"], P.meta["h"]), HERO_W, img_dir / name)
            dots = [{"zone": zid, "x": round(P.spots[zid]["anchor"][0] / P.meta["w"], 4),
                     "y": round(P.spots[zid]["anchor"][1] / P.meta["h"], 4)} for zid in mine]
            pics["heroes"].setdefault(s["id"], []).append({"panel": pid, "img": f"img/{name}", "dots": dots, **size})
            if k == 0:
                box = crop_box(P.union_bbox(mine), P.meta["w"], P.meta["h"], THUMB_CROP["aspect"],
                               THUMB_CROP["min_w"], THUMB_CROP["pad"])
                tname = f"t-{s['id']}.webp"
                P.save_crop(lit, box, THUMB_CROP["out_w"], img_dir / tname)
                pics["thumbs"][s["id"]] = f"img/{tname}"
    return pics


def _bake_key(m: dict, sc: dict) -> str:
    """Changes when a render, the label layout, a stop's zones or this script change."""
    h = hashlib.sha1(Path(__file__).read_bytes())
    h.update(json.dumps(sc, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    h.update(json.dumps([(s["id"], s.get("scene"), [z["id"] for z in s["zones"]]) for s in m["stops"]]).encode("utf-8"))
    for p in sorted(desk_scenes.STATIC.glob("*.webp")):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
    return h.hexdigest()


def pictures(m: dict, sc: dict, out: Path) -> dict:
    """bake(), skipped when the baked pictures in out/img are still current (text-only edits)."""
    key, index = _bake_key(m, sc), out / "img" / "bake.json"
    try:
        saved = json.loads(index.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            return saved["pics"]
    except (FileNotFoundError, ValueError):
        pass
    shutil.rmtree(out / "img", ignore_errors=True)
    pics = bake(m, sc, out)
    index.write_text(json.dumps({"key": key, "pics": pics}, ensure_ascii=False), encoding="utf-8")
    return pics


def icons(out: Path) -> None:
    """Home-screen icons and the link-preview picture, drawn here so no font or tool is needed."""
    ink, gold, paper = (29, 26, 22), (240, 180, 41), (243, 238, 228)
    big = Image.new("RGB", (512, 512), ink)
    d = ImageDraw.Draw(big)
    d.line([(168, 404), (168, 262), (344, 262), (344, 150)], fill=gold, width=30, joint="curve")
    for x, y in ((168, 404), (256, 262)):
        d.ellipse([x - 34, y - 34, x + 34, y + 34], fill=paper, outline=gold, width=12)
    cx, cy, r, w = 344, 132, 74, 20   # four-point spark = the finished model
    d.polygon([(cx, cy - r), (cx + w, cy - w), (cx + r, cy), (cx + w, cy + w), (cx, cy + r), (cx - w, cy + w),
               (cx - r, cy), (cx - w, cy - w)], fill=paper)
    big.save(out / "icon-512.png")
    big.resize((192, 192), Image.LANCZOS).save(out / "icon-192.png")
    big.resize((180, 180), Image.LANCZOS).save(out / "apple-touch-icon.png")
    og = Image.new("RGB", (1200, 630), paper)
    pic = Image.open(desk_scenes.STATIC / "dc.webp").convert("RGBA")
    pic.thumbnail((1120, 590), Image.LANCZOS)
    og.paste(pic, ((1200 - pic.width) // 2, (630 - pic.height) // 2), pic)
    og.save(out / "og.jpg", quality=86)


def _clean(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.name in (".vercel", "img"):   # the Vercel project link; baked pictures (see pictures())
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()


def build(out: Path = DIST) -> dict:
    m, sc, guide = tour.load_map(), desk_scenes.load(), load_guide()
    problems = check(m, sc, guide)
    _clean(out)
    pics = pictures(m, sc, out)
    data = payload(m, sc, guide, pics)
    leaked = PRIVATE_KEYS & set(data)
    if leaked:
        raise RuntimeError(f"개인 자료 키가 섞였습니다: {sorted(leaked)}")
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (out / "data.json").write_text(raw, encoding="utf-8")
    scenes_dir = out / "scenes"
    scenes_dir.mkdir(exist_ok=True)
    for p in sc.get("panels", {}).values():
        shutil.copy2(desk_scenes.STATIC / p["img"], scenes_dir / p["img"])
    icons(out)
    digest = hashlib.sha1(raw.encode("utf-8"))
    for name in ("app.js", "style.css"):
        digest.update((WEB / name).read_bytes())
    v = digest.hexdigest()[:10]
    for p in WEB.iterdir():
        text = p.read_text(encoding="utf-8")
        (out / p.name).write_text(text.replace("__V__", v).replace("__SITE__", SITE), encoding="utf-8")
    files = [p for p in out.rglob("*") if p.is_file() and ".vercel" not in p.parts]
    return {"out": str(out), "version": v, "problems": problems, "files": len(files),
            "bytes": sum(p.stat().st_size for p in files), "counts": data["counts"]}
