"""Exercise all desk tabs on a disposable journal and saved market data."""
import copy
import json
import re
import sys
import tempfile
import threading
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).with_name("giant_ui_tools")))
import pandas as pd
from playwright.sync_api import sync_playwright
from desk import config, giant, journal, marketdata, server
from desk.state import Desk


def run():
    saved = {f: pd.read_csv(marketdata._path(f), index_col=0, parse_dates=True) for f in marketdata.FIELDS}
    data = giant.add_comparison_views(json.loads(giant.CACHE_PATH.read_text(encoding="utf-8")))
    out = Path(__file__).with_name("desk_ui")
    out.mkdir(exist_ok=True)
    errors = []
    with tempfile.TemporaryDirectory() as scratch, \
            mock.patch.object(config, "JOURNAL_DIR", Path(scratch)), \
            mock.patch.object(config, "TRADES", Path(scratch) / "trades.json"), \
            mock.patch.object(config, "SETTINGS", Path(scratch) / "settings.json"), \
            mock.patch.object(marketdata, "history", side_effect=lambda *a, **kw: copy.deepcopy(saved)), \
            mock.patch.object(marketdata, "live_quotes", return_value={}):
        journal.add_entry({"date": "2026-09-01", "type": "deposit", "amount": 10000, "note": "UI 확인용 임시 계좌"})
        for ticker in ["NVDA", "DELL"]:
            journal.add_entry({"date": "2026-09-02", "type": "buy", "ticker": ticker,
                               "qty": 5, "price": float(saved["Close"][ticker].dropna().iloc[-1]), "note": "UI 확인용"})
        fixture = Desk().state()
        print("Fixture ready: saved prices and disposable account", flush=True)
        journal_before = config.TRADES.read_bytes()
        with mock.patch.object(server.DESK, "state", side_effect=lambda **kw: copy.deepcopy(fixture)), \
                mock.patch.object(giant, "payload", side_effect=lambda **kw: copy.deepcopy(data)):
            http = server.DeskServer(("127.0.0.1", 0), server.Handler)
            threading.Thread(target=http.serve_forever, daemon=True).start()
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe", headless=True)
                    page = browser.new_page(viewport={"width": 1920, "height": 1080}, reduced_motion="reduce")
                    page.on("pageerror", lambda exc: errors.append(str(exc)))
                    page.goto(f"http://127.0.0.1:{http.server_address[1]}/?theme=light#desk")
                    page.wait_for_selector("#tiles .tile")
                    page.wait_for_function("!document.querySelector('#app').classList.contains('loading')")
                    assert page.locator("#positions tbody tr").count() == 2
                    assert page.locator("#journal tbody tr").count() == 3
                    # Shortcuts scroll within a tab and leave the router's hash untouched.
                    page.get_by_role("button", name="매매 기록", exact=True).click()
                    assert page.evaluate("location.hash") == "#desk"
                    assert page.locator("#journal").bounding_box()["y"] >= page.locator(".topbar").bounding_box()["height"]
                    page.get_by_role("button", name="주문·계좌 요약", exact=True).click()
                    page.get_by_role("button", name="표로 보기", exact=True).first.click()
                    assert page.locator("#model-chart table").count() == 1
                    page.get_by_role("button", name="차트로 보기", exact=True).click()
                    # Both themes, desktop/tablet/mobile: fill the width, no page overflow.
                    for theme in ["light", "dark"]:
                        page.evaluate("t => document.documentElement.dataset.theme = t", theme)
                        for width in [2560, 1920, 1440, 1024, 768, 390, 360]:
                            page.set_viewport_size({"width": width, "height": 1080})
                            page.wait_for_timeout(180)
                            for tab, panel in [("desk", "app"), ("tour", "tour-view"), ("giant", "giant-view")]:
                                page.locator(f"#tab-{tab}").click()
                                if tab == "tour":
                                    page.wait_for_selector(".tr-flow-svg")
                                if tab == "giant":
                                    page.wait_for_selector("#gi-sectors .gi-row")
                                page.evaluate("scrollTo(0, 0)")
                                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (tab, theme, width, page.evaluate("document.documentElement.scrollWidth"))
                                bounds = page.locator(f"#{panel}").bounding_box()
                                assert bounds["width"] >= width * .97, (tab, width, bounds)
                                assert page.locator(f"#{panel} .section-nav").count() == 1
                                if width in [1920, 390]:
                                    page.screenshot(path=str(out / f"{tab}_{theme}_{width}.png"))
                            print(f"Layout passed: {theme}, {width}px, all tabs", flush=True)
                    # Field search handles aliases, IME input, empty results and focus continuity.
                    page.set_viewport_size({"width": 1920, "height": 1080})
                    page.locator("#gi-sector-search").fill("cpu")
                    assert page.locator("#gi-sectors .gi-row").count() == 1
                    assert page.evaluate("document.activeElement.id") == "gi-sector-search"
                    assert page.locator("#gi-whale .gi-row").count() == 1
                    assert page.locator("#gi-members tbody tr").count() == 4
                    page.locator("#gi-sector-search").fill("없는분야")
                    assert page.evaluate("GI.sel") is None
                    assert page.locator("#gi-members .gi-empty").count() == 1
                    page.locator("#gi-sector-search").fill("냉각")
                    assert page.locator("#gi-sectors .gi-row").count() >= 1
                    page.locator("#gi-sector-search").fill("")
                    # All S: deduplicated across fields, query independent, separate sector/stock grades.
                    page.get_by_role("button", name="전체 S등급 종목 보기", exact=True).click()
                    expected_s = sum(s["grade"] == "S" for s in data["stocks"].values())
                    assert page.locator("#gi-members tbody tr").count() == expected_s
                    page.locator("#gi-stock-search").fill("nvda")
                    assert page.evaluate("giStockRows().every(s => s.symbol === 'NVDA')")
                    assert page.evaluate("document.activeElement.id") == "gi-stock-search"
                    page.get_by_role("button", name="전체 S등급 종목 보기", exact=True).click()
                    assert page.evaluate("GI.stockQuery") == ""
                    assert page.locator("#gi-members tbody tr").count() == expected_s
                    page.get_by_role("radiogroup", name="분야 회복 등급", exact=True).get_by_role("radio", name=re.compile(r"^S \d+$")).click()
                    assert page.evaluate("giStockRows().length") == expected_s
                    page.get_by_role("radio", name="표시된 모든 분야", exact=True).click()
                    assert page.evaluate("giStockUniverse().length === new Set(giSectorRows().flatMap(r => r.members)).size")
                    page.get_by_role("radio", name="전체 AI 종목", exact=True).click()
                    # Switch metrics: all figures remain reachable and sorting still works.
                    for view, columns in [("가격·회복", 9), ("펀드·수급", 7), ("실적·규모", 9), ("모든 지표", 17)]:
                        page.get_by_role("radio", name=view, exact=True).click()
                        assert page.locator("#gi-members thead th").count() == columns
                        assert page.locator("#gi-members tbody tr").count() == expected_s
                        assert page.locator("#gi-members tbody tr").first.locator("td").count() == columns
                    page.get_by_role("button", name="선행 PER", exact=True).click()
                    assert page.evaluate("GI.sort.key") == "fwd_pe"
                    page.get_by_role("button", name="선행 PER ↓", exact=True).click()
                    assert page.evaluate("GI.sort.dir") == 1
                    assert page.evaluate("(() => { const rows = [...document.querySelectorAll('#gi-members tbody tr')]; const vals = rows.map(r => r.cells[15].textContent); const missing = vals.indexOf('—'); return missing < 0 || vals.slice(missing).every(v => v === '—'); })()")
                    page.get_by_role("radio", name="가격·회복", exact=True).click()
                    page.get_by_role("button", name="종목 회복 등급", exact=True).click()
                    assert page.evaluate("GI.sort.key") == "recovery"
                    page.locator("#gi-members").screenshot(path=str(out / "s_stock_comparison.png"))
                    page.get_by_role("button", name="전고점·50일선·실적 보기", exact=True).click()
                    assert page.locator(".gi-sector-table").evaluate("t => [...t.tHead.rows[0].cells].filter(c => getComputedStyle(c).display !== 'none').length") == 11
                    page.get_by_role("button", name="기본 지표 보기", exact=True).click()
                    assert page.locator(".gi-sector-table").evaluate("t => [...t.tHead.rows[0].cells].filter(c => getComputedStyle(c).display !== 'none').length") == 8
                    # Two-column explorer on wide screens and pinned table heading while scrolling.
                    sector = page.locator("#gi-sectors").bounding_box()
                    members = page.locator("#gi-members").bounding_box()
                    assert abs(sector["y"] - members["y"]) < 2 and members["x"] > sector["x"]
                    page.get_by_role("radiogroup", name="분야 회복 등급", exact=True).get_by_role("radio", name="전체 53", exact=True).click()
                    page.locator("#gi-sectors .gi-scroll").evaluate("n => n.scrollTop = 260")
                    box = page.locator("#gi-sectors .gi-scroll").bounding_box()
                    heading = page.locator("#gi-sectors th").first.bounding_box()
                    assert abs(heading["y"] - box["y"]) < 3
                    page.locator("#gi-sectors .gi-row").nth(6).click()
                    assert page.locator("#gi-sectors .gi-scroll").evaluate("n => n.scrollTop") > 0
                    page.set_viewport_size({"width": 1440, "height": 1080})
                    page.locator("#gi-sectors .gi-row").nth(2).click()
                    assert page.locator("#gi-members").bounding_box()["y"] < 220
                    page.set_viewport_size({"width": 1920, "height": 1080})
                    # SVG hover remains aligned when wide plots are capped in height.
                    page.get_by_role("button", name="비교 차트", exact=True).click()
                    dot = page.locator(".gi-dots-hot circle").first
                    if dot.count():
                        dot.hover()
                        assert page.locator("#gi-tip").is_visible()
                    # Remember the current reading position when returning to a tab.
                    old_y = page.evaluate("scrollY")
                    page.locator("#tab-desk").click()
                    page.locator("#tab-giant").click()
                    assert abs(page.evaluate("scrollY") - old_y) < 3
                    # Candidate selection clears searches; no stale scope or grade.
                    if page.locator("#gi-stars .gi-chip").count():
                        page.locator("#gi-stars .gi-chip").first.click()
                        assert page.evaluate("GI.stockScope") == "selected"
                        assert page.evaluate("GI.sectorGrade") == "all"
                    # Tour navigation, holdings lens, SVG/scene controls use the unchanged map.
                    page.locator("#tab-tour").click()
                    page.locator(".tr-node").nth(6).click()
                    assert page.evaluate("TR.data.map.stops.findIndex(s => s.id === TR.stop)") == 6
                    assert page.evaluate("location.hash") == "#tour"
                    page.locator(".tr-rooms .room").last.click()
                    assert page.locator(".tr-rooms .room[aria-pressed=true]").count() == 1
                    page.locator("#tr-visit").screenshot(path=str(out / "tour_visit.png"))
                    page.get_by_role("button", name="선택한 방문지", exact=True).click()
                    assert page.evaluate("location.hash") == "#tour"
                    # Theme action uses the same application control, including chart redrawing.
                    before = page.evaluate("document.documentElement.dataset.theme")
                    page.locator("#btn-theme").click()
                    assert page.evaluate("document.documentElement.dataset.theme") != before
                    assert not errors, errors
                    browser.close()
            finally:
                http.shutdown()
                http.server_close()
        assert config.TRADES.read_bytes() == journal_before, "Browser wrote to the disposable journal"
    print(json.dumps({"tabs": 3, "themes": ["light", "dark"], "widths": [2560, 1920, 1440, 1024, 768, 390, 360],
                      "s_stocks": expected_s, "page_errors": errors, "journal_writes_from_browser": 0}, ensure_ascii=False))


if __name__ == "__main__":
    run()
