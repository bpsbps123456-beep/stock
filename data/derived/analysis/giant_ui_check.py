"""Run the real Giant page against saved prices on an isolated desk HTTP server."""
import copy
import json
import sys
import tempfile
import threading
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).with_name("giant_ui_tools")))
from playwright.sync_api import sync_playwright
from desk import config, giant, server


def run():
    data = giant.add_comparison_views(json.loads(giant.CACHE_PATH.read_text(encoding="utf-8")))
    errors = []
    out = Path(__file__).with_name("giant_ui")
    out.mkdir(exist_ok=True)
    # Suppress unrelated account data loading only in this test browser. The full page,
    # tab routing, theme controls, CSS and Giant scripts are served by the actual Handler.
    app_script = (config.STATIC / "app.js").read_text(encoding="utf-8").replace(
        "\nload();\nwindow.addEventListener", "\nwindow.addEventListener", 1)
    with tempfile.TemporaryDirectory() as journal, mock.patch.object(config, "JOURNAL_DIR", Path(journal)), \
            mock.patch.object(giant, "payload", side_effect=lambda **kwargs: copy.deepcopy(data)):
        http = server.DeskServer(("127.0.0.1", 0), server.Handler)
        port = http.server_address[1]
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe", headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1050})
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                page.route("**/static/app.js", lambda route: route.fulfill(body=app_script, content_type="text/javascript"))
                page.goto(f"http://127.0.0.1:{port}/?theme=light#giant")
                page.wait_for_selector("#gi-sectors .gi-row")
                assert page.evaluate("GI.level") == "tour"
                assert page.locator("#gi-sectors .gi-row").count() == len(data["tour_sectors"])
                page.get_by_role("button", name="전체 분야 S종목 보기", exact=True).click()
                assert page.locator("#gi-members tbody tr").count() == 37
                assert page.evaluate("GI.stockScope") == "all"
                page.get_by_role("radiogroup", name="종목 회복 등급", exact=True).get_by_role("radio", name="전체 233", exact=True).click()
                # S fields across every parent, with matching whale rows.
                page.get_by_role("radiogroup", name="분야 회복 등급", exact=True).get_by_role("radio", name="S 4", exact=True).click()
                assert page.locator("#gi-sectors .gi-row").count() == 4
                assert page.locator("#gi-whale .gi-row").count() == 4
                page.get_by_role("radio", name="표시 분야 전체", exact=True).click()
                assert page.evaluate("giStockUniverse().length === new Set(giSectorRows().flatMap(r => r.members)).size")
                page.get_by_role("radiogroup", name="종목 회복 등급", exact=True).get_by_role("radio", name=__import__('re').compile(r"^S \d+$")).click()
                assert page.evaluate("giStockRows().every(s => s.grade === 'S')")
                # All S stocks, irrespective of their field's grade; no duplicates.
                page.get_by_role("radio", name="AI 전체", exact=True).click()
                assert page.locator("#gi-members tbody tr").count() == 37
                assert page.evaluate("new Set(giStockRows().map(s => s.symbol)).size") == 37
                assert page.evaluate("giStockRows().some(s => !giSectorRows().some(r => r.members.includes(s.symbol)))")
                page.screenshot(path=str(out / "all_s_light.png"), full_page=True)
                # Each small field can be ungraded while individual companies have a grade.
                page.evaluate("GI.sectorGrade='all'; GI.stockScope='selected'; GI.stockGrade='all'; giSelect('tour:cpu', false)")
                assert page.locator("#gi-members tbody tr").count() == 4
                page.evaluate("giSelect('tour:litho', false)")
                assert page.locator("#gi-members tbody tr").count() == 1
                # Empty grades, no stale selection, no runtime error; recover through controls.
                page.evaluate("GI.sectorGrade='Z'; giEnsureSelection(); giRenderSectors(); giRenderMembers(); giRenderCharts(); giRenderWhale()")
                assert page.evaluate("GI.sel") is None
                assert page.locator("#gi-sectors .gi-empty").count() == 1
                assert page.locator("#gi-members .gi-empty").count() == 1
                page.get_by_role("radiogroup", name="분야 회복 등급", exact=True).get_by_role("radio", name="전체 53", exact=True).click()
                assert page.evaluate("GI.sel") is not None
                # Switch levels with the S filter active, sort members, and select a candidate.
                page.get_by_role("radiogroup", name="분야 회복 등급", exact=True).get_by_role("radio", name="S 4", exact=True).click()
                page.get_by_role("radio", name="기존 세부 분야 47", exact=True).click()
                assert page.evaluate("giSectorRows().every(r => r.grade === 'S')")
                page.get_by_role("radio", name="대분류 12", exact=True).click()
                page.get_by_role("radio", name="탐방 세부 분야 53", exact=True).click()
                page.get_by_role("radio", name="AI 전체", exact=True).click()
                page.get_by_role("radiogroup", name="종목 회복 등급", exact=True).get_by_role("radio", name="S 37", exact=True).click()
                page.get_by_role("button", name="종목 회복 등급", exact=True).click()
                assert page.evaluate("GI.sort.key") == "recovery"
                if page.locator("#gi-stars .gi-chip").count():
                    page.locator("#gi-stars .gi-chip").first.click()
                    assert page.evaluate("GI.stockScope") == "selected"
                    assert page.evaluate("GI.sectorGrade") == "all"
                # Both themes and a narrow viewport; scrolling remains inside each table.
                for theme in ["light", "dark"]:
                    page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
                    for width in [1440, 390]:
                        page.set_viewport_size({"width": width, "height": 1050})
                        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (theme, width)
                        page.screenshot(path=str(out / f"{theme}_{width}.png"), full_page=True)
                        page.evaluate("scrollTo(0, 0)")
                        page.screenshot(path=str(out / f"top_{theme}_{width}.png"))
                browser.close()
        finally:
            http.shutdown()
            http.server_close()
    assert not errors, errors
    print(json.dumps({"fields": len(data["tour_sectors"]), "s_fields": 4, "s_stocks": 37,
                      "themes": ["light", "dark"], "widths": [1440, 390], "page_errors": errors,
                      "prices_computed_at": data["computed_at"], "price_date": data["anchor"]["last_date"]}, ensure_ascii=False))


if __name__ == "__main__":
    run()
