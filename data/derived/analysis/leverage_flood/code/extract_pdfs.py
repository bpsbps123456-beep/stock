"""Extract text from saved PDFs (KCMI, Cboe, Bian-He-Shue-Zhou, ESMA) and the BIS QR Sept 2026.

Writes plain-text dumps to the scratchpad so that keyword grep is possible.
Usage: py extract_pdfs.py
"""
import os
import re
import sys

import requests
from pypdf import PdfReader

TOOL_DIR = r"C:\Users\geunt\.claude\projects\C--Users-geunt-OneDrive-Desktop---\f45f5967-1c45-4296-b547-ebda57ff1943\tool-results"
SCRATCH = r"C:\Users\geunt\AppData\Local\Temp\claude\C--Users-geunt-OneDrive-Desktop---\f45f5967-1c45-4296-b547-ebda57ff1943\scratchpad"
os.makedirs(SCRATCH, exist_ok=True)

SAVED = {
    "kcmi": "webfetch-1791940630398-64ae4x.pdf",
    "cboe": "webfetch-1791940634075-vvm8mg.pdf",
    "bian": "webfetch-1791940685335-s1q381.pdf",
    "esma": "webfetch-1791940748062-inwmmb.pdf",
}

BIS_URLS = [
    "https://www.bis.org/publ/qtrpdf/r_qt2609.pdf",
    "https://www.bis.org/publ/qtrpdf/r_qt2609a.pdf",
    "https://www.bis.org/publications/qr-202609_0.pdf",
]


def dump(name, path):
    try:
        reader = PdfReader(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[{name}] open failed: {exc}")
        return None
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(f"\n===== PAGE {i+1} =====\n" + (page.extract_text() or ""))
        except Exception as exc:  # noqa: BLE001
            pages.append(f"\n===== PAGE {i+1} (extract failed: {exc}) =====\n")
    text = "".join(pages)
    out = os.path.join(SCRATCH, f"{name}.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[{name}] {len(reader.pages)} pages, {len(text)} chars -> {out}")
    return text


def main():
    for name, fn in SAVED.items():
        dump(name, os.path.join(TOOL_DIR, fn))

    headers = {"User-Agent": "Mozilla/5.0"}
    for url in BIS_URLS:
        try:
            r = requests.get(url, headers=headers, timeout=60)
        except Exception as exc:  # noqa: BLE001
            print(f"[bis] {url} -> error {exc}")
            continue
        print(f"[bis] {url} -> {r.status_code} {r.headers.get('content-type')} {len(r.content)} bytes")
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            tag = re.sub(r"[^a-z0-9]", "_", url.split("/")[-1].lower())
            path = os.path.join(SCRATCH, f"bis_{tag}")
            with open(path, "wb") as fh:
                fh.write(r.content)
            text = dump(f"bis_{tag}", path)
            if text and "Short on memory" in text:
                print(f"[bis] FOUND 'Short on memory' in {url}")


if __name__ == "__main__":
    main()
