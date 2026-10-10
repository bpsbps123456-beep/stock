"""야간 트레이딩 데스크 — the user's daily buy/sell dashboard for the 7-stock rule.

    py -m desk serve      # http://127.0.0.1:8321 (--port=N pins a port, --no-browser)
    py -m desk validate   # replay the engine against Codex #774 and check the frozen pause-rule numbers

Rule (docs/desk_system.md §3, Codex concentration study #774): TIME's US holdings ranked by
120-session momentum, top 7 kept while inside the top 12, TIME NAV weights, caps of 25% per
name, 60% per sector and 50% for semiconductors + unclassified (60% in #774 until 2026-10-10),
weekly at the open.
Decisions for a US session use PDFs dated strictly before it and closes through the prior
session.
"""
