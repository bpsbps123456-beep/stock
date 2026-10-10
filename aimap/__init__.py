"""AI 산업 지도 explainer — a public, phone-first reading site built from the desk's AI map.

    py -m aimap            # rebuild aimap/dist
    py -m aimap serve      # preview at http://127.0.0.1:8330/
    py -m aimap deploy     # rebuild, then publish aimap/dist to Vercel production

Content: desk/tour_map.json (stops, zones, companies, OpenAI·Anthropic deals) and the Blender
cutaways, plus the plain-language text in aimap/guide.json. No personal data is read.
Rules: docs/desk_system.md §8.3.
"""
