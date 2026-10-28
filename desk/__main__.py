"""py -m desk [serve|open|validate|digest|ant|news|bottom|forecast]   (open = 야간데스크.cmd: reuse or start, then show the page)"""

import sys


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if command == "validate":
        from .validate import main as run
        return run()
    if command == "serve":
        from .server import main as run
        return run()
    if command == "open":
        from .server import launch as run
        return run()
    if command == "ant":
        from .ant import main as run
        return run(sys.argv[2:])
    if command == "news":
        from .news import main as run
        return run(sys.argv[2:])
    if command == "bottom":
        from .bottom import main as run
        return run(sys.argv[2:])
    if command == "forecast":
        from .forecast import main as run
        return run(sys.argv[2:])
    if command == "digest":
        from .digest import briefing_block
        print(chr(10).join(briefing_block()))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
