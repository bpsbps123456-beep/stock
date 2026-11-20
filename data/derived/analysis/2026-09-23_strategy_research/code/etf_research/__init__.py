"""2026-09-23 ETF signal-combination research (252,720 configurations, concentration #774).

Moved out of the operating package ``gbrc.etf`` on 2026-11-21. The modules still use
``gbrc`` for snapshot storage, flows and the investable-universe rule
(``gbrc.etf.market.exclusion_reason``). Run from the ``code/`` folder, for example::

    cd data/derived/analysis/2026-09-23_strategy_research/code
    py -m etf_research.research_concentration --threads 8

Numba is needed for the grid engines: ``pip install -e ".[research]"``.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[6]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
