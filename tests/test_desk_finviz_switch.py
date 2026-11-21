import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from desk import config, finviz_switch as switch, journal, strategies
from desk.state import Desk, last_completed_session


class DeskRegimeTest(unittest.TestCase):
    def inputs(self, end="2026-11-19"):
        cal = switch.market_calendar()
        days = list(cal.sessions_in_range(cal.session_offset(end, -110), end).strftime("%Y-%m-%d"))
        close = pd.DataFrame(100., index=days, columns=["SOXX", "A", "B"])
        book = SimpleNamespace(days=days, cls=close, opn=close.copy(), ticker={"W US":"W"})
        weekly = dict(target={"W US": .8}, ranked=[{"key":"W US"}], rebalance_day=False)
        day = cal.next_session(end).date().isoformat()
        month_start = cal.date_to_session(day[:8] + "01", direction="next")
        basket = dict(symbols=["A","B"], source="finviz_close", effective_date=month_start.date().isoformat(),
                      signal_date=cal.previous_session(month_start).date().isoformat())
        return book, weekly, day, end, basket

    def test_exit_recovery_hold_and_month_open(self):
        b, w, day, end, basket = self.inputs()
        b.cls.loc[end, "SOXX"] = 80.
        b.cls.loc[end, "A"] = 200.
        out = switch.decision(b,w,day,end,basket)
        self.assertTrue(out["order_due"])
        self.assertEqual(out["mode"], "finviz")
        self.assertAlmostEqual(out["new_cash_target"], 5/8)
        self.assertEqual(out["new_targets"][0], {"ticker":"A","weight":.25})
        b.cls.iloc[-2, 0] = 80.
        self.assertFalse(switch.decision(b,w,day,end,basket)["order_due"])
        b.cls.loc[end, "SOXX"] = 100.
        recovery = switch.decision(b,w,day,end,None)
        self.assertTrue(recovery["order_due"])
        self.assertEqual(recovery["new_targets"],[{"ticker":"W","weight":.8}])
        b,w,day,end,basket = self.inputs("2026-10-30")
        b.cls.iloc[-2:, 0] = 80.
        out = switch.decision(b,w,day,end,basket)
        self.assertTrue(out["order_due"])
        self.assertEqual(out["new_targets"][0]["weight"],1/7)

    def test_hold_does_not_liquidate_and_cash_only_requires_a_real_empty_basket(self):
        b,w,day,end,basket = self.inputs()
        keep = switch.decision(b,w,day,end,None)
        self.assertEqual((keep["mode"],keep["order_due"]),("desk",False))
        b.cls.iloc[-1, 0] = 50.
        for snapshot in (None, basket):
            b.cls.loc[end, "A"] = np.nan
            failed = switch.decision(b,w,day,end,snapshot)
            self.assertEqual((failed["mode"],failed["order_due"],failed["new_cash_target"]),("blocked",False,None))
        cash = switch.decision(b,w,day,end,{**basket,"symbols":[]})
        self.assertEqual((cash["mode"],cash["order_due"],cash["new_cash_target"]),("finviz",True,1.))

    def test_exact_completed_exchange_window(self):
        b,w,day,end,basket = self.inputs()
        b.cls.iloc[-90, 0] = np.nan
        self.assertEqual(switch.decision(b,w,day,end,basket)["mode"],"blocked")
        b.cls.iloc[-90, 0] = 100.
        self.assertEqual(switch.decision(b,w,day,b.days[-2],basket)["mode"],"blocked")
        self.assertEqual(last_completed_session(datetime(2026,11,27,13,6,tzinfo=config.NY)), "2026-11-27")

    def test_only_first_observed_preopen_decision_is_locked(self):
        b,w,day,end,basket = self.inputs()
        result=switch.decision(b,w,day,end,basket)
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)
            pre=datetime(2026,11,20,9,0,tzinfo=config.NY)
            post=pre.replace(hour=10)
            switch.record_decision(result,{},day,post,folder)
            switch.record_decision(dict(result,mode="blocked"),{},day,pre,folder)
            self.assertFalse(list(folder.iterdir()))
            switch.record_decision(result,{"W":.8},day,pre,folder)
            switch.record_decision(dict(result,mode="finviz"),{},day,pre,folder)
            saved=json.loads((folder/(day+".json")).read_text(encoding="utf-8"))
            self.assertEqual((saved["mode"],saved["weekly_targets"]),("desk",{"W":.8}))

    def test_account_endpoint_never_turns_a_missing_basket_into_sell_orders(self):
        b, weekly, session, end, basket = self.inputs()
        b.cls["W"] = 100.
        b.pdfs = [end]
        b.fx = np.ones(len(b.days))
        b.etf = pd.Series(10.,index=pd.to_datetime(b.days))
        b.names, b.sector = {}, {"W US":"기타"}
        weekly.update(pdf=end, raw={}, ranked=[dict(key="W US", ticker="W", mom120=.1,
                                                   rank=1, time_weight=.1, sector="기타")])
        b.cls.loc[end,"SOXX"] = 45.
        route = switch.decision(b,weekly,session,end,None)
        now = datetime(2026,11,20,7,0,tzinfo=config.NY)
        path = SimpleNamespace(days=b.days, events=[])
        model = np.full(len(b.days),100.)
        stats = dict(model=model, etf=model, etf_last=len(model)-1, mdd=0., dd_now=0.,
                     vs_etf_20d=0., vs_etf_long=None)
        held = [dict(type="deposit",date=b.days[0],amount=1000),
                dict(type="buy",date=b.days[0],ticker="W",qty=3.,price=100.)]
        desk = Desk()
        paper = switch.decision(b, weekly, session, end, basket)
        with patch.object(desk,"_decision",return_value=(b,path,now,session,now,weekly,route,paper)), \
             patch.object(desk,"_get_quotes",return_value={}), patch.object(strategies,"model_status",return_value=stats), \
             patch.object(journal,"load_entries",return_value=held), \
             patch.object(journal,"load_settings",return_value={"tolerance":.02}):
            state = desk.state()
            self.assertEqual(state["finviz_trend_paper"]["mode"], "finviz")
            self.assertIsNone(state["cash_target"])
            self.assertEqual(state["actions"],[])
            self.assertEqual(state["targets"],[])
            self.assertEqual(state["account"]["positions"][0]["qty"],3.)
