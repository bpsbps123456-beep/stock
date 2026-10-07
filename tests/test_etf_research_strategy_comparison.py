"""Causal open accounting tests, independent of research output returns."""
import numpy as np
from gbrc.etf.research_strategy_comparison import (
    simulate_open, disclosure_indices, momentum120, claude_signals,
)


def signal(ranks, schedule=None, exposure=None, cap=1, buffer=0):
    ranks=np.array(ranks)
    return {'ranks':ranks,'basis':np.ones(ranks.shape),'max_stocks':cap,'buffer':buffer,
            'schedule':np.ones(len(ranks),bool) if schedule is None else np.array(schedule),
            'exposure':np.ones(len(ranks)) if exposure is None else np.array(exposure)}


def test_overnight_intraday_and_fresh_entry_are_separate():
    op=np.array([[100.],[121.]]);cl=np.array([[110.],[121.]])
    p=simulate_open(op,cl,np.ones(2),np.ones(2),signal([[0],[0]]))
    np.testing.assert_allclose(p['returns'],[.1,.1])
    fresh=simulate_open(op,cl,np.ones(2),np.ones(2),signal([[0],[0]]),start=1)
    np.testing.assert_allclose(fresh['returns'],[0.])


def test_open_rotation_sells_old_gap_then_earns_new_intraday():
    op=np.array([[100.,100.],[120.,100.]])
    cl=np.array([[100.,100.],[600.,110.]])
    p=simulate_open(op,cl,np.ones(2),np.ones(2),signal([[0],[1]]))
    np.testing.assert_allclose(p['nav'],[1.,1.32])
    assert p['holdings'][1]==[[1,1.0]]


def test_krw_cash_is_not_revalued_as_usd_cash():
    p=simulate_open(np.ones((2,1)),np.ones((2,1)),np.array([1.,2.]),np.array([2.,4.]),signal([[0],[0]],exposure=[0,0]))
    np.testing.assert_allclose(p['nav'],1.)


def test_missing_open_freezes_existing_position_without_exceeding_cap():
    op=np.array([[100.,100.],[np.nan,100.]])
    cl=np.array([[100.,100.],[110.,200.]])
    p=simulate_open(op,cl,np.ones(2),np.ones(2),signal([[0],[1]]))
    np.testing.assert_allclose(p['nav'],[1.,1.1])
    assert p['holdings'][1]==[[0,1.0]]
    assert p['counts']==[1,1]


def test_buffer_retains_incumbent_in_top_n_plus_buffer():
    s=signal([[0,1],[1,0]],cap=1,buffer=1)
    p=simulate_open(np.ones((2,2)),np.ones((2,2)),np.ones(2),np.ones(2),s)
    assert p['holdings'][1]==[[0,1.0]]


def test_us_session_delay_handles_weekend_pdf_without_looking_ahead():
    days=np.array(['2026-01-02','2026-01-05','2026-01-06'])
    pdf=np.array(['2026-01-02','2026-01-03','2026-01-06'])
    np.testing.assert_array_equal(disclosure_indices(days,pdf,0),[0,1,2])
    np.testing.assert_array_equal(disclosure_indices(days,pdf,1),[-1,0,1])
    np.testing.assert_array_equal(disclosure_indices(days,pdf,2),[-1,-1,0])


def test_current_and_future_closes_cannot_change_preopen_momentum():
    p=np.arange(1.,201.).reshape(-1,1); indices=np.arange(121,200)
    before=momentum120(p,indices); p[160:]*=1000
    after=momentum120(p,indices)
    np.testing.assert_allclose(before[indices<=160],after[indices<=160])
    assert np.any(before[indices>160]!=after[indices>160])


def test_time_clone_does_not_require_120_day_momentum_history():
    a={'prices':np.ones((1,2)),'prior_close':np.ones((1,2)),
       'weights':np.array([[[.6,.4],[0.,0.]]]),'dates':np.array(['2026-01-02'])}
    s=claude_signals(a,np.array([0]),np.array([[.3,np.nan]]),'TIME_clone')
    np.testing.assert_array_equal(s['ranks'],[[0,1]])
    h=claude_signals(a,np.array([0]),np.array([[.3,np.nan]]),'H6')
    np.testing.assert_array_equal(h['ranks'],[[0,-1]])
