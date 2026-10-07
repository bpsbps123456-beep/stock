import numpy as np
from gbrc.etf.research_concentration import project_caps, _simulate_one, candidate_grid


def data(opens,closes,order=None,schedule=None):
    op=np.asarray(opens,dtype=float);cl=np.asarray(closes,dtype=float);T,N=op.shape
    ranks=np.tile(np.arange(N),(T,1)) if order is None else np.asarray(order,dtype=np.int64)
    sched=np.zeros((T,3),dtype=np.bool_);sched[0]=True
    if schedule is not None: sched[:,1]=schedule
    return (op,cl,np.ones_like(op)*100,np.ones(T),np.ones(T),ranks,
            np.ones((3,T,N)),np.full((2,T,10),-1,dtype=np.int64),
            np.zeros((2,T,10)),np.ones((2,T)),sched,np.arange(N,dtype=np.int64),
            np.zeros(N,dtype=np.bool_),np.zeros(N,dtype=np.bool_),N+1)


def params(stock=1,sector=1,broad=1,overflow=0,trim=0,n=1,anchor=0,mix=0):
    return np.array([anchor,mix,n,0,1,stock,sector,broad,overflow,trim],dtype=float)


def test_canonical_grid_frozen_801():
    rows,configs=candidate_grid()
    assert len(rows)==len(configs)==801
    assert len(set(map(tuple,rows)))==801


def test_stock_and_sector_and_uncertainty_caps_are_simultaneous():
    raw=np.array([.5,.2,.2,.1]);sector=np.array([0,0,1,2]);broad=np.array([1,1,1,0],dtype=bool)
    p=project_caps(raw,np.zeros(4),sector,broad,.25,.4,.6,1)
    assert p.max()<=.25+1e-12
    assert p[:2].sum()<=.4+1e-12
    assert p[:3].sum()<=.6+1e-12
    assert p.sum()<=1
    np.testing.assert_allclose(p,[.25,.15,.2,.25])


def test_cash_vs_redistribution_and_permutation_invariance():
    raw=np.array([.8,.1,.1]);group=np.arange(3);sem=np.zeros(3,dtype=bool)
    cash=project_caps(raw,np.zeros(3),group,sem,.4,1.,1.,0)
    fill=project_caps(raw,np.zeros(3),group,sem,.4,1.,1.,1)
    np.testing.assert_allclose(cash,[.4,.1,.1])
    np.testing.assert_allclose(fill,[.4,.3,.3])
    perm=np.array([2,0,1]);p=project_caps(raw[perm],np.zeros(3),group[perm],sem[perm],.4,1.,1.,1)
    np.testing.assert_allclose(p,fill[perm])


def test_frozen_position_preserved_even_if_cap_impossible():
    p=project_caps(np.array([0.,.8]),np.array([.5,0.]),np.arange(2),np.zeros(2,dtype=bool),.25,1.,1.,1)
    np.testing.assert_allclose(p,[.5,.25])


def test_same_stock_two_signals_are_aggregated_before_cap():
    args=list(data([[100.,100.]],[[100.,100.]]))
    args[7][0,0,0]=0;args[8][0,0,0]=1
    result=_simulate_one(params(stock=.25,anchor=1,mix=.5),*args,0,True)
    np.testing.assert_allclose(result[3][0],[1.,0.])
    np.testing.assert_allclose(result[2][0],[.25,0.])


def test_overnight_gap_is_earned_before_next_open_cash_trim():
    args=data([[100.],[150.],[120.]],[[120.],[150.],[120.]])
    r,stats,w,raw,target,event=_simulate_one(params(stock=.5,trim=1),*args,0,True)
    np.testing.assert_allclose(np.cumprod(1+r),[1.1,1.25,1.125])
    np.testing.assert_allclose(w[:,0],[.6/1.1,.5,.5/1.125])
    assert stats[7]==1 and stats[16]==1
    assert target[1,0]<=raw[1,0]


def test_overlapping_regular_event_and_trim_is_one_fill():
    args=data([[100.],[150.]],[[120.],[150.]],schedule=[True,True])
    r,stats,w,raw,target,event=_simulate_one(params(stock=.5,trim=1),*args,0,True)
    assert stats[11]==2 and stats[16]==0
    np.testing.assert_allclose(np.cumprod(1+r),[1.1,1.25])


def test_missing_open_freezes_old_shares_and_flags_realized_cap():
    args=data([[100.],[np.nan]],[[120.],[150.]],schedule=[True,True])
    r,stats,w,raw,target,event=_simulate_one(params(stock=.5,trim=1),*args,0,True)
    np.testing.assert_allclose(np.cumprod(1+r),[1.1,1.25])
    assert stats[15]==1 and stats[7]==2
    assert stats[0]==1


def test_fresh_start_resets_momentum_buffer():
    args=data([[100.,100.],[100.,100.]],[[100.,100.],[100.,200.]],order=[[0,1],[1,0]],schedule=[True,True])
    full=_simulate_one(params(),*args,0,True)
    fresh=_simulate_one(params(),*args,1,True)
    np.testing.assert_allclose(full[2][-1],[1.,0.])
    np.testing.assert_allclose(fresh[2][-1],[0.,1.])
    np.testing.assert_allclose(fresh[0],[1.])
