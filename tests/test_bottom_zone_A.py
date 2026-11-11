"""Causal and boundary contracts for the fixed bottom-zone research."""
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]/'data/derived/analysis/bottom_zone_20261111/code'


def module(name):
    old=sys.path[:]
    try:
        sys.path.insert(0,str(HERE))
        spec=importlib.util.spec_from_file_location('zone_test_'+name,HERE/(name+'.py'))
        m=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:sys.path[:]=old


class BottomZoneATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=module('panel_A')
        cls.r=module('run_A')

    def test_setting_contract(self):
        self.assertEqual(len(self.p.SIGNALS),19)
        self.assertEqual(len(set(self.p.SIGNALS)),19)
        self.assertEqual(19+8+7,34)

    def test_future_tampering_leaves_signals_unchanged(self):
        idx=pd.bdate_range('2023-01-03',periods=350)
        rng=np.random.default_rng(12)
        px=pd.DataFrame({'open':100+np.cumsum(rng.normal(0,.4,350)),
                         'close':100+np.cumsum(rng.normal(0,.4,350))},index=idx)
        src=pd.DataFrame(index=idx)
        for col in ['SOXS_buy','SOXL_buy','total_buy','total_net']:
            src[col]=rng.uniform(10,100,350)
        src['entry']=idx+pd.offsets.BDay(3)
        src['known_date']=src.entry
        src['published']=True
        src.index.name='date'
        cust=pd.DataFrame({'custody_date':idx,'source_date':idx,'entry':idx+pd.offsets.BDay(4),
                           'known_date':idx+pd.offsets.BDay(4),'amount':np.arange(350)+1000.,
                           'total':5000.,'shares':np.arange(350)+100.})
        cnn=pd.Series(rng.uniform(5,95,350),index=idx)
        notes=pd.DataFrame({'date':[str(idx[10].date()),str(idx[-1].date())],'family':['R','D']})
        x=self.p.assemble(src,cust,px,cnn,notes)
        cut=idx[260]
        src2=src.copy();src2.loc[src2.index>cut,['SOXS_buy','SOXL_buy','total_buy','total_net']]*=100
        px2=px.copy();px2.loc[px2.index>cut]*=3
        cu2=cust.copy();cu2.loc[cu2.custody_date>cut,['amount','shares']]*=10
        cnn2=cnn.copy();cnn2.loc[cnn2.index>cut]=99
        notes2=notes.copy();notes2.loc[1,'family']='T'
        y=self.p.assemble(src2,cu2,px2,cnn2,notes2)
        pd.testing.assert_frame_equal(x.loc[:cut,self.p.SIGNALS],y.loc[:cut,self.p.SIGNALS])

    def test_missing_window_and_new_source_not_zero(self):
        idx=pd.bdate_range('2023-01-03',periods=200)
        px=pd.DataFrame({'open':100.,'close':100.},index=idx)
        src=pd.DataFrame({'SOXS_buy':10.,'SOXL_buy':20.,'total_buy':50.,'total_net':10.},index=idx)
        src.loc[idx[150],'SOXS_buy']=np.nan
        v=self.p.source_signals(src,px)
        self.assertTrue(v.loc[idx[150:155],'S5'].isna().all())
        self.assertTrue(v.loc[idx[155:],'S5'].notna().all())

    def test_incomplete_failure_is_scored_success_pending(self):
        idx=pd.bdate_range('2026-01-02',periods=3)
        px=pd.DataFrame({'open':[100.,100.,100.],'close':[94.,100.,100.]},index=idx)
        x=self.r.outcomes(px)
        self.assertEqual(x.Z.iloc[0],0)
        self.assertTrue(pd.isna(x.Z.iloc[1]))
        self.assertTrue(x.R10.isna().all())

    def test_observable_times_and_cnn_date_labels(self):
        if not (HERE.parent/'data/source_flows.csv').exists():
            self.skipTest('Fresh raw edition not yet collected')
        args=self.p.inputs()
        self.assertAlmostEqual(args[3].loc['2023-01-03'],36.6)
        x=self.p.assemble(*args)
        self.assertAlmostEqual(x.loc['2023-01-04','G'],36.6)
        known=pd.to_datetime(x.source_known)
        self.assertTrue((known.dropna()<=known.dropna().index).all())
        known=pd.to_datetime(x.custody_known)
        self.assertTrue((known.dropna()<=known.dropna().index).all())
        after=x.loc[x.index>pd.Timestamp(args[4].date.max()),['ND','NR','NS','NT']]
        self.assertTrue(after.isna().all().all())
        self.assertTrue(x.loc[x.source_date.isna(),'S1H_status'].eq('NO_NEW_SOURCE').all())

    def test_kleene_combinations(self):
        x=pd.DataFrame({'s':[1.,0.,np.nan],'x':[np.nan,1.,1.],'t':[1.,np.nan,1.]})
        reps=dict(S='s',X='x',T='t',G=None,M=None,N=None)
        y=self.r.combo(x,reps)
        self.assertTrue(pd.isna(y.S_AND_X.iloc[0]))
        self.assertEqual(y.S_AND_X.iloc[1],0)
        self.assertEqual(y.K2.iloc[0],1)
        self.assertEqual(y.K2.iloc[2],1)
        self.assertTrue(pd.isna(y.K3.iloc[2]))

    def test_match_no_reuse_excludes_nearby(self):
        idx=pd.bdate_range('2026-01-02',periods=15)
        x=pd.DataFrame({'active_state':True,'dd60':np.arange(15),'ret5':np.arange(15)*2.,'ret1':np.arange(15)*3.},index=idx)
        pairs=self.r.matching(x,np.array([5,6]),np.array([3,4,8,9,10]))
        self.assertEqual(len({j for i,j in pairs}),len(pairs))
        self.assertTrue(all(abs(i-j)>2 for i,j in pairs))

    def test_event_spacing_cross_year(self):
        idx=pd.bdate_range('2025-12-22',periods=25)
        x=pd.DataFrame({'down':True,'s':1.},index=idx)
        self.assertEqual(self.r.event_indices(x,'s').tolist(),[0,10,20])


if __name__=='__main__':unittest.main()
