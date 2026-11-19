import time, numpy as np
import harness as H
t0=time.time()
b=H.resample(3)
e60=H.ema(b.Close,60)
cross=(b.Close<e60)&(b.Close.shift(1)>=e60.shift(1))
pos=H.dedup(H.exec_index(cross,b),30)
ev=H.event_metrics(pos)
ctx=H.to_1m_mask(H.ctx_rally(b,20,0.015),b)
for rng,nm in ((H.TRAIN,'TRAIN'),(H.TEST,'TEST')):
    print(nm,H.summarize(ev,rng,'exit',ctx))
    state=H.exec_index(b.Close<e60,b)
    print(nm,'hold',H.hold_sim(state,rng))
print(round(time.time()-t0,1),'s')
