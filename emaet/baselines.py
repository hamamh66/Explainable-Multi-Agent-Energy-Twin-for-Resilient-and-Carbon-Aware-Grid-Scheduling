import numpy as np
from .freq_filter import FrequencyFilter

def centralized_mpc_baseline(load, res, price, carbon):
    net = np.maximum(load.sum(axis=1) - res.sum(axis=1), 0.0)
    cost = price * net; co2 = carbon * net
    freq_state={'f':0.0,'df':0.0}; ff=FrequencyFilter()
    f_hist=[]
    for t in range(len(net)):
         deltaP = float(res[t].sum()) - float(net[t])
         ok, freq_state = ff.admissible(deltaP, freq_state)
         if not ok:
             deltaP *= 0.95
             ok, freq_state = ff.admissible(deltaP, freq_state)
         f_hist.append(freq_state['f'])
    rec = float(np.mean([max(0.0, 0.2-abs(f)) for f in f_hist]))
    return {'avg_recovery_proxy': rec, 'carbon': float(co2.sum()), 'cost': float(cost.sum()), 'interpretability': 0.0}

def mas_wo_xai_baseline(load, res, price, carbon):
    net = np.maximum(load.sum(axis=1) - res.sum(axis=1), 0.0)
    cost = price * net; co2 = carbon * net
    return {'avg_recovery_proxy': 0.0, 'carbon': float(co2.sum()), 'cost': float(cost.sum()), 'interpretability': 40.0}
