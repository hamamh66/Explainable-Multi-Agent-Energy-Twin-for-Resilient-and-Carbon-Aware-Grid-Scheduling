import numpy as np, json
import pandas as pd
from .panoramic import make_panoramic_prior
from .agents import Agent, DigitalTwin
from .leader import Leader
from .freq_filter import FrequencyFilter
from .explain import permutation_importance

class EMAETSystem:
    def __init__(self, buses=33, hours=168, res_share=0.35, storage_frac=0.25, seed=42):
        self.buses=buses; self.hours=hours; self.res_share=res_share; self.storage_frac=storage_frac
        self.rng = np.random.default_rng(seed)
        self.twins=[]; self.agents=[]
        self.leader = Leader(); self.freq_filter = FrequencyFilter()
        self.results=[]

    def setup(self):
        load, res = make_panoramic_prior(self.hours, self.buses, self.res_share)
        self.load, self.res = load, res
        t = np.arange(self.hours)
        self.price = 40 + 10*np.sin(2*np.pi*(t%24)/24 - 1.0) + 5*self.rng.normal(size=self.hours)
        self.carbon = 420 - 160*(res.mean(axis=1)/np.maximum(load.mean(axis=1),1e-3))
        for b in range(self.buses):
            cap = 0.0
            if self.rng.random() < self.storage_frac:
                cap = float(self.rng.uniform(0.5, 1.5))
            twin = DigitalTwin(bus_id=b, p_max=2.0, storage_capacity=cap, storage_soc=float(self.rng.uniform(0.2,0.8)))
            self.twins.append(twin)
            self.agents.append(Agent(bus_id=b, twin=twin))
        self.freq_state={'f':0.0,'df':0.0}

    def step(self, t:int):
        leader_msg = self.leader.broadcast(t, self.carbon, self.price)
        total_net = 0.0
        for b, ag in enumerate(self.agents):
            obs = {
                'load_hat': float(self.load[t,b]),
                'res_hat': float(self.res[t,b]),
                'freq_margin': float(max(0.0, 0.2 - abs(self.freq_state['f']))),
                'carbon_intensity': float(self.carbon[t]),
                'price': float(self.price[t]),
                'soc': float(ag.twin.storage_soc)
            }
            a = ag.act(obs, leader_msg)
            if ag.twin.storage_capacity>0:
                ag.twin.storage_soc = float(np.clip(
                    ag.twin.storage_soc + (a['charge'] - a['discharge']) / max(ag.twin.storage_capacity,1e-6),
                    0.0, 1.0
                ))
            total_net += a['net_consumption']
        total_res = float(self.res[t].sum())
        deltaP = total_res - total_net
        ok, new_freq = self.freq_filter.admissible(deltaP, self.freq_state)
        if not ok:
            corr = np.sign(-new_freq['df'])*0.05*abs(new_freq['df'])
            total_net = total_net + corr
            ok, new_freq = self.freq_filter.admissible(total_res - total_net, self.freq_state)
        self.freq_state = new_freq
        cost = self.price[t]*max(total_net - total_res, 0.0)
        co2  = self.carbon[t]*max(total_net - total_res, 0.0)
        rt = max(0.0, 0.2 - abs(self.freq_state['f']))
        agg_imp = {}
        if t % 12 == 0:
            import numpy as np
            idxs = self.rng.choice(self.buses, size=min(3,self.buses), replace=False)
            xai = []
            for idx in idxs:
                ag = self.agents[idx]
                obs = {
                    'load_hat': float(self.load[t,idx]),
                    'res_hat': float(self.res[t,idx]),
                    'freq_margin': float(max(0.0, 0.2 - abs(self.freq_state['f']))),
                    'carbon_intensity': float(self.carbon[t]),
                    'price': float(self.price[t]),
                    'soc': float(ag.twin.storage_soc)
                }
                xai.append(permutation_importance(ag, obs, leader_msg, n_repeats=6))
            feats = list(xai[0].keys())
            agg_imp = {k: float(np.mean([d[k] for d in xai])) for k in feats}
        self.results.append({
            't': t, 'total_net_load': total_net, 'total_res': total_res, 'deltaP': deltaP,
            'freq_f': self.freq_state['f'], 'freq_df': self.freq_state['df'],
            'cost': cost, 'co2': co2, 'recovery_proxy': rt,
            'explainability': json.dumps(agg_imp) if agg_imp else ''
        })

    def run(self):
        self.setup()
        for t in range(self.hours):
            self.step(t)
        return pd.DataFrame(self.results)
