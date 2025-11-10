from dataclasses import dataclass, field
from typing import Dict
import numpy as np

def moving_avg(x, k=3):
    if k <= 1: return x.copy()
    pad = np.pad(x, (k//2, k-1-k//2), mode='edge')
    kern = np.ones(k)/k
    return np.convolve(pad, kern, mode='valid')

@dataclass
class DigitalTwin:
    bus_id: int
    p_max: float
    storage_capacity: float = 0.0
    storage_soc: float = 0.0

    def forecast(self, load_hist, res_hist, t:int) -> Dict[str,float]:
        L = moving_avg(load_hist[-6:], k=3).mean() if len(load_hist)>=6 else load_hist[-1]
        R = moving_avg(res_hist[-6:], k=3).mean() if len(res_hist)>=6 else res_hist[-1]
        return {'load_hat': float(max(L,0.0)), 'res_hat': float(max(R,0.0))}

@dataclass
class Agent:
    bus_id: int
    twin: DigitalTwin
    price_weight: float = 1.0
    carbon_weight: float = 1.0
    resilience_weight: float = 1.2
    flex_kwh: float = 0.2
    last_action: Dict[str, float] = field(default_factory=dict)

    def act(self, obs: Dict[str,float], leader_prices: Dict[str,float]) -> Dict[str,float]:
        load_hat = obs['load_hat']; res_hat = obs['res_hat']
        freq_m = obs['freq_margin']; ci = obs['carbon_intensity']
        price  = obs['price']; soc = obs.get('soc', 0.0)
        flex = min(self.flex_kwh, max(0.0, (0.2 - freq_m)))
        charge_signal = 1.0 if price < leader_prices['avg_price'] and ci < leader_prices['avg_ci'] else 0.0
        charge = discharge = 0.0
        if self.twin.storage_capacity > 0:
            if charge_signal and soc < 0.9:
                charge = min(0.3*self.twin.storage_capacity, self.twin.storage_capacity - soc*self.twin.storage_capacity)
            elif not charge_signal and soc > 0.2:
                discharge = min(0.3*self.twin.storage_capacity, soc*self.twin.storage_capacity)
        net_consumption = max(0.0, load_hat - res_hat - discharge + charge - flex)
        score = (self.price_weight*price*net_consumption +
                 self.carbon_weight*ci*net_consumption -
                 self.resilience_weight*freq_m)
        action = {'flex':flex,'charge':charge,'discharge':discharge,'net_consumption':net_consumption,'score':score}
        self.last_action = action
        return action
