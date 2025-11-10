from dataclasses import dataclass
import numpy as np

@dataclass
class Leader:
    w_res: float = 1.0
    w_carbon: float = 1.0
    w_cost: float = 1.0
    def broadcast(self, t:int, carbon_series, price_series):
        t0 = max(0, t-24)
        return {
            'avg_ci': float(np.mean(carbon_series[t0:t+1])),
            'avg_price': float(np.mean(price_series[t0:t+1])),
            'weights': (self.w_res, self.w_carbon, self.w_cost)
        }
