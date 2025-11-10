from dataclasses import dataclass

@dataclass
class FrequencyFilter:
    H_eff: float = 3.0
    D_eff: float = 1.0
    rocof_limit: float = 1.0
    nadir_limit: float = -0.5
    def admissible(self, deltaP: float, freq_state):
        dfdt = (deltaP - self.D_eff*freq_state['df']) / (2*self.H_eff)
        new_df = freq_state['df'] + dfdt
        new_f  = freq_state['f'] + new_df
        ok = (abs(dfdt) <= self.rocof_limit) and (new_f >= self.nadir_limit)
        return ok, {'f': new_f, 'df': new_df}
