import numpy as np

def permutation_importance(agent, obs, leader_prices, n_repeats=8):
    base = agent.act(obs, leader_prices)['score']
    feats = ['load_hat','res_hat','freq_margin','carbon_intensity','price','soc']
    imps = {}
    for k in feats:
        vals = []
        for _ in range(n_repeats):
            obs_perm = obs.copy()
            obs_perm[k] = obs_perm[k] + np.random.normal(scale=0.1*abs(obs_perm[k])+1e-3)
            vals.append(agent.act(obs_perm, leader_prices)['score'])
        imps[k] = float(np.mean(np.abs(np.array(vals) - base)))
    s = sum(imps.values()) + 1e-9
    for k in imps:
        imps[k] = 100.0*imps[k]/s
    return imps
