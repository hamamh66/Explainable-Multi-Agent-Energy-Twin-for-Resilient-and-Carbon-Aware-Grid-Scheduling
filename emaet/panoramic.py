import numpy as np
RNG = np.random.default_rng(42)

def make_panoramic_prior(hours=168, buses=33, res_share=0.35):
    t = np.arange(hours)
    daily = 0.6 + 0.4*np.sin(2*np.pi*(t%24)/24 - 1.2) + 0.1*RNG.normal(size=hours)
    weekly = 0.9 + 0.1*np.sin(2*np.pi*t/(24*7))
    base = np.clip(daily*weekly, 0.1, None)
    bus_scales = np.clip(RNG.normal(1.0, 0.15, size=buses), 0.6, 1.4)
    load = np.outer(base, bus_scales)
    solar = np.clip(1.2*np.sin(2*np.pi*(t%24)/24 - np.pi/2), 0, None)
    wind = np.clip(0.6 + 0.4*np.sin(2*np.pi*t/(24) + 1.1) + 0.25*RNG.normal(size=hours), 0, 1.5)
    res = res_share * (0.7*solar[:,None] + 0.3*wind[:,None]) * bus_scales
    return load, res
