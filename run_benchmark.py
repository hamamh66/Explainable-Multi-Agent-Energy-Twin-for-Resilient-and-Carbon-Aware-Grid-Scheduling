import argparse, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from emaet import make_panoramic_prior, EMAETSystem
from emaet.baselines import centralized_mpc_baseline, mas_wo_xai_baseline

def radar_plot(df, out_path):
    metrics = ['AvgRecoveryProxy','Carbon','Cost','Interpretability']
    labels = list(df['Method'])
    norm = {}
    for m in metrics:
        x = df[m].values.astype(float)
        if m in ['Carbon','Cost']:
            x = x.max() - x + x.min()
        mn, mx = x.min(), x.max() if x.max()>x.min() else x.min()+1e-9
        norm[m] = (x - mn) / (mx - mn + 1e-9)
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(6,6))
    for i, label in enumerate(labels):
        vals = [norm[m][i] for m in metrics]
        vals += vals[:1]
        plt.polar(angles, vals, marker='o', label=label)
    plt.xticks(angles[:-1], metrics)
    plt.yticks([0.25,0.5,0.75], ['0.25','0.5','0.75'])
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.10))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight'); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=int, default=168)
    ap.add_argument('--buses', type=int, default=33)
    ap.add_argument('--res-share', type=float, default=0.4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    load, res = make_panoramic_prior(hours=args.hours, buses=args.buses, res_share=args.res_share)
    t = np.arange(args.hours)
    price = 40 + 10*np.sin(2*np.pi*(t%24)/24 - 1.0) + 5*np.random.default_rng(args.seed).normal(size=args.hours)
    carbon = 420 - 160*(res.mean(axis=1)/np.maximum(load.mean(axis=1),1e-3))

    sys = EMAETSystem(buses=args.buses, hours=args.hours, res_share=args.res_share, seed=args.seed)
    df = sys.run()

    emaet_avg_recovery = float(df['recovery_proxy'].mean())
    emaet_carbon = float(((df['total_net_load']-df['total_res']).clip(lower=0)*carbon).sum())
    emaet_cost = float(((df['total_net_load']-df['total_res']).clip(lower=0)*price).sum())
    emaet_interp = 100.0 * (df['explainability'].apply(lambda s: 1 if isinstance(s,str) and len(s)>0 else 0).sum() / len(df))

    from emaet.baselines import centralized_mpc_baseline, mas_wo_xai_baseline
    mpc = centralized_mpc_baseline(load, res, price, carbon)
    mas = mas_wo_xai_baseline(load, res, price, carbon)

    summary = pd.DataFrame([
        {'Method':'EMAET (Proposed)','AvgRecoveryProxy':emaet_avg_recovery,'Carbon':emaet_carbon,'Cost':emaet_cost,'Interpretability':emaet_interp},
        {'Method':'Centralized MPC','AvgRecoveryProxy':mpc['avg_recovery_proxy'],'Carbon':mpc['carbon'],'Cost':mpc['cost'],'Interpretability':mpc['interpretability']},
        {'Method':'MAS w/o XAI','AvgRecoveryProxy':mas['avg_recovery_proxy'],'Carbon':mas['carbon'],'Cost':mas['cost'],'Interpretability':mas['interpretability']},
    ])
    os.makedirs('outputs', exist_ok=True); os.makedirs('outputs/plots', exist_ok=True)
    df.to_csv('outputs/emaet_timeseries_demo.csv', index=False)
    summary.to_csv('outputs/emaet_benchmark_summary.csv', index=False)
    radar_plot(summary, 'outputs/plots/benchmark_radar.png')
    print(summary.to_string(index=False))
    print('\nSaved outputs to ./outputs')

if __name__ == '__main__':
    main()
