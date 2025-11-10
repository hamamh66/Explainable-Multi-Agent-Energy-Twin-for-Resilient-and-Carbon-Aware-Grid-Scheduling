# EMAET: Explainable Multi‑Agent Energy Twin (Colab‑Ready)

Minimal, reproducible reference implementation for the EMAET framework with baselines.
Runs out‑of‑the‑box on **Google Colab** and locally. No proprietary data required.

## Quickstart (Colab)

1. Open Colab and run:
```python
!git clone <YOUR-FORK-URL>.git
%cd emaet_repo
!pip install -r requirements.txt
!python run_benchmark.py --hours 168 --buses 33 --res-share 0.4 --seed 42
```
2. Outputs are written to `outputs/`:
   - `emaet_timeseries_demo.csv`
   - `emaet_benchmark_summary.csv`
   - `plots/benchmark_radar.png`

## Quickstart (no Git, Colab only)

```python
# Create package from notebook cell
!pip install -r requirements.txt
!python run_benchmark.py --hours 168 --buses 33
```

## Results
The script reproduces the paper's demo-level comparison (EMAET vs Centralized MPC, MAS w/o XAI).

## Cite
If you use this code, please cite the EMAET paper (preprint) and this repository.

## License
MIT
