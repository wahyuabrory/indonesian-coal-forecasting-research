"""Generate static research figures from cached snapshots and committed metrics.

Figure 1: return distribution + rolling volatility per equity.
Figure 2: model RMSE relative to Naive per equity (from docs/results.md test RMSE).
Figure 3: feature-ablation effect (E1-E0, E2-E0) per model and equity.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "yahoo_cache"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SYMBOLS = {"ADRO": "adro_jk", "PTBA": "ptba_jk", "ITMG": "itmg_jk"}

# --- Figure 1 ---
fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=False)
for i, (name, slug) in enumerate(SYMBOLS.items()):
    df = pd.read_csv(CACHE / f"{slug}.csv", parse_dates=["Date"]).sort_values("Date")
    ret = np.log(df["Adj Close"]).diff()
    vol = ret.rolling(20).std()
    ax = axes[i, 0]
    ax.hist(ret.dropna(), bins=80)
    ax.set_title(f"{name} daily log-return distribution")
    ax.set_xlabel("log return")
    ax = axes[i, 1]
    ax.plot(df["Date"], vol)
    ax.set_title(f"{name} 20-day rolling volatility")
    ax.axvline(pd.Timestamp("2024-10-01"), linestyle="--", linewidth=1)
    if i == 0:
        axes[i, 1].text(
            pd.Timestamp("2024-10-01"), vol.max(), " ADRO/AADI event",
            fontsize=8, va="top",
        )
fig.tight_layout()
fig.savefig(FIG / "fig1_return_volatility.png", dpi=120)

# --- Figure 2: fold-selected winner test RMSE relative to Naive ---
final = pd.read_csv(ROOT / "results" / "final_test_metrics.csv")
eqs = ["ADRO.JK", "PTBA.JK", "ITMG.JK"]
fam_label = {"xgboost": "XGBoost", "gru": "GRU", "transformer": "Transformer"}
labels, vals = [], []
for eq in eqs:
    e0 = final[(final.target == eq) & (final.feature_group == "E0")]
    naive = float(e0[e0.model == "zero_return_naive"]["rmse"].iloc[0])
    champ = e0[e0.model != "zero_return_naive"].iloc[0]
    r = float(champ["rmse"])
    labels.append(f"{eq.replace('.JK', '')} {fam_label[champ['model']]}")
    vals.append((r - naive) / naive * 100)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, vals)
ax.axhline(0, linewidth=1)
ax.set_ylabel("test RMSE vs Naive (%)  [E0; negative = better]")
ax.set_title("Fold-selected model RMSE relative to zero-return Naive (final test)")
fig.tight_layout()
fig.savefig(FIG / "fig2_model_vs_naive.png", dpi=120)

# --- Figure 3: ablation delta vs E0 (fold-selected model test RMSE) ---
abl = pd.read_csv(ROOT / "results" / "feature_ablation.csv")
groups = ["E1-E0", "E2-E0", "E3-E0"]
x = np.arange(len(groups))
fig, ax = plt.subplots(figsize=(8, 4))
for j, eq in enumerate(eqs):
    base = float(abl[(abl.target == eq) & (abl.feature_group == "E0")]["test_rmse"].iloc[0])
    vals = [float(abl[(abl.target == eq) & (abl.feature_group == g)]["test_rmse"].iloc[0]) - base
            for g in ("E1", "E2", "E3")]
    ax.bar(x + (j - 1) * 0.25, vals, width=0.25, label=eq.replace(".JK", ""))
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.axhline(0, linewidth=1)
ax.set_ylabel("test RMSE delta (negative = improvement)")
ax.set_title("Feature-ablation effect vs stock history (fold-selected model)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig3_ablation.png", dpi=120)

print("wrote", sorted(p.name for p in FIG.glob("*.png")))
