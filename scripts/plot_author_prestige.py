"""
Plot: Author Prestige vs Disruptiveness
Visualizes the null result: author h-index has no significant effect on disruptiveness.

Three panels:
1. First Author H-index vs sqrt(Rela_Dz)
2. Last Author H-index vs sqrt(Rela_Dz)
3. Max Author H-index vs sqrt(Rela_Dz)

Usage:
    python scripts/plot_author_prestige.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = "data/raw/ai4s_metrics_full_merged.csv"
OUTPUT_DIR = "data/regression/regression_v4"
DPI = 600
FIG_WIDTH = 12
FIG_HEIGHT = 4.5

# ============================================================
# Load Data
# ============================================================
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=["Rela_Dz"])

# Create sqrt(Rela_Dz)
df["sqrt_Rela_Dz"] = np.sqrt(df["Rela_Dz"])

# Author prestige variables
prestige_vars = [
    ("first_author_hindex", "First Author H-index", "#2c7bb6"),
    ("last_author_hindex", "Last Author H-index", "#fdae61"),
    ("max_author_hindex", "Max Author H-index", "#d7191c"),
]

# ============================================================
# Plot
# ============================================================
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))

for idx, (var_name, var_label, color) in enumerate(prestige_vars):
    ax = axes[idx]

    # Drop missing for this variable
    plot_df = df.dropna(subset=[var_name, "sqrt_Rela_Dz"])

    x = plot_df[var_name].values
    y = plot_df["sqrt_Rela_Dz"].values

    # Scatter plot
    ax.scatter(x, y, alpha=0.25, s=15, c=color, edgecolors="none", label="Observed")

    # Linear fit
    slope = np.polyfit(x, y, 1)[0]
    x_range = np.linspace(x.min(), x.max(), 100)
    y_fit = np.polyval([slope, np.mean(y) - slope * np.mean(x)], x_range)
    ax.plot(x_range, y_fit, color="black", linewidth=2, linestyle="--",
            label=f"Linear fit (slope={slope:.4f})")

    # Labels
    ax.set_xlabel(var_label, fontsize=12)
    if idx == 0:
        ax.set_ylabel("Disruptiveness\nsqrt(Rela_Dz)", fontsize=12)
    else:
        ax.set_ylabel("")

    ax.set_title(f"{var_label}", fontsize=13, fontweight="bold")

    # Add stats text box
    from scipy import stats
    r, p = stats.pearsonr(x, y)
    textstr = f"r = {r:.3f}\np = {p:.4f}\nn = {len(plot_df)}"
    props = dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8)
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

plt.suptitle("Author Prestige vs Disruptiveness\n(No significant relationship)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUTPUT_DIR, "author_prestige_vs_disruptiveness.pdf")
png_path = os.path.join(OUTPUT_DIR, "author_prestige_vs_disruptiveness.png")
plt.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()
print("Done!")
