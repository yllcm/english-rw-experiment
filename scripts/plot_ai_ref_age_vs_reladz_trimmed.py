"""
Plot: AI Ref Age vs Rela_Dz (Disruptiveness) - Trimmed version (ai_ref_age <= 15)
Shows the U-shaped relationship with quadratic fit, excluding extreme values.

Usage:
    python scripts/plot_ai_ref_age_vs_reladz_trimmed.py
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
FIG_WIDTH = 8
FIG_HEIGHT = 5

# ============================================================
# Load Data
# ============================================================
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=["ai_ref_age", "Rela_Dz"])

# Filter: keep only ai_ref_age <= 15
df = df[df["ai_ref_age"] <= 15].copy()

x = df["ai_ref_age"].values
y = df["Rela_Dz"].values

print(f"N = {len(df)} (ai_ref_age <= 15)")
print(f"ai_ref_age range: [{x.min():.1f}, {x.max():.1f}]")
print(f"Rela_Dz range: [{y.min():.4f}, {y.max():.4f}]")

# ============================================================
# Quadratic Fit
# ============================================================
# y = a + b*x + c*x^2
X_design = np.column_stack([np.ones_like(x), x, x**2])
coefs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
a, b, c = coefs

# Predicted values and confidence band
x_sorted = np.sort(x)
X_pred = np.column_stack([np.ones_like(x_sorted), x_sorted, x_sorted**2])
y_pred = X_pred @ coefs

# Bootstrap CI for the fitted curve
n_bootstrap = 5000
n_obs = len(x)
pred_samples = np.zeros((n_bootstrap, len(x_sorted)))

np.random.seed(42)
for i in range(n_bootstrap):
    idx = np.random.choice(n_obs, n_obs, replace=True)
    x_boot = x[idx]
    y_boot = y[idx]
    X_boot = np.column_stack([np.ones_like(x_boot), x_boot, x_boot**2])
    try:
        coefs_boot, _, _, _ = np.linalg.lstsq(X_boot, y_boot, rcond=None)
        pred_samples[i] = X_pred @ coefs_boot
    except:
        continue

ci_lower = np.percentile(pred_samples, 2.5, axis=0)
ci_upper = np.percentile(pred_samples, 97.5, axis=0)

# Find minimum point of the quadratic
# y = a + b*x + c*x^2, minimum at x = -b/(2*c)
x_min = -b / (2 * c)
y_min = a + b * x_min + c * x_min**2
print(f"\nQuadratic fit: y = {a:.4f} + {b:.4f}*x + {c:.4f}*x^2")
print(f"Minimum point: x = {x_min:.2f}, y = {y_min:.4f}")

# ============================================================
# Plot
# ============================================================
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Scatter plot with transparency
ax.scatter(x, y, alpha=0.3, s=20, c="#2c7bb6", edgecolors="none", label="Observed data")

# Fitted curve with CI
ax.plot(x_sorted, y_pred, color="#d7191c", linewidth=2.5, label="Quadratic fit")
ax.fill_between(x_sorted, ci_lower, ci_upper, color="#d7191c", alpha=0.12, label="95% CI")

# Mark minimum point
ax.axvline(x=x_min, color="#fdae61", linestyle="--", linewidth=1.5, alpha=0.8)
ax.plot(x_min, y_min, marker="D", color="#fdae61", markersize=8, zorder=5)
ax.annotate(
    f"Minimum\n(x={x_min:.1f} yr, y={y_min:.3f})",
    xy=(x_min, y_min),
    xytext=(x_min + 2, y_min + 0.03),
    fontsize=10,
    color="#fdae61",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#fdae61", lw=1.5),
)

# Labels and title
ax.set_xlabel("AI Reference Age (years)", fontsize=13)
ax.set_ylabel("Disruptiveness (Rela_Dz)", fontsize=13)
ax.set_title("AI Reference Age vs Disruptiveness\n(ai_ref_age ≤ 15, trimmed)", fontsize=14, fontweight="bold")

# Legend
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

# Add regression stats text box
r2 = 1 - np.sum((y - (a + b * x + c * x**2))**2) / np.sum((y - np.mean(y))**2)
textstr = f"Quadratic fit\n$R^2$ = {r2:.3f}\nn = {len(df)}"
props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=props)

plt.tight_layout()

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUTPUT_DIR, "ai_ref_age_vs_reladz_trimmed.pdf")
png_path = os.path.join(OUTPUT_DIR, "ai_ref_age_vs_reladz_trimmed.png")
plt.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
print(f"\nSaved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()
print("Done!")
