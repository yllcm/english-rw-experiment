"""
Plot: Year Fixed Effects Trend for Disruptiveness Model
Shows the systematic decline of disruptiveness from 2017 to 2025.

The coefficients represent the change in sqrt(Rela_Dz) relative to the
baseline year (2017), controlling for all other predictors.

Usage:
    python scripts/plot_year_fe_trend.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = "data/raw/ai4s_metrics_full_merged.csv"
OUTPUT_DIR = "data/regression/regression_v4"
DPI = 600
FIG_WIDTH = 8
FIG_HEIGHT = 5

# ============================================================
# Load and prepare data (same as regression_analysis.py)
# ============================================================
df = pd.read_csv(INPUT_FILE)

# Keep needed columns
keep_cols = [
    "sqrt_Rela_Dz", "Rela_Dz",
    "discipline_variety", "discipline_similarity", "discipline_balance",
    "ai4s_balance", "ai_ref_age", "num_authors", "num_references",
    "num_institutions", "has_international_collab", "journal_impact",
    "open_access", "first_author_hindex", "last_author_hindex",
    "max_author_hindex", "publication_year", "primary_discipline",
]
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols].copy()

# Create centered variables
df["ai_ref_age_c"] = df["ai_ref_age"] - df["ai_ref_age"].mean()
df["ai_ref_age_c_sq"] = df["ai_ref_age_c"] ** 2

# Create sqrt(Rela_Dz)
df["sqrt_Rela_Dz"] = np.sqrt(df["Rela_Dz"])

# Create discipline dummies
disc_counts = df["primary_discipline"].value_counts()
rare_disc = disc_counts[disc_counts < 5].index
df["primary_discipline"] = df["primary_discipline"].replace(rare_disc, "Other")
disc_dummies = pd.get_dummies(df["primary_discipline"], prefix="disc", drop_first=True)
for col in disc_dummies.columns:
    disc_dummies[col] = disc_dummies[col].astype(int)
df = pd.concat([df, disc_dummies], axis=1)

# Create year dummies
year_dummies = pd.get_dummies(df["publication_year"], prefix="year", drop_first=True)
for col in year_dummies.columns:
    year_dummies[col] = year_dummies[col].astype(int)
df = pd.concat([df, year_dummies], axis=1)

# Build predictor list
core_preds = ["discipline_variety", "discipline_similarity", "discipline_balance",
              "ai4s_balance", "ai_ref_age_c", "ai_ref_age_c_sq"]
controls = ["num_authors", "num_references", "num_institutions",
            "has_international_collab", "journal_impact", "open_access",
            "first_author_hindex", "last_author_hindex", "max_author_hindex"]
disc_fe_cols = list(disc_dummies.columns)
year_fe_cols = list(year_dummies.columns)

X_names = core_preds + controls + disc_fe_cols + year_fe_cols
X_names = [c for c in X_names if c in df.columns]

# Drop rows with NaN
plot_df = df[X_names + ["sqrt_Rela_Dz", "primary_discipline"]].dropna()
print(f"N after dropna: {len(plot_df)}")

# Run OLS with cluster SE
y = plot_df["sqrt_Rela_Dz"]
X = plot_df[X_names].copy()
X = add_constant(X)

model = sm.OLS(y, X)
results = model.fit(cov_type='cluster', cov_kwds={'groups': plot_df["primary_discipline"]})

# ============================================================
# Extract Year FE coefficients
# ============================================================
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
year_coefs = []
year_cis_lower = []
year_cis_upper = []

for yr in years:
    col_name = f"year_{yr}"
    if col_name in results.params.index:
        coef = results.params[col_name]
        se = results.bse[col_name]
        year_coefs.append(coef)
        year_cis_lower.append(coef - 1.96 * se)
        year_cis_upper.append(coef + 1.96 * se)
    else:
        year_coefs.append(np.nan)
        year_cis_lower.append(np.nan)
        year_cis_upper.append(np.nan)

# Add baseline year (2017, coefficient = 0)
all_years = [2017] + years
all_coefs = [0.0] + year_coefs
all_cis_lower = [0.0] + year_cis_lower
all_cis_upper = [0.0] + year_cis_upper

print("Year FE Coefficients (relative to 2017):")
print(f"{'Year':<8} {'Coef':>8} {'95% CI Lower':>14} {'95% CI Upper':>14}")
print("-" * 50)
for yr, c, lo, hi in zip(all_years, all_coefs, all_cis_lower, all_cis_upper):
    sig = ""
    if yr > 2017:
        col = f"year_{yr}"
        if col in results.pvalues.index and results.pvalues[col] < 0.05:
            sig = " *"
    print(f"{yr:<8} {c:>8.4f} {lo:>14.4f} {hi:>14.4f}{sig}")

# ============================================================
# Plot
# ============================================================
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Plot the coefficients with error bars
ax.errorbar(
    all_years, all_coefs,
    yerr=[
        [c - lo for c, lo in zip(all_coefs, all_cis_lower)],
        [hi - c for c, hi in zip(all_coefs, all_cis_upper)],
    ],
    fmt='o-', color="#d7191c", linewidth=2, markersize=8,
    capsize=5, capthick=1.5, ecolor='gray', elinewidth=1.5,
    label="Year FE coefficient (vs 2017)"
)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

# Highlight significant years
for yr, c, lo, hi in zip(all_years, all_coefs, all_cis_lower, all_cis_upper):
    if yr == 2017:
        continue
    col = f"year_{yr}"
    if col in results.pvalues.index and results.pvalues[col] < 0.001:
        ax.annotate("***", xy=(yr, hi + 0.008), ha='center', fontsize=11,
                    fontweight='bold', color='#d7191c')
    elif col in results.pvalues.index and results.pvalues[col] < 0.01:
        ax.annotate("**", xy=(yr, hi + 0.008), ha='center', fontsize=11,
                    fontweight='bold', color='#d7191c')
    elif col in results.pvalues.index and results.pvalues[col] < 0.05:
        ax.annotate("*", xy=(yr, hi + 0.008), ha='center', fontsize=11,
                    fontweight='bold', color='#d7191c')

# Labels and title
ax.set_xlabel("Publication Year", fontsize=13)
ax.set_ylabel("Year FE Coefficient\n(relative to 2017)", fontsize=13)
ax.set_title("Declining Disruptiveness Over Time\n(Year Fixed Effects, sqrt(Rela_Dz))",
             fontsize=14, fontweight="bold")

# X-axis ticks
ax.set_xticks(all_years)
ax.set_xticklabels(all_years, rotation=45)

# Add annotation for the trend
ax.annotate(
    "Systematic decline:\n2020-2025 all significant",
    xy=(2022.5, -0.15),
    fontsize=10, color="#d7191c", fontstyle="italic",
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
)

# Legend
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

plt.tight_layout()

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUTPUT_DIR, "year_fe_trend.pdf")
png_path = os.path.join(OUTPUT_DIR, "year_fe_trend.png")
plt.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
print(f"\nSaved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()
print("Done!")
