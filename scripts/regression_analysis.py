"""
Regression Analysis Script - AI4S Interdisciplinary Research

Three models:
  Model 1: citation_impact ~ core predictors + controls
  Model 2: cit_interdisciplinarity ~ core predictors + controls
  Model 3: Rela_Dz ~ core predictors + controls

Core predictors: discipline_variety, discipline_similarity, discipline_balance,
                  ai4s_balance, ai_ref_age
Controls:        num_authors, publication_year, num_references,
                  num_institutions, has_international_collab, journal_impact, open_access

Usage:
    python regression_analysis.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools.tools import add_constant

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = "data/raw/ai4s_metrics_full_merged.csv"
OUTPUT_DIR = "data/regression/regression_v4"
USE_STANDARDIZED = True

# Core predictors
CORE_PREDICTORS = [
    "discipline_variety",
    "discipline_similarity",
    "discipline_balance",
    "ai4s_balance",
    "ai_ref_age_c",     # centered ai_ref_age
    "ai_ref_age_c_sq",  # squared term (based on centered value)
]

# Control variables
CONTROLS = [
    "num_authors",
    "num_references",
    "num_institutions",
    "has_international_collab",
    "journal_impact",
    "open_access",
    "first_author_hindex",
    "last_author_hindex",
    "max_author_hindex",
]

# Discipline fixed effects (auto-generated from primary_discipline)
# Set to True to include discipline dummy variables in the model
ENABLE_DISCIPLINE_FE = True
# Minimum number of papers per discipline to include as a dummy
# Disciplines with fewer papers will be grouped into "Other"
DISCIPLINE_FE_MIN_COUNT = 5

# Year fixed effects (auto-generated from publication_year)
# Replaces the continuous publication_year_c with year dummy variables
# to capture non-linear time effects (e.g., 2023 LLM breakout)
ENABLE_YEAR_FE = True

# Three dependent variables
# sqrt_Rela_Dz is the main model (square root transform reduces right skew)
# Original Rela_Dz is kept as robustness check
DEPENDENT_VARS = {
    "citation_impact": "Citation Impact (log)",
    "cit_interdisciplinarity": "Citing Interdisciplinarity",
    "sqrt_Rela_Dz": "Disruptiveness Index (sqrt)",
}

ALL_PREDICTORS = CORE_PREDICTORS + CONTROLS

# Variable name mapping for charts (English labels)
VAR_LABELS = {
    "citation_impact": "Citation Impact",
    "cit_interdisciplinarity": "Citing Interdisc.",
    "Rela_Dz": "Disruptiveness",
    "discipline_variety": "Discipline Variety",
    "discipline_similarity": "Discipline Similarity",
    "discipline_balance": "Discipline Balance",
    "ai4s_balance": "AI4S Balance",
    "ai_ref_age": "AI Ref Age",
    "ai_ref_age_c": "AI Ref Age (centered)",
    "ai_ref_age_c_sq": "AI Ref Age Sq (centered)",
    "num_authors": "Num Authors",
    "publication_year": "Pub Year",
    "publication_year_c": "Pub Year (centered)",
    "num_references": "Num References",
    "num_institutions": "Num Institutions",
    "has_international_collab": "Intl. Collab.",
    "journal_impact": "Journal Impact",
    "open_access": "Open Access",
    "first_author_hindex": "First Author H-index",
    "last_author_hindex": "Last Author H-index",
    "max_author_hindex": "Max Author H-index",
    "max_author_cited_by": "Max Author Cited By",
}


# ============================================================
# Data Loading & Cleaning
# ============================================================

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean data

    Args:
        filepath: CSV file path

    Returns:
        Cleaned DataFrame
    """
    print(f"Reading data: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Check for control variable columns with _ctrl suffix
    col_map = {}
    for col in ALL_PREDICTORS + list(DEPENDENT_VARS.keys()):
        if col in df.columns:
            col_map[col] = col
        elif f"{col}_ctrl" in df.columns:
            col_map[f"{col}_ctrl"] = col
            print(f"  Column mapping: {col}_ctrl -> {col}")

    # Rename columns
    if col_map:
        rename_dict = {k: v for k, v in col_map.items() if k != v}
        df = df.rename(columns=rename_dict)
        if rename_dict:
            print(f"  Renamed columns: {rename_dict}")

    # Select needed columns (only those that exist in CSV)
    # ai_ref_age_c, ai_ref_age_c_sq, publication_year_c are created later in main()
    skip_cols = ["ai_ref_age_c", "ai_ref_age_c_sq", "publication_year_c"]
    needed_cols = list(DEPENDENT_VARS.keys()) + [c for c in ALL_PREDICTORS
                                                  if c not in skip_cols]
    available_cols = [c for c in needed_cols if c in df.columns]
    missing_cols = [c for c in needed_cols if c not in df.columns]

    if missing_cols:
        print(f"  Warning: missing columns: {missing_cols}")

    # Keep ai_ref_age and publication_year if they exist (needed for centering later)
    for keep_col in ["ai_ref_age", "publication_year"]:
        if keep_col in df.columns and keep_col not in available_cols:
            available_cols.append(keep_col)

    # Keep Rela_Dz if it exists (needed for sqrt transform later)
    if "Rela_Dz" in df.columns and "Rela_Dz" not in available_cols:
        available_cols.append("Rela_Dz")

    # Keep primary_discipline if it exists (needed for discipline fixed effects)
    if "primary_discipline" in df.columns and "primary_discipline" not in available_cols:
        available_cols.append("primary_discipline")

    df = df[available_cols].copy()

    # Ensure numeric types (skip string columns like primary_discipline)
    for col in df.columns:
        if col == "primary_discipline":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check missing values
    print("\nMissing values per column:")
    missing_counts = df.isnull().sum()
    missing_cols_with_na = missing_counts[missing_counts > 0]
    if len(missing_cols_with_na) > 0:
        print(missing_cols_with_na)
    else:
        print("  No missing values")

    # Handle missing values
    for col in df.columns:
        na_count = df[col].isnull().sum()
        if na_count > 0 and na_count < len(df) * 0.5:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Column '{col}': {na_count} missing, filled with median {median_val:.4f}")
        elif na_count > 0:
            print(f"  Column '{col}': {na_count} missing (>50%), dropping column")
            df = df.drop(columns=[col])

    # Drop remaining rows with NaN
    before_drop = len(df)
    df = df.dropna()
    after_drop = len(df)
    if before_drop > after_drop:
        print(f"Dropped rows with NaN: {before_drop} -> {after_drop} (removed {before_drop - after_drop})")

    print(f"Final data: {df.shape[0]} rows, {df.shape[1]} variables")
    return df


# ============================================================
# Descriptive Statistics
# ============================================================

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics

    Args:
        df: DataFrame

    Returns:
        Descriptive statistics table
    """
    stats = df.describe().T
    stats = stats[["count", "mean", "std", "min", "max"]]
    stats["count"] = stats["count"].astype(int)
    stats = stats.round(4)
    return stats


def correlation_heatmap(df: pd.DataFrame, save_path: str):
    """
    Plot correlation heatmap (core variables only, excluding year/disc dummies)

    Args:
        df: DataFrame
        save_path: Save path
    """
    # Exclude raw variables that have centered versions (avoid duplicates)
    # Also exclude string columns (like primary_discipline), year/disc dummy variables
    exclude_cols = ["publication_year", "ai_ref_age", "primary_discipline"]
    plot_cols = [c for c in df.columns if c not in exclude_cols]
    # Exclude year_* and disc_* dummy variables
    plot_cols = [c for c in plot_cols if not c.startswith("year_") and not c.startswith("disc_")]
    # Also exclude any non-numeric columns
    plot_cols = [c for c in plot_cols if pd.api.types.is_numeric_dtype(df[c])]
    corr = df[plot_cols].corr(method="pearson")

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    labels = [VAR_LABELS.get(c, c) for c in corr.columns]

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Variable Correlation Matrix (Core Variables)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Correlation heatmap saved: {save_path}")


# ============================================================
# OLS Regression
# ============================================================

def run_ols_regression(df: pd.DataFrame, y_name: str, y_label: str,
                       X_names: list, save_dir: str,
                       cluster_col: str = None):
    """
    Run OLS regression with multiple standard error specifications

    Args:
        df: DataFrame
        y_name: Dependent variable name
        y_label: Dependent variable label
        X_names: Independent variable names
        save_dir: Save directory
        cluster_col: Column name for cluster-robust standard errors (e.g., "primary_discipline")

    Returns:
        Regression results (ordinary OLS)
    """
    print(f"\n{'=' * 70}")
    print(f"Model: {y_label} ({y_name})")
    print(f"{'=' * 70}")

    # Prepare data
    y = df[y_name]
    X = df[X_names].copy()

    # Debug: check for object columns
    obj_cols_in_X = [c for c in X.columns if X[c].dtype == 'object']
    if obj_cols_in_X:
        print(f"  WARNING: Object columns in X: {obj_cols_in_X}")
        for c in obj_cols_in_X:
            X[c] = pd.to_numeric(X[c], errors='coerce')

    # Debug: check all dtypes
    for c in X.columns:
        if X[c].dtype not in ['int64', 'float64', 'int32', 'float32', 'int8', 'int16', 'uint8', 'uint16', 'uint32', 'uint64', 'bool']:
            print(f"  WARNING: Column '{c}' has dtype {X[c].dtype}")

    # Add constant
    X = add_constant(X)

    # Debug: check X after add_constant
    print(f"  X shape: {X.shape}, dtypes: {X.dtypes.value_counts().to_dict()}")
    print(f"  y dtype: {y.dtype}")

    # Run OLS with ordinary standard errors
    model = sm.OLS(y, X)
    results = model.fit()

    # Run OLS with robust standard errors (HC3 - Huber-White)
    results_robust = model.fit(cov_type="HC3")

    # Run OLS with cluster-robust standard errors (by discipline)
    results_cluster = None
    if cluster_col is not None and cluster_col in df.columns:
        # Create cluster groups (ensure they are categorical with group labels)
        cluster_groups = df[cluster_col].astype(str)
        # statsmodels requires cluster groups to be encoded as integers
        # We pass the raw group labels and let statsmodels handle it
        results_cluster = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': cluster_groups},
        )

    # Print results
    print("\n--- Ordinary Standard Errors ---")
    print(results.summary())
    print("\n--- Robust Standard Errors (HC3) ---")
    print(results_robust.summary())
    if results_cluster is not None:
        print(f"\n--- Cluster-Robust SE (by {cluster_col}) ---")
        print(results_cluster.summary())

    # Save results to text file (all versions)
    result_file = os.path.join(save_dir, f"regression_{y_name}.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {y_label} ({y_name})\n")
        f.write(f"N: {len(df)}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write("=== Ordinary Standard Errors ===\n\n")
        f.write(results.summary().as_text())
        f.write("\n\n")
        f.write("=== Robust Standard Errors (HC3) ===\n\n")
        f.write(results_robust.summary().as_text())
        if results_cluster is not None:
            f.write(f"\n\n=== Cluster-Robust SE (by {cluster_col}) ===\n\n")
            f.write(results_cluster.summary().as_text())
    print(f"Regression results saved: {result_file}")

    # Standardized coefficients (Beta)
    if USE_STANDARDIZED:
        y_std = (y - y.mean()) / y.std()
        X_std = X.copy()
        for col in X_names:
            X_std[col] = (df[col] - df[col].mean()) / df[col].std()
        model_std = sm.OLS(y_std, X_std)
        results_std = model_std.fit()

        beta_file = os.path.join(save_dir, f"regression_{y_name}_beta.txt")
        with open(beta_file, "w", encoding="utf-8") as f:
            f.write(f"Standardized Coefficients (Beta) - {y_label}\n")
            f.write(f"{'=' * 70}\n\n")
            f.write(results_std.summary().as_text())
        print(f"Standardized coefficients saved: {beta_file}")

    # Save standard error comparison table (OLS vs HC3 vs Cluster)
    se_compare_file = os.path.join(save_dir, f"regression_{y_name}_se_comparison.txt")
    with open(se_compare_file, "w", encoding="utf-8") as f:
        f.write(f"Standard Error Comparison - {y_label}\n")
        f.write(f"{'=' * 70}\n\n")

        # Determine which SE columns to include
        se_types = [("OLS", results.bse)]
        se_types.append(("HC3", results_robust.bse))
        if results_cluster is not None:
            cluster_label = f"Cluster({cluster_col})"
            se_types.append((cluster_label, results_cluster.bse))

        # Header
        header = f"{'Variable':<25} {'Coef':>8}"
        for label, _ in se_types:
            header += f" {f'SE({label})':>12} {f't({label})':>8} {f'p({label})':>8} {'Sig':>4}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        # For each variable, collect all SE types
        for var in results.params.index:
            coef = results.params[var]
            line = f"{var:<25} {coef:>8.4f}"

            for label, se_dict in se_types:
                se = se_dict[var]
                t_val = coef / se if se != 0 else 0
                # Compute p-value using t-distribution
                from scipy import stats as scipy_stats
                p_val = 2 * (1 - scipy_stats.t.cdf(abs(t_val), df=len(df) - len(X_names) - 1))

                sig = ""
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                elif p_val < 0.1:
                    sig = "."

                line += f" {se:>12.4f} {t_val:>8.3f} {p_val:>8.4f} {sig:>4}"

            f.write(line + "\n")

        f.write("\n\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1\n")
        f.write("\nSE types:\n")
        f.write("  OLS: Ordinary standard errors (homoscedasticity assumed)\n")
        f.write("  HC3: Heteroscedasticity-consistent (Huber-White)\n")
        if results_cluster is not None:
            f.write(f"  Cluster({cluster_col}): Cluster-robust (accounts for within-{cluster_col} correlation)\n")
    print(f"SE comparison saved: {se_compare_file}")

    return results


# ============================================================
# Model Diagnostics
# ============================================================

def model_diagnostics(results, X_names: list, df: pd.DataFrame,
                      save_dir: str, model_name: str):
    """
    Model diagnostics

    Args:
        results: OLS results
        X_names: Independent variable names
        df: DataFrame
        save_dir: Save directory
        model_name: Model name
    """
    diag_file = os.path.join(save_dir, f"diagnostics_{model_name}.txt")
    y = results.model.endog
    X = results.model.exog

    with open(diag_file, "w", encoding="utf-8") as f:
        f.write(f"Model Diagnostics - {model_name}\n")
        f.write(f"{'=' * 70}\n\n")

        # 1. VIF Multicollinearity
        f.write("1. Multicollinearity (VIF)\n")
        f.write("-" * 50 + "\n")
        vif_data = pd.DataFrame()
        vif_data["Variable"] = ["const"] + X_names
        vif_data["VIF"] = [
            variance_inflation_factor(X, i) for i in range(X.shape[1])
        ]
        vif_data["Tolerance"] = 1 / vif_data["VIF"]
        f.write(vif_data.to_string(index=False))
        f.write("\n\nVIF > 10 indicates severe multicollinearity\n\n")

        # 2. Breusch-Pagan heteroscedasticity test
        f.write("2. Heteroscedasticity Test (Breusch-Pagan)\n")
        f.write("-" * 50 + "\n")
        bp_test = het_breuschpagan(results.resid, X)
        labels = ["LM Statistic", "LM p-value", "F Statistic", "F p-value"]
        for label, value in zip(labels, bp_test):
            f.write(f"{label}: {value:.6f}\n")
        f.write("\np > 0.05 indicates no heteroscedasticity\n\n")

        # 3. Model fit metrics
        f.write("3. Model Fit Metrics\n")
        f.write("-" * 50 + "\n")
        f.write(f"R-squared:     {results.rsquared:.4f}\n")
        f.write(f"Adj R-squared: {results.rsquared_adj:.4f}\n")
        f.write(f"F-statistic:   {results.fvalue:.4f}\n")
        f.write(f"F p-value:     {results.f_pvalue:.6f}\n")
        f.write(f"AIC:           {results.aic:.2f}\n")
        f.write(f"BIC:           {results.bic:.2f}\n")
        f.write(f"N:             {int(results.nobs)}\n")

    print(f"Model diagnostics saved: {diag_file}")


# ============================================================
# Visualization
# ============================================================

def plot_regression_coefficients(results_list: list, model_names: list,
                                 X_names: list, save_path: str):
    """
    Plot regression coefficients comparison across models
    (core predictors + controls only, excluding year/disc dummies)

    Args:
        results_list: List of regression results
        model_names: List of model names
        X_names: List of predictor names
        save_path: Save path
    """
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))

    if n_models == 1:
        axes = [axes]

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    # Core variable names to display (exclude year_* and disc_* dummies)
    core_var_names = [v for v in X_names if not v.startswith("year_") and not v.startswith("disc_")]

    for idx, (results, name) in enumerate(zip(results_list, model_names)):
        ax = axes[idx]

        params = results.params
        conf = results.conf_int()
        pvalues = results.pvalues

        # Only show core variables
        var_names = [v for v in core_var_names if v in params.index]
        coefs = [params[v] for v in var_names]
        ci_lower = [conf.loc[v, 0] for v in var_names]
        ci_upper = [conf.loc[v, 1] for v in var_names]
        p_vals = [pvalues[v] for v in var_names]
        labels = [VAR_LABELS.get(v, v) for v in var_names]

        bar_colors = []
        for p in p_vals:
            if p < 0.05:
                bar_colors.append(colors[idx])
            else:
                bar_colors.append("lightgray")

        y_pos = range(len(var_names))
        ax.barh(y_pos, coefs, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.errorbar(coefs, y_pos, xerr=[
            [c - l for c, l in zip(coefs, ci_lower)],
            [u - c for c, u in zip(coefs, ci_upper)],
        ], fmt="none", ecolor="black", capsize=3)

        ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Coefficient (95% CI)", fontsize=11)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Regression coefficients plot saved: {save_path}")


def plot_residual_diagnostics(results, save_dir: str, model_name: str):
    """
    Plot residual diagnostics

    Args:
        results: OLS results
        save_dir: Save directory
        model_name: Model name
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Fitted
    ax1 = axes[0]
    fitted = results.fittedvalues
    residuals = results.resid
    ax1.scatter(fitted, residuals, alpha=0.6, color="#2E86AB", edgecolors="black", linewidth=0.5)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Fitted Values", fontsize=11)
    ax1.set_ylabel("Residuals", fontsize=11)
    ax1.set_title("Residuals vs Fitted", fontsize=12)
    ax1.grid(alpha=0.3)

    # Q-Q plot
    ax2 = axes[1]
    sm.qqplot(residuals, line="s", ax=ax2, color="#2E86AB", alpha=0.6)
    ax2.set_title("Q-Q Plot (Normality)", fontsize=12)
    ax2.grid(alpha=0.3)

    plt.suptitle(f"Residual Diagnostics - {model_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"residuals_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Residual diagnostics plot saved: {save_path}")


def plot_scatter_matrix(df: pd.DataFrame, save_path: str):
    """
    Plot scatter matrix of core variables

    Args:
        df: DataFrame
        save_path: Save path
    """
    plot_vars = CORE_PREDICTORS + list(DEPENDENT_VARS.keys())

    available_vars = [v for v in plot_vars if v in df.columns]
    plot_df = df[available_vars].copy()
    plot_df = plot_df.rename(columns=VAR_LABELS)

    plot_df = plot_df[[VAR_LABELS.get(v, v) for v in available_vars]]

    if plot_df.shape[1] < 2:
        print("  Scatter matrix: insufficient variables, skipping")
        return

    g = sns.PairGrid(plot_df, diag_sharey=False)
    g.map_upper(sns.scatterplot, alpha=0.6, s=50, color="#2E86AB")
    g.map_lower(sns.kdeplot, cmap="Blues", fill=True, alpha=0.5)
    g.map_diag(sns.histplot, kde=True, color="#2E86AB")

    plt.suptitle("Core Variable Scatter Matrix", fontsize=14, y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter matrix saved: {save_path}")


# ============================================================
# Robustness Check: Square Root Transform (for Rela_Dz)
# ============================================================

def run_sqrt_robustness(df: pd.DataFrame, y_name: str, y_label: str,
                        X_names: list, save_dir: str):
    """
    Run OLS on sqrt-transformed Rela_Dz as robustness check.

    Square root transform reduces right skew while preserving
    interpretability (unlike log which can't handle zeros).

    Args:
        df: DataFrame
        y_name: Dependent variable name
        y_label: Dependent variable label
        X_names: Independent variable names
        save_dir: Save directory
    """
    print(f"\n{'=' * 70}")
    print(f"Robustness Check: sqrt({y_label})")
    print(f"{'=' * 70}")

    y_orig = df[y_name]
    y_sqrt = np.sqrt(y_orig)

    print(f"\nSquare root transform applied: {y_name} -> sqrt({y_name})")
    print(f"  Original: mean={y_orig.mean():.4f}, median={y_orig.median():.4f}, skew={y_orig.skew():.4f}")
    print(f"  sqrt:     mean={y_sqrt.mean():.4f}, median={y_sqrt.median():.4f}, skew={y_sqrt.skew():.4f}")

    # Prepare data
    X = df[X_names].copy()
    X = add_constant(X)

    # Run OLS on sqrt-transformed y
    model = sm.OLS(y_sqrt, X)
    results = model.fit()

    print("\n" + results.summary().as_text())

    # Save results
    result_file = os.path.join(save_dir, f"sqrt_robustness_{y_name}.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Robustness Check: sqrt({y_label})\n")
        f.write(f"Transform: square root (reduces right skew)\n")
        f.write(f"N: {len(df)}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(results.summary().as_text())
        f.write("\n\n")
        f.write("Model Fit Metrics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"R-squared:     {results.rsquared:.4f}\n")
        f.write(f"Adj R-squared: {results.rsquared_adj:.4f}\n")
        f.write(f"AIC:           {results.aic:.2f}\n")
        f.write(f"BIC:           {results.bic:.2f}\n")

    print(f"Robustness check results saved: {result_file}")

    # Residual diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Fitted
    ax1 = axes[0]
    fitted = results.fittedvalues
    residuals = results.resid
    ax1.scatter(fitted, residuals, alpha=0.6, color="#F18F01", edgecolors="black", linewidth=0.5)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Fitted Values", fontsize=11)
    ax1.set_ylabel("Residuals", fontsize=11)
    ax1.set_title("Residuals vs Fitted (sqrt)", fontsize=12)
    ax1.grid(alpha=0.3)

    # Q-Q plot
    ax2 = axes[1]
    sm.qqplot(residuals, line="s", ax=ax2, color="#F18F01", alpha=0.6)
    ax2.set_title("Q-Q Plot (sqrt)", fontsize=12)
    ax2.grid(alpha=0.3)

    plt.suptitle(f"Robustness Check: sqrt({y_label})", fontsize=14, y=1.02)
    plt.tight_layout()
    diag_file = os.path.join(save_dir, f"sqrt_robustness_{y_name}_diagnostics.png")
    plt.savefig(diag_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Robustness check diagnostics plot saved: {diag_file}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    """Main function"""
    # Allow command-line overrides: python regression_analysis.py <input_file> <output_dir>
    global INPUT_FILE, OUTPUT_DIR
    if len(sys.argv) >= 2:
        INPUT_FILE = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_DIR = sys.argv[2]

    print("=" * 70)
    print("AI4S Regression Analysis")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    print("\n" + "=" * 70)
    print("Step 1: Data Loading & Cleaning")
    print("=" * 70)
    df = load_and_clean_data(INPUT_FILE)

    if len(df) < 10:
        print(f"Warning: Small sample size ({len(df)} rows), results may be unreliable")

    # Center ai_ref_age and create squared term (reduce multicollinearity)
    if "ai_ref_age" in df.columns:
        ai_ref_mean = df["ai_ref_age"].mean()
        df["ai_ref_age_c"] = df["ai_ref_age"] - ai_ref_mean
        df["ai_ref_age_c_sq"] = df["ai_ref_age_c"] ** 2
        print(f"Centered ai_ref_age (mean={ai_ref_mean:.4f})")
        print(f"Created ai_ref_age_c and ai_ref_age_c_sq (centered squared term)")

    # Center publication_year (reduce const VIF from large values like 2020-2023)
    if "publication_year" in df.columns:
        pub_year_mean = df["publication_year"].mean()
        df["publication_year_c"] = df["publication_year"] - pub_year_mean
        print(f"Centered publication_year (mean={pub_year_mean:.4f})")
        print(f"Created publication_year_c")

    # Create sqrt(Rela_Dz) as the main dependent variable for disruptiveness model
    # Square root transform reduces right skew while preserving interpretability
    if "Rela_Dz" in df.columns:
        df["sqrt_Rela_Dz"] = np.sqrt(df["Rela_Dz"])
        print(f"\nCreated sqrt_Rela_Dz (main DV for disruptiveness model)")
        print(f"  Rela_Dz:     mean={df['Rela_Dz'].mean():.4f}, median={df['Rela_Dz'].median():.4f}, skew={df['Rela_Dz'].skew():.4f}")
        print(f"  sqrt_Rela_Dz: mean={df['sqrt_Rela_Dz'].mean():.4f}, median={df['sqrt_Rela_Dz'].median():.4f}, skew={df['sqrt_Rela_Dz'].skew():.4f}")

    # Generate discipline fixed effects (dummy variables)
    discipline_fe_cols = []
    if ENABLE_DISCIPLINE_FE and "primary_discipline" in df.columns:
        print("\n" + "-" * 50)
        print("Generating Discipline Fixed Effects...")
        print("-" * 50)

        # Count papers per discipline
        disc_counts = df["primary_discipline"].value_counts()
        print(f"Discipline distribution ({len(disc_counts)} disciplines):")
        for disc, count in disc_counts.items():
            print(f"  {disc}: {count} papers")

        # Group rare disciplines into "Other"
        rare_disciplines = disc_counts[disc_counts < DISCIPLINE_FE_MIN_COUNT].index
        if len(rare_disciplines) > 0:
            df["primary_discipline"] = df["primary_discipline"].replace(
                rare_disciplines, "Other"
            )
            print(f"\nGrouped {len(rare_disciplines)} rare disciplines into 'Other'")
            print(f"(threshold: < {DISCIPLINE_FE_MIN_COUNT} papers)")

        # Create dummy variables (drop_first=True to avoid perfect multicollinearity)
        discipline_dummies = pd.get_dummies(
            df["primary_discipline"], prefix="disc", drop_first=True
        )
        discipline_fe_cols = list(discipline_dummies.columns)
        # Convert bool to int (0/1) for statsmodels compatibility
        for col in discipline_fe_cols:
            discipline_dummies[col] = discipline_dummies[col].astype(int)
        df = pd.concat([df, discipline_dummies], axis=1)

        print(f"Created {len(discipline_fe_cols)} discipline dummy variables:")
        for col in discipline_fe_cols:
            count = df[col].sum()
            print(f"  {col}: {int(count)} papers")

        # Add discipline dummies to VAR_LABELS
        for col in discipline_fe_cols:
            disc_name = col.replace("disc_", "")
            VAR_LABELS[col] = f"Discipline: {disc_name}"

    # Generate year fixed effects (dummy variables)
    # Replaces continuous publication_year_c with year dummies to capture
    # non-linear time effects (e.g., 2023 LLM breakout)
    year_fe_cols = []
    if ENABLE_YEAR_FE and "publication_year" in df.columns:
        print("\n" + "-" * 50)
        print("Generating Year Fixed Effects...")
        print("-" * 50)

        # Show year distribution
        year_counts = df["publication_year"].value_counts().sort_index()
        print(f"Year distribution ({len(year_counts)} years):")
        for yr, count in year_counts.items():
            print(f"  {int(yr)}: {int(count)} papers")

        # Create dummy variables (drop_first=True to avoid perfect multicollinearity)
        # Use the earliest year (2017) as the reference category
        year_dummies = pd.get_dummies(
            df["publication_year"], prefix="year", drop_first=True
        )
        year_fe_cols = list(year_dummies.columns)
        # Convert bool to int (0/1) and ensure numeric dtype
        for col in year_fe_cols:
            year_dummies[col] = year_dummies[col].astype(int)
        # Ensure all columns are numeric (not object)
        year_dummies = year_dummies.astype({col: 'int64' for col in year_fe_cols})
        df = pd.concat([df, year_dummies], axis=1)

        print(f"\nCreated {len(year_fe_cols)} year dummy variables:")
        for col in year_fe_cols:
            count = df[col].sum()
            print(f"  {col}: {int(count)} papers")

        # Add year dummies to VAR_LABELS
        for col in year_fe_cols:
            yr_name = col.replace("year_", "")
            VAR_LABELS[col] = f"Year: {yr_name}"

    # Rebuild available_X after creating centered variables, discipline FE, and year FE
    available_X = [v for v in ALL_PREDICTORS if v in df.columns]
    if discipline_fe_cols:
        available_X = available_X + discipline_fe_cols
    if year_fe_cols:
        available_X = available_X + year_fe_cols
    print(f"\nTotal predictors (including FE): {len(available_X)}")
    print(f"Available predictors ({len(available_X)}): {available_X}")

    # 2. Descriptive statistics
    print("\n" + "=" * 70)
    print("Step 2: Descriptive Statistics")
    print("=" * 70)
    stats = descriptive_stats(df)
    print("\nDescriptive Statistics:")
    print(stats.to_string())

    stats_file = os.path.join(OUTPUT_DIR, "descriptive_stats.csv")
    stats.to_csv(stats_file)
    print(f"\nDescriptive statistics saved: {stats_file}")

    # 3. Correlation matrix
    print("\n" + "=" * 70)
    print("Step 3: Correlation Analysis")
    print("=" * 70)
    corr_file = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    correlation_heatmap(df, corr_file)

    # 4. Scatter matrix
    scatter_file = os.path.join(OUTPUT_DIR, "scatter_matrix.png")
    plot_scatter_matrix(df, scatter_file)

    # 5. Regression analysis
    print("\n" + "=" * 70)
    print("Step 4: OLS Regression Analysis")
    print("=" * 70)

    # available_X already built above (including discipline FE and year FE)
    print(f"Available predictors ({len(available_X)}): {available_X}")

    results_list = []
    model_names = []

    for y_name, y_label in DEPENDENT_VARS.items():
        if y_name not in df.columns:
            print(f"  Skipping {y_name}: not in data")
            continue

        # Use primary_discipline for cluster-robust standard errors
        # (after grouping rare disciplines, the column still exists in df)
        cluster_col = "primary_discipline" if "primary_discipline" in df.columns else None
        results = run_ols_regression(
            df, y_name, y_label, available_X, OUTPUT_DIR,
            cluster_col=cluster_col,
        )
        results_list.append(results)
        model_names.append(y_label)

        model_diagnostics(results, available_X, df, OUTPUT_DIR, y_name)
        plot_residual_diagnostics(results, OUTPUT_DIR, y_name)

    # 6. Coefficient comparison plot
    if results_list:
        print("\n" + "=" * 70)
        print("Step 5: Coefficient Visualization")
        print("=" * 70)
        coef_file = os.path.join(OUTPUT_DIR, "regression_coefficients.png")
        plot_regression_coefficients(results_list, model_names, available_X, coef_file)

    # 7. Robustness check: sqrt transform for Rela_Dz (reduces right skew)
    print("\n" + "=" * 70)
    print("Step 6: Robustness Check - sqrt(Rela_Dz)")
    print("=" * 70)
    if "Rela_Dz" in df.columns:
        run_sqrt_robustness(df, "Rela_Dz", "Disruptiveness Index", available_X, OUTPUT_DIR)
    else:
        print("  Skipping robustness check: Rela_Dz not in data")

    # 8. Summary
    print("\n" + "=" * 70)
    print("Regression Analysis Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Finish time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\nGenerated files:")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
