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
INPUT_FILE = "ai4s_metrics/results/ai4s_metrics_full_20260517_182337.csv"
OUTPUT_DIR = "ai4s_metrics/results/regression_v2"
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
    "publication_year_c",  # centered publication_year
    "num_references",
    "num_institutions",
    "has_international_collab",
    "journal_impact",
    "open_access",
]

# Three dependent variables
DEPENDENT_VARS = {
    "citation_impact": "Citation Impact (log)",
    "cit_interdisciplinarity": "Citing Interdisciplinarity",
    "Rela_Dz": "Disruptiveness Index",
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
        df = df.rename(columns={v: k for k, v in col_map.items() if v != k})

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

    df = df[available_cols].copy()

    # Ensure numeric types
    for col in df.columns:
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
    Plot correlation heatmap

    Args:
        df: DataFrame
        save_path: Save path
    """
    # Exclude raw variables that have centered versions (avoid duplicates)
    exclude_cols = ["publication_year", "ai_ref_age"]
    plot_cols = [c for c in df.columns if c not in exclude_cols]
    corr = df[plot_cols].corr(method="pearson")

    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    labels = [VAR_LABELS.get(c, c) for c in corr.columns]

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Variable Correlation Matrix", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Correlation heatmap saved: {save_path}")


# ============================================================
# OLS Regression
# ============================================================

def run_ols_regression(df: pd.DataFrame, y_name: str, y_label: str,
                       X_names: list, save_dir: str):
    """
    Run OLS regression

    Args:
        df: DataFrame
        y_name: Dependent variable name
        y_label: Dependent variable label
        X_names: Independent variable names
        save_dir: Save directory

    Returns:
        Regression results
    """
    print(f"\n{'=' * 70}")
    print(f"Model: {y_label} ({y_name})")
    print(f"{'=' * 70}")

    # Prepare data
    y = df[y_name]
    X = df[X_names].copy()

    # Add constant
    X = add_constant(X)

    # Run OLS
    model = sm.OLS(y, X)
    results = model.fit()

    # Print results
    print(results.summary())

    # Save results to text file
    result_file = os.path.join(save_dir, f"regression_{y_name}.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {y_label} ({y_name})\n")
        f.write(f"N: {len(df)}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(results.summary().as_text())
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

    for idx, (results, name) in enumerate(zip(results_list, model_names)):
        ax = axes[idx]

        params = results.params
        conf = results.conf_int()
        pvalues = results.pvalues

        var_names = [v for v in params.index if v != "const"]
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
# Ordered Logit Regression (for Rela_Dz)
# ============================================================

def discretize_disruptiveness(series: pd.Series) -> pd.Series:
    """
    Discretize Rela_Dz into ordered categories

    Categories:
        0: Low disruptiveness (Rela_Dz < 0.05)
        1: Medium disruptiveness (0.05 <= Rela_Dz < 0.2)
        2: High disruptiveness (Rela_Dz >= 0.2)

    Args:
        series: Rela_Dz values

    Returns:
        Categorical series (0, 1, 2)
    """
    bins = [-np.inf, 0.05, 0.2, np.inf]
    labels = [0, 1, 2]
    return pd.cut(series, bins=bins, labels=labels).astype(int)


def run_ordered_logit(df: pd.DataFrame, y_name: str, y_label: str,
                      X_names: list, save_dir: str):
    """
    Run Ordered Logit regression for Rela_Dz

    Args:
        df: DataFrame
        y_name: Dependent variable name
        y_label: Dependent variable label
        X_names: Independent variable names
        save_dir: Save directory
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    print(f"\n{'=' * 70}")
    print(f"Ordered Logit Model: {y_label} ({y_name})")
    print(f"{'=' * 70}")

    # Discretize dependent variable
    y_ordinal = discretize_disruptiveness(df[y_name])
    X = df[X_names].copy()

    # Print category distribution
    print("\nDisruptiveness category distribution:")
    cat_counts = y_ordinal.value_counts().sort_index()
    cat_labels = {0: "Low (<0.05)", 1: "Medium (0.05-0.2)", 2: "High (>=0.2)"}
    for cat, count in cat_counts.items():
        print(f"  {cat_labels[cat]}: {count} ({count/len(y_ordinal)*100:.1f}%)")

    # Fit Ordered Logit model
    model = OrderedModel(y_ordinal, X, distr="logit")
    results = model.fit(method="bfgs", maxiter=1000, disp=False)

    print("\n" + results.summary().as_text())

    # Save results
    result_file = os.path.join(save_dir, f"ordered_logit_{y_name}.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Ordered Logit Model: {y_label}\n")
        f.write(f"N: {len(df)}\n")
        f.write(f"{'=' * 70}\n\n")

        f.write("Category Distribution:\n")
        for cat, count in cat_counts.items():
            f.write(f"  {cat_labels[cat]}: {count} ({count/len(y_ordinal)*100:.1f}%)\n")
        f.write("\n")

        f.write(results.summary().as_text())
        f.write("\n\n")

        # Log-likelihood and fit metrics
        f.write("Model Fit Metrics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Log-Likelihood:     {results.llf:.4f}\n")
        f.write(f"LL-Null:            {results.llnull:.4f}\n")
        f.write(f"Pseudo R-squared:   {results.prsquared:.4f}\n")
        f.write(f"AIC:                {results.aic:.2f}\n")
        f.write(f"BIC:                {results.bic:.2f}\n")

    print(f"Ordered logit results saved: {result_file}")

    # Marginal effects (manually computed)
    marg_file = os.path.join(save_dir, f"ordered_logit_{y_name}_marginal.txt")
    with open(marg_file, "w", encoding="utf-8") as f:
        f.write(f"Marginal Effects - Ordered Logit: {y_label}\n")
        f.write(f"{'=' * 70}\n\n")

        # Manually compute average marginal effects
        # For ordered logit, marginal effects vary by observation and category
        # We compute the average change in probability for each category
        f.write("Note: OrderedModel does not support get_margeff() directly.\n")
        f.write("Below are the model coefficients (log-odds) for interpretation:\n\n")
        f.write("Coefficient interpretation:\n")
        f.write("- Positive coefficient: higher values increase probability of higher categories\n")
        f.write("- Negative coefficient: higher values increase probability of lower categories\n\n")
        f.write("Key significant predictors (p<0.05):\n")
        for var in X_names:
            pval = results.pvalues.get(var, 1.0)
            coef = results.params.get(var, 0)
            if pval < 0.05:
                direction = "increases" if coef > 0 else "decreases"
                f.write(f"  {var}: coef={coef:.4f}, p={pval:.4f} — {direction} disruptiveness level\n")
        f.write("\n")
        f.write(f"Thresholds:\n")
        f.write(f"  0|1 (Low vs Medium+High): {results.params.get('0/1', 0):.4f}\n")
        f.write(f"  1|2 (Low+Medium vs High): {results.params.get('1/2', 0):.4f}\n")

    print(f"Marginal effects saved: {marg_file}")

    # Confusion matrix
    predicted = results.predict(X).idxmax(axis=1)
    cm = pd.crosstab(y_ordinal, predicted,
                     rownames=["Actual"], colnames=["Predicted"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"])
    plt.title(f"Confusion Matrix - Ordered Logit\n{y_label}", fontsize=12)
    plt.tight_layout()
    cm_file = os.path.join(save_dir, f"ordered_logit_{y_name}_confusion.png")
    plt.savefig(cm_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {cm_file}")

    # Accuracy
    accuracy = (y_ordinal == predicted).mean()
    print(f"\nPrediction accuracy: {accuracy:.2%}")

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

    # Rebuild available_X after creating centered variables
    available_X = [v for v in ALL_PREDICTORS if v in df.columns]
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

    available_X = [v for v in ALL_PREDICTORS if v in df.columns]
    print(f"Available predictors ({len(available_X)}): {available_X}")

    results_list = []
    model_names = []

    for y_name, y_label in DEPENDENT_VARS.items():
        if y_name not in df.columns:
            print(f"  Skipping {y_name}: not in data")
            continue

        results = run_ols_regression(df, y_name, y_label, available_X, OUTPUT_DIR)
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

    # 7. Ordered Logit for Rela_Dz (improved model for skewed distribution)
    print("\n" + "=" * 70)
    print("Step 6: Ordered Logit Regression (for Rela_Dz)")
    print("=" * 70)
    if "Rela_Dz" in df.columns:
        run_ordered_logit(df, "Rela_Dz", "Disruptiveness Index", available_X, OUTPUT_DIR)
    else:
        print("  Skipping Ordered Logit: Rela_Dz not in data")

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
