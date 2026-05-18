"""
Plot Rela_Dz (Disruptiveness Index) results for paper

Generates:
  1. Predicted probability plot (num_references vs disruptiveness categories)
  2. Coefficient forest plot (all predictors with 95% CI)
  3. Category distribution bar chart
  4. Combined summary figure for paper
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = "ai4s_metrics/results/ai4s_metrics_full_20260517_182337.csv"
OUTPUT_DIR = "ai4s_metrics/results/regression_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variable labels for plotting
VAR_LABELS = {
    "discipline_variety": "Discipline Variety",
    "discipline_similarity": "Discipline Similarity",
    "discipline_balance": "Discipline Balance",
    "ai4s_balance": "AI4S Balance",
    "ai_ref_age": "AI Ref. Age",
    "num_authors": "Num. Authors",
    "publication_year": "Pub. Year",
    "num_references": "Num. References",
    "num_institutions": "Num. Institutions",
    "has_international_collab": "Intl. Collaboration",
    "open_access": "Open Access",
}

PREDICTORS = list(VAR_LABELS.keys())

# Colors
COLORS = {
    "Low": "#4C72B0",
    "Medium": "#F0A030",
    "High": "#C44E52",
}
CATEGORY_ORDER = ["Low", "Medium", "High"]


# ============================================================
# Data Preparation
# ============================================================

def load_and_prepare():
    """Load data and fit Ordered Logit model"""
    df = pd.read_csv(INPUT_FILE)
    print(f"Data loaded: {df.shape[0]} rows")

    # Prepare variables
    X = df[PREDICTORS].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Discretize Rela_Dz
    bins = [-np.inf, 0.05, 0.2, np.inf]
    labels = [0, 1, 2]
    y_ordinal = pd.cut(df["Rela_Dz"], bins=bins, labels=labels).astype(int)

    # Drop NaN
    valid = ~(X.isnull().any(axis=1) | y_ordinal.isna())
    X = X[valid]
    y_ordinal = y_ordinal[valid]
    print(f"Valid observations: {len(X)}")

    # Fit model
    model = OrderedModel(y_ordinal, X, distr="logit")
    results = model.fit(method="bfgs", maxiter=1000, disp=False)
    print(f"Model fitted. Pseudo R² = {results.prsquared:.4f}")

    return df, X, y_ordinal, results


# ============================================================
# Plot 1: Predicted Probabilities (num_references)
# ============================================================

def plot_predicted_probabilities(results, X, save_path):
    """
    Plot predicted probabilities for each disruptiveness category
    as a function of num_references (the only significant predictor)
    """
    # Create a range of num_references values
    ref_range = np.linspace(0, 500, 100)

    # Create prediction data (hold other variables at mean)
    X_mean = X.mean().to_dict()
    pred_data = pd.DataFrame([X_mean.copy() for _ in range(len(ref_range))])
    pred_data["num_references"] = ref_range

    # Predict probabilities
    probs = results.predict(pred_data)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, cat in enumerate(CATEGORY_ORDER):
        ax.plot(ref_range, probs.iloc[:, i],
                color=COLORS[cat], linewidth=2.5,
                label=cat, alpha=0.9)

    ax.set_xlabel("Number of References", fontsize=13)
    ax.set_ylabel("Predicted Probability", fontsize=13)
    ax.set_title("Predicted Disruptiveness Probability\nby Number of References",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, loc="center right")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    # Add annotation
    ax.text(0.98, 0.95,
            "Other predictors held at mean",
            transform=ax.transAxes, fontsize=9,
            ha="right", va="top", style="italic",
            color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Predicted probabilities plot saved: {save_path}")


# ============================================================
# Plot 2: Coefficient Forest Plot
# ============================================================

def plot_coefficient_forest(results, save_path):
    """
    Plot coefficient forest plot with 95% CI
    Suitable for academic papers
    """
    params = results.params
    conf = results.conf_int()

    # Get only predictor coefficients (exclude thresholds)
    var_names = [v for v in params.index if v not in ["0/1", "1/2"]]
    coefs = [params[v] for v in var_names]
    ci_lower = [conf.loc[v, 0] for v in var_names]
    ci_upper = [conf.loc[v, 1] for v in var_names]
    p_vals = [results.pvalues[v] for v in var_names]
    labels = [VAR_LABELS.get(v, v) for v in var_names]

    # Sort by coefficient value
    sorted_idx = np.argsort(coefs)
    coefs_sorted = [coefs[i] for i in sorted_idx]
    ci_lower_sorted = [ci_lower[i] for i in sorted_idx]
    ci_upper_sorted = [ci_upper[i] for i in sorted_idx]
    p_vals_sorted = [p_vals[i] for i in sorted_idx]
    labels_sorted = [labels[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = range(len(var_names))
    colors_bar = []
    for p in p_vals_sorted:
        if p < 0.001:
            colors_bar.append("#C44E52")
        elif p < 0.01:
            colors_bar.append("#E8856A")
        elif p < 0.05:
            colors_bar.append("#F0A030")
        else:
            colors_bar.append("#B0B0B0")

    bars = ax.barh(y_pos, coefs_sorted, color=colors_bar,
                   edgecolor="black", linewidth=0.5, height=0.6)
    ax.errorbar(coefs_sorted, y_pos,
                xerr=[[c - l for c, l in zip(coefs_sorted, ci_lower_sorted)],
                      [u - c for c, u in zip(coefs_sorted, ci_upper_sorted)]],
                fmt="none", ecolor="black", capsize=3, linewidth=1)

    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels_sorted, fontsize=11)
    ax.set_xlabel("Coefficient (log-odds) with 95% CI", fontsize=12)
    ax.set_title("Ordered Logit: Predictors of Disruptiveness (Rela_Dz)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Add significance stars
    for i, (p, coef) in enumerate(zip(p_vals_sorted, coefs_sorted)):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = ""
        if star:
            x_pos = coef + 0.3 if coef >= 0 else coef - 0.3
            ha = "left" if coef >= 0 else "right"
            ax.text(x_pos, i, star, va="center", ha=ha, fontsize=14,
                    fontweight="bold", color="#C44E52")

    # Add legend for significance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C44E52", label='p < 0.001 ***'),
        Patch(facecolor="#E8856A", label='p < 0.01 **'),
        Patch(facecolor="#F0A030", label='p < 0.05 *'),
        Patch(facecolor="#B0B0B0", label='n.s.'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Coefficient forest plot saved: {save_path}")


# ============================================================
# Plot 3: Category Distribution
# ============================================================

def plot_category_distribution(y_ordinal, save_path):
    """Plot distribution of disruptiveness categories"""
    counts = y_ordinal.value_counts().sort_index()
    labels = CATEGORY_ORDER
    values = [counts.get(i, 0) for i in range(3)]
    percentages = [v / sum(values) * 100 for v in values]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(labels, values, color=[COLORS[c] for c in labels],
                  edgecolor="black", linewidth=0.8, width=0.5)

    # Add count and percentage labels
    for bar, val, pct in zip(bars, values, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"n = {val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Number of Papers", fontsize=13)
    ax.set_title("Distribution of Disruptiveness Categories\n(Rela_Dz)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Category distribution plot saved: {save_path}")


# ============================================================
# Plot 4: Confusion Matrix (improved)
# ============================================================

def plot_confusion_matrix(results, X, y_actual, save_path):
    """Plot improved confusion matrix with percentages"""
    y_pred = results.predict(X).idxmax(axis=1)

    # Confusion matrix with counts and percentages
    cm = pd.crosstab(y_actual, y_pred,
                     rownames=["Actual"], colnames=["Predicted"])

    # Normalize to percentages
    cm_pct = cm.div(cm.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot heatmap with both count and percentage
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            count = cm.iloc[i, j]
            pct = cm_pct.iloc[i, j]
            if count > 0:
                text = f"{int(count)}\n({pct:.1f}%)"
            else:
                text = "0"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if pct > 50 else "black")

    # Draw heatmap
    im = ax.imshow(cm.values, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(CATEGORY_ORDER, fontsize=11)
    ax.set_yticklabels(CATEGORY_ORDER, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix\n(Accuracy: {((y_actual == y_pred).mean() * 100):.1f}%)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


# ============================================================
# Plot 5: Summary Figure for Paper
# ============================================================

def plot_summary_figure(results, X, y_ordinal, save_path):
    """
    Create a combined 2x2 figure for paper
    Top-left: Category distribution
    Top-right: Coefficient forest plot
    Bottom-left: Predicted probabilities
    Bottom-right: Confusion matrix
    """
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(16, 12))

    # === Panel A: Category Distribution ===
    ax1 = fig.add_subplot(2, 2, 1)
    counts = y_ordinal.value_counts().sort_index()
    values = [counts.get(i, 0) for i in range(3)]
    percentages = [v / sum(values) * 100 for v in values]

    bars = ax1.bar(CATEGORY_ORDER, values, color=[COLORS[c] for c in CATEGORY_ORDER],
                   edgecolor="black", linewidth=0.8, width=0.5)
    for bar, val, pct in zip(bars, values, percentages):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"n = {val}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Number of Papers", fontsize=11)
    ax1.set_title("(a) Disruptiveness Category Distribution", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(values) * 1.3)
    ax1.grid(axis="y", alpha=0.3)

    # === Panel B: Coefficient Forest Plot ===
    ax2 = fig.add_subplot(2, 2, 2)
    params = results.params
    conf = results.conf_int()
    var_names = [v for v in params.index if v not in ["0/1", "1/2"]]
    coefs = [params[v] for v in var_names]
    ci_lower = [conf.loc[v, 0] for v in var_names]
    ci_upper = [conf.loc[v, 1] for v in var_names]
    p_vals = [results.pvalues[v] for v in var_names]
    labels = [VAR_LABELS.get(v, v) for v in var_names]

    sorted_idx = np.argsort(coefs)
    coefs_sorted = [coefs[i] for i in sorted_idx]
    ci_lower_sorted = [ci_lower[i] for i in sorted_idx]
    ci_upper_sorted = [ci_upper[i] for i in sorted_idx]
    p_vals_sorted = [p_vals[i] for i in sorted_idx]
    labels_sorted = [labels[i] for i in sorted_idx]

    y_pos = range(len(var_names))
    colors_bar = []
    for p in p_vals_sorted:
        if p < 0.001:
            colors_bar.append("#C44E52")
        elif p < 0.01:
            colors_bar.append("#E8856A")
        elif p < 0.05:
            colors_bar.append("#F0A030")
        else:
            colors_bar.append("#B0B0B0")

    ax2.barh(y_pos, coefs_sorted, color=colors_bar,
             edgecolor="black", linewidth=0.5, height=0.6)
    ax2.errorbar(coefs_sorted, y_pos,
                 xerr=[[c - l for c, l in zip(coefs_sorted, ci_lower_sorted)],
                       [u - c for c, u in zip(coefs_sorted, ci_upper_sorted)]],
                 fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(labels_sorted, fontsize=9)
    ax2.set_xlabel("Coefficient (log-odds) with 95% CI", fontsize=11)
    ax2.set_title("(b) Predictors of Disruptiveness", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # Add significance stars
    for i, (p, coef) in enumerate(zip(p_vals_sorted, coefs_sorted)):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = ""
        if star:
            x_pos = coef + 0.3 if coef >= 0 else coef - 0.3
            ha = "left" if coef >= 0 else "right"
            ax2.text(x_pos, i, star, va="center", ha=ha, fontsize=12,
                     fontweight="bold", color="#C44E52")

    legend_elements = [
        Patch(facecolor="#C44E52", label='p < 0.001'),
        Patch(facecolor="#E8856A", label='p < 0.01'),
        Patch(facecolor="#F0A030", label='p < 0.05'),
        Patch(facecolor="#B0B0B0", label='n.s.'),
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc="lower right")

    # === Panel C: Predicted Probabilities ===
    ax3 = fig.add_subplot(2, 2, 3)
    ref_range = np.linspace(0, 500, 100)
    X_mean = X.mean().to_dict()
    pred_data = pd.DataFrame([X_mean.copy() for _ in range(len(ref_range))])
    pred_data["num_references"] = ref_range
    probs = results.predict(pred_data)

    for i, cat in enumerate(CATEGORY_ORDER):
        ax3.plot(ref_range, probs.iloc[:, i],
                 color=COLORS[cat], linewidth=2.5, label=cat, alpha=0.9)
    ax3.set_xlabel("Number of References", fontsize=11)
    ax3.set_ylabel("Predicted Probability", fontsize=11)
    ax3.set_title("(c) Predicted Probability by References", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10, loc="center right")
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(alpha=0.3)

    # === Panel D: Confusion Matrix ===
    ax4 = fig.add_subplot(2, 2, 4)
    y_pred = results.predict(X).idxmax(axis=1)
    cm = pd.crosstab(y_ordinal, y_pred)
    cm_pct = cm.div(cm.sum(axis=1), axis=0) * 100

    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            count = cm.iloc[i, j]
            pct = cm_pct.iloc[i, j]
            text = f"{int(count)}\n({pct:.1f}%)" if count > 0 else "0"
            ax4.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                     fontsize=10, fontweight="bold",
                     color="white" if pct > 50 else "black")

    ax4.imshow(cm.values, cmap="Blues", aspect="auto")
    ax4.set_xticks(range(len(cm.columns)))
    ax4.set_yticks(range(len(cm.index)))
    ax4.set_xticklabels(CATEGORY_ORDER, fontsize=10)
    ax4.set_yticklabels(CATEGORY_ORDER, fontsize=10)
    ax4.set_xlabel("Predicted", fontsize=11)
    ax4.set_ylabel("Actual", fontsize=11)
    accuracy = (y_ordinal == y_pred).mean() * 100
    ax4.set_title(f"(d) Confusion Matrix (Acc: {accuracy:.1f}%)",
                  fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Summary figure saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Rela_Dz Results Visualization for Paper")
    print("=" * 60)

    # Load data and fit model
    df, X, y_ordinal, results = load_and_prepare()

    # Print model summary
    print(f"\nModel Fit Summary:")
    print(f"  Pseudo R² (McFadden): {results.prsquared:.4f}")
    print(f"  Log-Likelihood: {results.llf:.2f}")
    print(f"  LL-Null: {results.llnull:.2f}")
    print(f"  AIC: {results.aic:.2f}")
    print(f"  BIC: {results.bic:.2f}")
    print(f"  Prediction Accuracy: {(y_ordinal == results.predict(X).idxmax(axis=1)).mean() * 100:.1f}%")

    # Significant predictors
    print(f"\nSignificant Predictors (p < 0.05):")
    for var in PREDICTORS:
        pval = results.pvalues.get(var, 1.0)
        coef = results.params.get(var, 0)
        if pval < 0.05:
            direction = "increases" if coef > 0 else "decreases"
            print(f"  {var}: coef={coef:.4f}, p={pval:.4f} — {direction} disruptiveness")
    print(f"  (Only num_references is significant)")

    # Generate plots
    print(f"\nGenerating plots...")

    # 1. Predicted probabilities
    plot_predicted_probabilities(
        results, X,
        os.path.join(OUTPUT_DIR, "rela_dz_predicted_probs.png")
    )

    # 2. Coefficient forest plot
    plot_coefficient_forest(
        results,
        os.path.join(OUTPUT_DIR, "rela_dz_coefficient_forest.png")
    )

    # 3. Category distribution
    plot_category_distribution(
        y_ordinal,
        os.path.join(OUTPUT_DIR, "rela_dz_category_distribution.png")
    )

    # 4. Confusion matrix (improved)
    plot_confusion_matrix(
        results, X, y_ordinal,
        os.path.join(OUTPUT_DIR, "rela_dz_confusion_matrix.png")
    )

    # 5. Summary figure (2x2 for paper)
    plot_summary_figure(
        results, X, y_ordinal,
        os.path.join(OUTPUT_DIR, "rela_dz_summary_figure.png")
    )

    print(f"\n{'=' * 60}")
    print("All plots generated!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
