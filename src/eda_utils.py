"""
eda_utils.py
Reusable exploratory-analysis and visualization functions.
SmartCare Hospital AI Dataset | Task 04 - EDA | Option C - Disease Risk Classification
CCS3440 Artificial Intelligence Coursework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
RISK_ORDER = ["Low", "Medium", "High"]
RISK_PALETTE = {"Low": "#4C9F70", "Medium": "#F2B134", "High": "#D64545"}


def undo_onehot(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Reconstruct a single readable categorical column from a block of
    one-hot dummy columns named '<prefix>_<category>'. Needed because Task 03
    saved the modelling-ready (already one-hot encoded) dataset -- EDA needs
    the readable category back for interpretable plots."""
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    if not cols:
        raise ValueError(f"No one-hot columns found with prefix '{prefix}_'")
    return df[cols].idxmax(axis=1).str.replace(prefix + "_", "", regex=False)


def prepare_eda_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add back readable labels: the target label, and readable versions of
    every one-hot-encoded categorical block used in this analysis."""
    df = df.copy()
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    df["disease_risk_level_label"] = df["disease_risk_level"].map(risk_map)

    for prefix in ["department", "bmi_category", "bp_category",
                    "blood_sugar_category", "age_group", "gender", "room_type"]:
        df[prefix + "_cat"] = undo_onehot(df, prefix)
    return df


def descriptive_stats(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Overall summary statistics for numeric columns."""
    return df[numeric_cols].describe().T.round(2)


def descriptive_stats_by_risk(df: pd.DataFrame, numeric_cols: list,
                               target: str = "disease_risk_level_label") -> pd.DataFrame:
    """Mean of each numeric column, grouped by risk class -- shows separation."""
    return df.groupby(target)[numeric_cols].mean().reindex(RISK_ORDER).round(2)


def plot_histograms(df, cols, target="disease_risk_level_label", ncols=3):
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols):
        sns.histplot(data=df, x=col, hue=target, hue_order=RISK_ORDER, palette=RISK_PALETTE,
                     kde=True, element="step", ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig


def plot_boxplots(df, cols, target="disease_risk_level_label", ncols=3):
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols):
        sns.boxplot(data=df, x=target, y=col, hue=target, order=RISK_ORDER, hue_order=RISK_ORDER,
                    palette=RISK_PALETTE, legend=False, ax=axes[i])
        axes[i].set_title(f"{col} by Risk Level")
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig


def plot_scatter_pairs(df, pairs, target="disease_risk_level_label", ncols=2):
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    for i, (x, y) in enumerate(pairs):
        sns.scatterplot(data=df, x=x, y=y, hue=target, hue_order=RISK_ORDER,
                         palette=RISK_PALETTE, alpha=0.6, ax=axes[i])
        axes[i].set_title(f"{x} vs {y}")
    for j in range(len(pairs), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, numeric_cols, figsize=(9, 7)):
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap - Clinical & Operational Features")
    plt.tight_layout()
    return fig


def plot_class_distribution(df, target="disease_risk_level_label"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    counts = df[target].value_counts().reindex(RISK_ORDER)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, order=RISK_ORDER,
                hue_order=RISK_ORDER, palette=RISK_PALETTE, legend=False, ax=axes[0])
    axes[0].set_title("Disease Risk Level - Counts")
    axes[0].set_ylabel("Number of Patients")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center")

    pct = counts / counts.sum() * 100
    axes[1].pie(pct.values, labels=[f"{i}\n{p:.1f}%" for i, p in zip(pct.index, pct.values)],
                colors=[RISK_PALETTE[i] for i in pct.index], startangle=90)
    axes[1].set_title("Disease Risk Level - Proportion")
    plt.tight_layout()
    return fig


def plot_categorical_vs_target(df, cat_col, target="disease_risk_level_label", figsize=None):
    ct = pd.crosstab(df[cat_col], df[target], normalize="index")[RISK_ORDER] * 100
    fig, ax = plt.subplots(figsize=figsize or (max(6, len(ct) * 0.9), 5))
    ct.plot(kind="bar", stacked=True, color=[RISK_PALETTE[c] for c in RISK_ORDER], ax=ax)
    ax.set_ylabel("% of patients")
    ax.set_title(f"{cat_col} vs Disease Risk Level (row %)")
    ax.legend(title="Risk Level", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig
