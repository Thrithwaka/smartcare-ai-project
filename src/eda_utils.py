"""
eda_utils.py
-------------
Task 04 — Exploratory Data Analysis
Owner: Avishka

Reads  : data/processed/processed_full.csv  (Chanuu's output)
         — or data/raw/... if you want to explore BEFORE cleaning, which is
           often more informative for justifying preprocessing decisions.
Writes : reports/figures/eda/*.png

Reusable plotting/statistics functions — call these from
notebooks/02_eda.ipynb and save every figure so it can go straight into the
technical report (Task 09) without re-running notebooks.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")
EDA_FIGURES_DIR = os.path.join(config.FIGURES_DIR, "eda")


def _save_fig(fig, name: str):
    os.makedirs(EDA_FIGURES_DIR, exist_ok=True)
    path = os.path.join(EDA_FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved figure: {path}")


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Standard descriptive stats (mean, std, min, max, quartiles) for numeric columns."""
    return df.describe().T


def describe_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency counts for categorical columns — flag rare categories here."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    summary = {}
    for col in cat_cols:
        summary[col] = df[col].value_counts()
    return summary


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------
def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30):
    """Histogram for a single numeric column (e.g. Age, BMI, Blood Sugar)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df[column], bins=bins, kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    _save_fig(fig, f"histogram_{column}.png")
    plt.show()


def plot_boxplot(df: pd.DataFrame, column: str, by: str = None):
    """Boxplot for a numeric column, optionally grouped by a categorical column (e.g. risk level)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if by:
        sns.boxplot(data=df, x=by, y=column, ax=ax)
        ax.set_title(f"{column} by {by}")
    else:
        sns.boxplot(data=df, y=column, ax=ax)
        ax.set_title(f"Boxplot of {column}")
    _save_fig(fig, f"boxplot_{column}{'_by_' + by if by else ''}.png")
    plt.show()


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str = None):
    """Scatterplot between two numeric columns, optionally colored by target class."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, alpha=0.7)
    ax.set_title(f"{x} vs {y}" + (f" (by {hue})" if hue else ""))
    _save_fig(fig, f"scatter_{x}_vs_{y}.png")
    plt.show()


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap across all numeric features."""
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    _save_fig(fig, "correlation_heatmap.png")
    plt.show()
    return corr


# ---------------------------------------------------------------------------
# Class distribution (important for Option C — check for class imbalance)
# ---------------------------------------------------------------------------
def plot_class_distribution(df: pd.DataFrame, target_col: str = config.TARGET_VARIABLE):
    """
    Bar chart of class counts for disease_risk_level (Low/Medium/High).
    Flag imbalance here — it drives decisions later: class_weight in models,
    stratified splitting (already done in preprocessing.py), and choice of
    evaluation metric (macro-F1 over plain accuracy if imbalanced).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    order = config.CLASS_LABELS if set(config.CLASS_LABELS).issubset(set(df[target_col].unique())) else None
    sns.countplot(data=df, x=target_col, order=order, ax=ax)
    ax.set_title(f"Class Distribution: {target_col}")
    for container in ax.containers:
        ax.bar_label(container)
    _save_fig(fig, "class_distribution.png")
    plt.show()

    counts = df[target_col].value_counts()
    print("Class proportions:")
    print((counts / counts.sum() * 100).round(1))
    return counts
