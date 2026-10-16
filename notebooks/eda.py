"""
Exploratory Data Analysis (EDA) Script
Generates publication-quality charts for travel time distributions,
temporal patterns, route features, and correlation heatmaps.
Saves all figures to reports/figures/ for documentation and UI presentation.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data_preprocessing import load_dataset, clean_and_prepare_dataset

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "figures"))


def run_eda(save_figures: bool = True):
    """Executes complete EDA suite and creates all required visual plots."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    raw_df = load_dataset()
    df = clean_and_prepare_dataset(raw_df)

    # Set aesthetic style
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("muted")

    # 1. Journey-time distribution
    fig1, ax1 = plt.subplots(figsize=(8, 4.5), dpi=150)
    sns.histplot(df["TotalTime"], kde=True, bins=20, color="#2563EB", ax=ax1, edgecolor="white")
    ax1.axvline(df["TotalTime"].median(), color="#DC2626", linestyle="--", linewidth=1.5,
                label=f"Median: {df['TotalTime'].median():.1f} min")
    ax1.axvline(df["TotalTime"].mean(), color="#059669", linestyle="-.", linewidth=1.5,
                label=f"Mean: {df['TotalTime'].mean():.1f} min")
    ax1.set_title("Distribution of Journey Duration (TotalTime)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Journey Duration (minutes)", fontsize=10)
    ax1.set_ylabel("Trip Frequency", fontsize=10)
    ax1.legend(frameon=True, facecolor="#F8FAFC")
    plt.tight_layout()
    if save_figures:
        fig1.savefig(os.path.join(FIGURES_DIR, "eda_totaltime_dist.png"))

    # 2. Trip counts and duration by destination
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    # Counts
    dest_counts = df["GoingTo"].value_counts()
    ax2a.bar(dest_counts.index, dest_counts.values, color=["#0284C7", "#0D9488"], width=0.5, edgecolor="white")
    ax2a.set_title("Trip Counts by Destination", fontsize=11, fontweight="bold")
    ax2a.set_ylabel("Number of Trips", fontsize=10)
    for i, v in enumerate(dest_counts.values):
        ax2a.text(i, v + 1, str(v), ha="center", fontweight="bold")

    # Duration Boxplot
    sns.boxplot(x="GoingTo", y="TotalTime", data=df, ax=ax2b, hue="GoingTo", palette=["#0284C7", "#0D9488"], legend=False)
    ax2b.set_title("Travel Time by Destination", fontsize=11, fontweight="bold")
    ax2b.set_ylabel("TotalTime (min)", fontsize=10)
    ax2b.set_xlabel("Destination", fontsize=10)
    plt.tight_layout()
    if save_figures:
        fig2.savefig(os.path.join(FIGURES_DIR, "eda_destination_comparison.png"))

    # 3. Journey duration by departure hour
    fig3, ax3 = plt.subplots(figsize=(9, 4.5), dpi=150)
    sns.boxplot(x="StartHour", y="TotalTime", data=df, ax=ax3, color="#8B5CF6")
    ax3.set_title("Journey Duration by Departure Hour", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Departure Hour (24-hr format)", fontsize=10)
    ax3.set_ylabel("Journey Duration (minutes)", fontsize=10)
    # Add peak hours annotation
    ax3.text(1.5, 75, "Morning Rush (GSK)", color="#1E40AF", fontweight="bold", fontsize=9)
    ax3.text(6.5, 75, "Evening Rush (Home)", color="#B45309", fontweight="bold", fontsize=9)
    plt.tight_layout()
    if save_figures:
        fig3.savefig(os.path.join(FIGURES_DIR, "eda_hour_vs_time.png"))

    # 4. Journey duration by day of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    fig4, ax4 = plt.subplots(figsize=(9, 4.5), dpi=150)
    sns.boxplot(x="DayOfWeek", y="TotalTime", data=df, order=day_order, ax=ax4, hue="DayOfWeek", palette="Set2", legend=False)
    ax4.set_title("Journey Duration Across Days of the Week", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xlabel("Day of Week", fontsize=10)
    ax4.set_ylabel("Journey Duration (minutes)", fontsize=10)
    plt.tight_layout()
    if save_figures:
        fig4.savefig(os.path.join(FIGURES_DIR, "eda_dayofweek_vs_time.png"))

    # 5. Distance versus journey duration
    fig5, ax5 = plt.subplots(figsize=(8, 4.5), dpi=150)
    sns.scatterplot(
        x="Distance",
        y="TotalTime",
        hue="Take407All",
        style="GoingTo",
        data=df,
        palette={"No": "#EF4444", "Yes": "#10B981"},
        s=80,
        alpha=0.8,
        ax=ax5,
    )
    ax5.set_title("Distance vs. Journey Duration (Toll vs. Free Route)", fontsize=12, fontweight="bold", pad=10)
    ax5.set_xlabel("Trip Distance (km)", fontsize=10)
    ax5.set_ylabel("Journey Duration (minutes)", fontsize=10)
    ax5.legend(title="Took 407 Toll?", frameon=True, facecolor="#F8FAFC")
    plt.tight_layout()
    if save_figures:
        fig5.savefig(os.path.join(FIGURES_DIR, "eda_distance_vs_time.png"))

    # 6. Correlation heatmap of numeric predictors
    fig6, ax6 = plt.subplots(figsize=(7, 5.5), dpi=150)
    df_corr = df.copy()
    df_corr["IsToll"] = (df_corr["Take407All"] == "Yes").astype(int)
    df_corr["IsHome"] = (df_corr["GoingTo"] == "Home").astype(int)
    numeric_corr_cols = [
        "TotalTime",
        "Distance",
        "DepartureDecimal",
        "StartHour",
        "IsToll",
        "IsHome",
        "Month",
    ]
    corr_matrix = df_corr[numeric_corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax6,
    )
    ax6.set_title("Feature Correlation Heatmap (Pre-departure Features)", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    if save_figures:
        fig6.savefig(os.path.join(FIGURES_DIR, "eda_correlation_heatmap.png"))

    print(f"Successfully generated all 6 EDA figures in {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda(save_figures=True)
