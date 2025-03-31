import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", context="talk", palette="muted")


def plot_committee_performance(res: pd.DataFrame, baseline: pd.DataFrame, title="Committee Performance over Rounds"):
    """
    Shows two subplots:
    - Top: Raw performance over rounds
    - Bottom: Regression trendlines
    Includes slope summary legend box.
    """

    # --- Data Preparation ---
    res_plot = res.copy()
    baseline_plot = baseline.copy()

    res_plot["Round"] = range(1, len(res_plot) + 1)
    res_melted = res_plot.melt(id_vars="Round", var_name="Committee", value_name="Performance")

    baseline_plot["Round"] = range(1, len(baseline_plot) + 1)

    # --- Set up figure ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

    # === Plot 1: Raw Performance ===
    sns.lineplot(ax=axes[0], data=res_melted, x="Round", y="Performance", hue="Committee", marker="o")
    axes[0].plot(
        baseline_plot["Round"],
        baseline_plot["baseline"],
        label="Baseline",
        color="black",
        linestyle="--",
        linewidth=2.5,
        marker="x",
        markersize=6
    )
    axes[0].set_title(f"{title} — Raw")
    axes[0].set_ylabel("Performance")
    axes[0].legend(title="Committee")

    # === Plot 2: Regression Trends ===
    x = res_plot["Round"]
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    slopes = {}

    for i, committee in enumerate(res_plot.columns.drop("Round")):
        y = res_plot[committee]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        slope = z[0]
        slopes[committee] = slope
        axes[1].plot(x, p(x), label=committee, color=color_cycle[i], linewidth=2)

    # --- Baseline regression ---
    y_base = baseline_plot["baseline"]
    z_base = np.polyfit(x, y_base, 1)
    p_base = np.poly1d(z_base)
    slope_base = z_base[0]
    slopes["Baseline"] = slope_base

    axes[1].plot(x, p_base(x), label="Baseline", color="black", linestyle="--", linewidth=2.5)

    # --- Average Committee regression ---
    committee_columns = res_plot.columns.drop("Round")
    y_avg = res_plot[committee_columns].mean(axis=1)
    z_avg = np.polyfit(x, y_avg, 1)
    p_avg = np.poly1d(z_avg)
    slope_avg = z_avg[0]
    slopes["Avg. Committee"] = slope_avg

    axes[1].plot(x, p_avg(x), label="Avg. Committee", color="gray", linestyle="--", linewidth=2.5)

    axes[1].set_title(f"{title} — Linear Trend")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Performance")

    # --- Build slope legend box ---
    slope_text = "Slope Coefficients:\n" + "\n".join(
        f"{name: <15} {slope:.2f}" for name, slope in slopes.items()
    )

    # Place text box in the figure, not the subplot
    # fig.text(0.5, 0.03, slope_text, ha='center', fontsize=11, family='monospace', bbox=dict(facecolor='white', edgecolor='gray'))

    axes[1].text(
    0.98, 0.02,  # Near bottom-right corner
    slope_text,
    transform=axes[1].transAxes,
    ha='right', va='bottom',
    fontsize=10, family='monospace',
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5')
    )


    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for slope text
    plt.savefig('results_lineplot.png')
    plt.show()

def plot_committee_boxplot(res: pd.DataFrame, baseline: pd.DataFrame, title="Performance Distribution per Committee"):
    """
    Shows boxplots for each committee including baseline as a full distribution,
    and adds a horizontal line for the baseline average.
    """
    # Prepare committee data
    res_melted = res.reset_index().melt(id_vars="index", var_name="Committee", value_name="Performance")
    res_melted['Type'] = 'Committee'

    # Prepare baseline data separately
    baseline_melted = baseline.reset_index().melt(id_vars="index", var_name="Committee", value_name="Performance")
    baseline_melted["Type"] = 'Baseline'

    # Combine data
    combined = pd.concat([res_melted, baseline_melted])

    plt.figure(figsize=(12, 6))
    
    # Boxplot with clear separation and color
    sns.boxplot(data=combined, x="Committee", y="Performance", hue="Type",
                palette={"Committee": "lightsteelblue", "Baseline": "black"}, dodge=False)
    
    # Add vertical separator line before baseline
    baseline_pos = len(res.columns) - 0.5
    plt.axvline(x=baseline_pos, color="gray", linestyle='-', linewidth=2)

    plt.title(title)
    plt.xlabel("Committees / Baseline")
    plt.ylabel("Performance in Pct.")

    plt.legend().remove()
    
    plt.tight_layout()
    plt.savefig('results_boxplot.png')
    plt.show()
