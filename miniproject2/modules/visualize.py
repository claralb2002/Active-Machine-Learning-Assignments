import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", context="talk", palette="muted")


def plot_committee_performance(res: pd.DataFrame, baseline: pd.DataFrame, title="Committee Performance over Rounds", normalize=False):
    """
    Shows two subplots:
    - Top: Raw performance over rounds
    - Bottom: Regression trendlines with slope annotations
    If `normalize=True`, subtracts initial value to show relative improvement.
    """

    # --- Data Preparation ---
    res_plot = res.copy()
    baseline_plot = baseline.copy()

    if normalize:
        res_plot = res_plot.subtract(res_plot.iloc[0])  # Normalize each line to start at 0
        baseline_plot["baseline"] = baseline_plot["baseline"] - baseline_plot["baseline"].iloc[0]
        title += " (Normalized)"

    res_plot["Round"] = range(1, len(res_plot) + 1)
    res_melted = res_plot.melt(id_vars="Round", var_name="Committee", value_name="Performance")

    baseline_plot["Round"] = range(1, len(baseline_plot) + 1)

    # --- Set up figure ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

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

    for i, committee in enumerate(res_plot.columns.drop("Round")):
        y = res_plot[committee]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        slope = z[0]
        axes[1].plot(x, p(x), label=committee, color=color_cycle[i], linewidth=2)

        # Annotate slope
        axes[1].text(
            x.iloc[-1] + 0.5, p(x.iloc[-1]),
            f"↑ {slope:.2f}", fontsize=10,
            color=color_cycle[i], va="center"
        )

    # Baseline regression
    y_base = baseline_plot["baseline"]
    z_base = np.polyfit(x, y_base, 1)
    p_base = np.poly1d(z_base)
    slope_base = z_base[0]

    axes[1].plot(x, p_base(x), label="Baseline", color="black", linestyle="--", linewidth=2.5)
    axes[1].text(x.iloc[-1] + 0.5, p_base(x.iloc[-1]), f"↑ {slope_base:.2f}", fontsize=10, color="black", va="center")

    axes[1].set_title(f"{title} — Linear Trend")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Performance")
    axes[1].legend(title="Committee")
    plt.tight_layout()
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
