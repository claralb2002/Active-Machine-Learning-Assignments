import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Apply a clean seaborn style
sns.set_theme(style="whitegrid", context="talk", palette="muted")

def plot_committee_performance(res: pd.DataFrame, title="Committee Performance over Rounds"):
    """
    Given a DataFrame `res` with rounds as index and committees as columns,
    plots a line chart showing performance over rounds using seaborn.
    """
    res_plot = res.copy()
    res_plot["Round"] = range(1, len(res_plot) + 1)
    res_melted = res_plot.melt(id_vars="Round", var_name="Committee", value_name="Performance")

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=res_melted, x="Round", y="Performance", hue="Committee", marker="o")

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Performance")
    plt.legend(title="Committee", loc="best")
    plt.tight_layout()
    plt.show()

def plot_committee_boxplot(res: pd.DataFrame, title="Performance Distribution per Committee"):
    """
    Given a DataFrame `res` with rounds as index and committees as columns,
    plots a boxplot showing the distribution of performance for each committee using seaborn.
    """
    res_melted = res.reset_index().melt(id_vars="index", var_name="Committee", value_name="Performance")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=res_melted, x="Committee", y="Performance")

    plt.title(title)
    plt.xlabel("Committee")
    plt.ylabel("Performance")
    plt.tight_layout()
    plt.show()
