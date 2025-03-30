import pandas as pd
import matplotlib.pyplot as plt

def plot_committee_performance(res: pd.DataFrame, title="Committee Performance over Rounds"):
    """
    Given a DataFrame `res` with rounds as index and committees as columns,
    plots a line chart showing performance over rounds.
    """
    # Ensure round index is clean and readable (optional)
    if not res.index.str.startswith("Round").all():
        res.index = [f"Round {i+1}" for i in range(len(res))]

    plt.figure(figsize=(12, 6))
    for column in res.columns:
        plt.plot(res.index, res[column], marker='o', label=column)

    plt.xlabel("Rounds")
    plt.ylabel("Performance")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_committee_boxplot(res: pd.DataFrame, title="Performance Distribution per Committee"):
    """
    Given a DataFrame `res` with rounds as index and committees as columns,
    plots a boxplot showing the distribution of performance for each committee.
    """
    plt.figure(figsize=(10, 6))
    res.boxplot()
    plt.ylabel("Performance")
    plt.title(title)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
