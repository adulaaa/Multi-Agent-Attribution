import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def bar_plot_attributions(method_names, scores, ylabel="Attribution Score", title=None):
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(method_names))
    ax.bar(x, scores)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig

def heatmap(matrix, labels, title="Influence Heatmap", cmap="coolwarm"):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(matrix, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels,
                cmap=cmap, vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig
