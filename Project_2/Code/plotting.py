
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

label_size = 20

## Regular plotting
def plot(x_list, y_list, labels, title, xlabel, ylabel, filename):
    # Turn x, y and labels into lists
    if type(x_list) == np.ndarray:
        x_list = [x_list]
    if type(y_list) == np.ndarray:
        y_list = [y_list]
    if type(labels) == str:
        labels = [labels]

    plt.figure()
    plt.title(title, fontsize=label_size)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.xticks(fontsize=label_size)

    for x, y, label in zip(x_list, y_list, labels):
        plt.plot(x, y, label=label)

    plt.legend(fontsize=label_size)
    plt.savefig(filename, bbox_inches='tight')

## Heatmap
def heatmap(data, xticks, yticks, title, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(data, xticklabels=xticks, yticklabels=yticks, annot=True, ax=ax, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.yticks(fontsize=label_size)
    plt.xticks(fontsize=label_size)
    plt.savefig(filename, bbox_inches='tight')

## Søyleplot
def barplot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.title(title, fontsize=label_size)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.yticks(fontsize=label_size)
    if type(x[0]) == str:
        plt.xticks(fontsize=label_size, rotation=90)
    plt.bar(x, y)
    plt.savefig(filename, bbox_inches='tight')
