import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pandas import Series


def scatter_plot_2d(
    x_data: Series | NDArray,
    y_data: Series | NDArray,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    _, ax = plt.subplots()

    ax.scatter(x_data, y_data, c="blue", zorder=2)
    ax.grid(True, alpha=0.7, zorder=1)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def bar_binary_plot(data: Series, title: str, label: str) -> None:
    _, ax = plt.subplots()

    counts = data.value_counts()

    ax.bar(counts.index, height=data.max())
    
    plt.xticks(counts.index)
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel("frequency")
    plt.show()


if __name__ == "__main__":
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    scatter_plot_2d(x, y, "sine wave", "x", "y")
