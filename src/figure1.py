import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LINE_KWARGS = {
    'color': 'purple'
}


DEFAULT_SCATTER_KWARGS = {
    'marker': 'o',
    'color': 'purple'
}


def hstripe(ax: mpl.axes.Axes,
            x: np.ndarray,
            labels: List[str] = None,
            limit: int = None,
            line_kwargs: Dict[str, Any] = {},
            scatter_kwargs: Dict[str, Any] = {}) -> None:
    """
    Plots a horizontal stripe plot in the given axis.

    Parameters
    ===
    ax : matplotlib.axes.Axes
        The axis to add the plot to.
    x : numpy.ndarray
        An array of shape (n_classes, 3) where:
            x[:, 0] is the lower bounds
            x[:, 1] is the midpoints
            x[:, 2] is the upper bounds
    labels : list
        A list containing `n_classes` labels. Default: class indices are used.
    limit : int
        Limits the number of data points displayed; the middle data points are skipped.
        Default: all data points plotted.
    line_kwargs : dict
        Keyword arguments passed to the line plot.
    scatter_kwargs : dict
        Keyword arguments passed to the scatter plot.
    """
    num_rows = x.shape[0]
    labels = labels if labels is not None else list(range(num_rows))

    # Combine default and custom kwargs
    # TODO: @rloganiv - find a clearner way to merge dictionaries
    _line_kwargs = DEFAULT_LINE_KWARGS.copy()
    _line_kwargs.update(line_kwargs)
    _scatter_kwargs = DEFAULT_SCATTER_KWARGS.copy()
    _scatter_kwargs.update(scatter_kwargs)

    # Apply limit
    sentinel = np.empty((1, 3))
    sentinel[:] = np.nan
    if limit is not None:
        x = np.concatenate((x[:limit], sentinel, x[-limit:]), axis=0)
        labels = labels[:limit] + ['...'] + labels[-limit:]
        num_rows = x.shape[0]

    # Plot
    for i, row in enumerate(x):
        low, mid, high = row.tolist()
        ax.plot((low, high), (i, i), **_line_kwargs)
        ax.plot(mid, i, **_scatter_kwargs)

    # Add labels
    ax.set_ylim(-1, num_rows)
    ax.set_yticks(np.arange(num_rows))
    ax.set_yticklabels(labels)


def figure1(accuracy: np.ndarray,
            ece: np.ndarray,
            labels: List[str] = None,
            limit: int = None,
            reverse: bool = False):
    """
    Replicates Figure 1 in [CITE PAPER].

    Parameters
    ===
    accuracy : np.ndarray
        An array of shape (n_classes, 3) where:
            x[:, 0] is the lower bounds
            x[:, 1] is the midpoints
            x[:, 2] is the upper bounds
        Contains the accuracy data plotted in the left plot.
    ece : np.ndarray
        Simlar to left, except data is plotted in the right plot.
    labels : list
        A list containing `n_classes` labels. Default: class indices are used.
    limit : int
        Limits the number of data points displayed; the middle data points are skipped.
        Default: all data points plotted.
    reverse : bool
        Whether to reverse the vertical ordering. Default: highest value to be at top.

    Returns
    ===
    fig, axes : The generated matplotlib Figure and Axes.
    """
    assert accuracy.shape == ece.shape

    # Sort the datapoints and labels
    sort_indices = np.argsort(accuracy[:, 1])
    if reverse:
        sort_indices = sort_indices[::-1]
    accuracy = accuracy[sort_indices]
    ece = ece[sort_indices]
    if labels is not None:
        labels = [labels[i] for i in sort_indices]

    # Plot
    # TODO: @rloganiv - set figsize to something reasonable
    fig, axes = plt.subplots(ncols=2, figsize=(4, 4), sharey=True)
    hstripe(axes[0], accuracy, labels=labels, limit=limit)
    axes[0].set_title('Accuracy')
    hstripe(axes[1], ece, labels=labels, limit=limit)
    axes[1].set_title('ECE')

    return fig, axes


if __name__ == '__main__':
    x = np.array([
        [3.0, 3.5, 4.0],
        [2.0, 2.5, 3.0],
        [1.0, 1.5, 2.0],
        [4.0, 4.5, 5.0]
    ])
    labels = ['apples', 'bananas', 'oranges', 'plutartos']
    fig, axes = figure1(x, -x, labels, limit=1)
    plt.show()
