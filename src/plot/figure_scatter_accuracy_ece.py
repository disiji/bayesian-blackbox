######################################CONSTANTS######################################
num_samples = 1000

DEFAULT_RC = {
    'lines.markersize': 2,
    'font.size': 8,
    'font.family': 'serif',
    'font.serif': ['Times'],
    # 'text.usetex': True,
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'legend.fontsize': 5,
    'legend.loc': 'lower right',
    'figure.titlesize': 8,
    'xtick.labelsize': 4,
    'ytick.labelsize': 4,
}

DEFAULT_PLOT_KWARGS = {
    'linewidth': 1
}

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches

######################################CONSTANTS######################################
import sys

sys.path.insert(0, '..')

from typing import Dict, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from data_utils import DATAFILE_LIST, prepare_data, NUM_CLASSES_DICT, FIGURE_DIR, DATASET_NAMES, TOPK_DICT
from models import BetaBernoulli, ClasswiseEce


def plot_scatter(ax: mpl.axes.Axes,
                 accuracy_samples: np.ndarray,
                 ece_samples: np.ndarray,
                 limit=5,
                 plot_kwargs: Dict[str, Any] = {}) -> mpl.axes.Axes:
    _plot_kwargs = DEFAULT_PLOT_KWARGS.copy()
    _plot_kwargs.update(plot_kwargs)
    # plot
    x = np.mean(accuracy_samples, axis=1)
    y = np.mean(ece_samples, axis=1)
    xerr = np.std(accuracy_samples, axis=1)
    yerr = np.std(ece_samples, axis=1)

    # most accuracy top k
    idx = x.argsort()[-limit:][::-1]
    ax.errorbar(x[idx], y[idx],
                xerr=xerr[idx],
                yerr=yerr[idx],
                fmt='o', alpha=0.8, color='b', **_plot_kwargs)

    # least accuracy top k
    idx = x.argsort()[:limit]
    ax.errorbar(x[idx], y[idx],
                xerr=xerr[idx],
                yerr=yerr[idx],
                fmt='o', alpha=0.8, color='r', **_plot_kwargs)

    # other predicted classes
    idx = x.argsort()[limit:-limit]
    ax.errorbar(x[idx], y[idx],
                xerr=xerr[idx],
                yerr=yerr[idx],
                fmt='o', alpha=0.2, color='k', **_plot_kwargs)

    ax.set_xlabel('Accuracy')
    # ax.set_ylabel('ECE')

    return ax


def main() -> None:
    with mpl.rc_context(rc=DEFAULT_RC):
        fig, axes = plt.subplots(ncols=3, nrows=2, dpi=300, sharey=False)
        idx = 0
        for dataset in DATASET_NAMES:
            datafile = DATAFILE_LIST[dataset]
            num_classes = NUM_CLASSES_DICT[dataset]

            categories, observations, confidences, idx2category, category2idx, labels = prepare_data(datafile, False)

            # accuracy models
            accuracy_model = BetaBernoulli(k=num_classes, prior=None)
            accuracy_model.update_batch(categories, observations)

            # ece models for each class
            ece_model = ClasswiseEce(num_classes, num_bins=10, pseudocount=2)
            ece_model.update_batch(categories, observations, confidences)

            # draw samples from posterior of classwise accuracy
            accuracy_samples = accuracy_model.sample(num_samples)  # (num_categories, num_samples)
            ece_samples = ece_model.sample(num_samples)  # (num_categories, num_samples)

            plot_kwargs = {}
            axes[idx // 3, idx % 3] = plot_scatter(axes[idx // 3, idx % 3], accuracy_samples, ece_samples,
                                                   limit=TOPK_DICT[dataset], plot_kwargs=plot_kwargs)
            axes[idx // 3, idx % 3].set_title(DATASET_NAMES[dataset])
            idx += 1

    axes[0, 0].set_ylabel('ECE')
    axes[1, 0].set_ylabel('ECE')
    fig.set_size_inches(TEXT_WIDTH, 4.0)
    fig.subplots_adjust(bottom=0.05, wspace=0.2)
    fig.delaxes(axes.flatten()[5])
    figname = FIGURE_DIR + 'scatter.pdf'
    fig.tight_layout()
    fig.savefig(figname, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
