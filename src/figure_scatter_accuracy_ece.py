import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from data_utils import datafile_dict, datasize_dict, num_classes_dict, prepare_data, num_classes_dict
from data_utils import datasize_dict
from models import BetaBernoulli, ClasswiseEce
from typing import Dict, Any

num_samples = 1000

TOPK_DICT = {'cifar100': 10,
             'imagenet': 10,
             'svhn': 3,
             '20newsgroup': 3,
             'dbpedia': 3}
DATASET_NAMES = {
    'cifar100': 'CIFAR-100',
    'imagenet': 'ImageNet',
    'svhn': 'SVHN',
    '20newsgroup': '20 Newsgroups',
    'dbpedia': 'DBpedia',
}
COLOR = {'non-active_no_prior': 'non-active',
         'non-active_uniform': 'non-active_uniform',
         'non-active_informed': 'non-active_informed',
         'ts_uniform': 'ts(uniform)',
         'ts_informed': 'ts(informative)'
         }
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
    'figure.titlesize': 6,
    'xtick.labelsize': 4,
    'ytick.labelsize': 4,
}

DEFAULT_PLOT_KWARGS = {
    'linewidth': 1
}

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875


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
        fig, axes = plt.subplots(ncols=len(TOPK_DICT), dpi=300, sharey=False)
        idx = 0
        for dataset in DATASET_NAMES:
            datafile = datafile_dict[dataset]
            num_classes = num_classes_dict[dataset]

            categories, observations, confidences, idx2category, category2idx, labels = prepare_data(datafile, False)

            # accuracy models
            accuracy_model = BetaBernoulli(k=num_classes, prior=None)
            accuracy_model.update_batch(categories, observations)

            # ece models for each class
            ece_model = ClasswiseEce(num_classes, num_bins=10, pseudocount=2)
            ece_model.update_batch(categories, observations, confidences)

            # draw samples from posterior of classwise accuracy
            accuracy_samples = accuracy_model.sample(num_samples)  # (num_categories, num_samples)
            ece_samples = ece_model.sample(num_samples) # (num_categories, num_samples)

            plot_kwargs = {}
            axes[idx] = plot_scatter(axes[idx], accuracy_samples, ece_samples, limit=TOPK_DICT[dataset],
                                     plot_kwargs=plot_kwargs)
            axes[idx].set_title(DATASET_NAMES[dataset])
            idx += 1

    axes[0].set_ylabel('ECE')
    fig.set_size_inches(TEXT_WIDTH, 1.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.05, wspace=0.25)
    figname = '../figures/scatter.pdf'
    fig.savefig(figname, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
