######################################CONSTANTS######################################
DEFAULT_RC = {
    'lines.markersize': 2,
    'font.size': 6,
    'font.family': 'serif',
    'font.serif': ['Times'],
    # 'text.usetex': True,
    'axes.titlesize': 6,
    'axes.labelsize': 5,
    'legend.fontsize': 5,
    'legend.loc': 'lower right',
    'figure.titlesize': 6,
    'xtick.labelsize': 4,
    'ytick.labelsize': 4,
}
DEFAULT_PLOT_KWARGS = {
    'linewidth': 1,
    'linestyle': "--",
}
num_bins = 10
TEXT_WIDTH = 6.299213  # Inches

######################################CONSTANTS######################################
import sys

sys.path.insert(0, '..')

import argparse
import random
from typing import Dict, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from data_utils import DATAFILE_LIST, DATASIZE_DICT, prepare_data, FIGURE_DIR, DATASET_NAMES
from models import SumOfBetaEce


def plot_bayesian_reliability_diagram(ax, ece_model, plot_kwargs: Dict[str, Any] = {}) -> mpl.axes.Axes:
    """
    Plots a Bayesian reliability diagram, along with histogram of scores, given a trained SumOfBetaEce model "ece_model".
    :param ax:
    :param ece_model: SumOfBetaEce
    :param plot_kwargs:
    :return: ax
    """
    _plot_kwargs = DEFAULT_PLOT_KWARGS.copy()
    _plot_kwargs.update(plot_kwargs)

    params_alpha, params_beta = ece_model.get_params()
    counts = ece_model.counts_per_bin
    params_weight = counts / sum(counts)
    num_bins = params_weight.shape[0]

    mean_acc = params_alpha / (params_alpha + params_beta)
    beta_posterior_p025 = beta.ppf(0.025, params_alpha, params_beta)
    beta_posterior_p975 = beta.ppf(0.975, params_alpha, params_beta)

    error_upper = beta_posterior_p975 - mean_acc
    error_lower = mean_acc - beta_posterior_p025

    x = [i + 0.5 for i in range(num_bins)]
    # ax1.grid(True)
    ax.plot(x, mean_acc, c="r", **_plot_kwargs)
    ax.errorbar(x, mean_acc, yerr=(error_lower, error_upper), fmt='o', color='r', **_plot_kwargs)
    ax.plot(np.linspace(0, 1, num_bins + 1), c="gray", **_plot_kwargs)
    ax.fill_between(x, mean_acc, np.linspace(0, 1, num_bins + 1)[:-1] + 0.05, color="gray", alpha=0.3, **_plot_kwargs)

    # ax1.set_xlabel("Score(Model Confidence)")
    ax.set_xlim((0.0, num_bins))
    ax.set_xticks(range(num_bins + 1))
    ax.set_xticklabels(["%.1f" % i for i in np.linspace(0, 1, num_bins + 1)])
    # ax.set_ylabel("Accuracy")
    ax.set_ylim((0.0, 1.0))
    ax.set_yticks(np.linspace(0, 1, 10 + 1))
    ax.set_yticklabels(["%.1f" % i for i in np.linspace(0, 1, 10 + 1)])

    # add histogram to the reliability diagram
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.bar(x, params_weight, color=color, alpha=0.5, label="Histogram", width=1.0, **_plot_kwargs)
    # ax2.set_ylabel('Histogram', color=color)
    ax2.set_ylim((0.0, 2.0))
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_yticklabels([0, 0.5, 1.0], color=color, fontsize=3)
    ax2.yaxis.set_label_coords(1.01, 0.25)

    return ax


def main(args: argparse.Namespace):
    with mpl.rc_context(rc=DEFAULT_RC):

        fig, axes = plt.subplots(nrows=3, ncols=len(DATASET_NAMES), dpi=300, sharey=True, sharex=True)
        col_idx = 0

        for dataset in DATASET_NAMES:

            categories, observations, confidences, idx2category, category2idx, labels = prepare_data(
                DATAFILE_LIST[dataset], False)
            tmp = list(zip(confidences, observations))
            random.shuffle(tmp)
            confidences, observations = zip(*tmp)

            for (row_idx, N) in enumerate([100, 1000, DATASIZE_DICT[dataset]]):
                print(row_idx, N)

                ece_model = SumOfBetaEce(num_bins=num_bins, pseudocount=args.pseudocount)
                ece_model.update_batch(confidences[:N], observations[:N])

                plot_kwargs = {}
                axes[row_idx, col_idx] = plot_bayesian_reliability_diagram(axes[row_idx, col_idx],
                                                                           ece_model, plot_kwargs=plot_kwargs)

            axes[0, col_idx].set_title(DATASET_NAMES[dataset])
            axes[2, col_idx].set_xlabel("Score(Model Confidence)")
            col_idx += 1

        for row_idx in range(3):
            axes[row_idx, 0].set_ylabel("Accuracy")

        axes[0, 0].text(-5, 0.5, "N=100", verticalalignment='center', rotation=90)
        axes[1, 0].text(-5, 0.5, "N=1000", verticalalignment='center', rotation=90)
        axes[2, 0].text(-5, 0.5, "Label all data", verticalalignment='center', rotation=90)

        fig.tight_layout()
        fig.set_size_inches(TEXT_WIDTH, 3.5)
        fig.subplots_adjust(bottom=0.2, wspace=0.3)

    fig.savefig(FIGURE_DIR + 'reliability_pseudocount%d.pdf' % args.pseudocount, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pseudocount', type=int, default=2, help='Takes value from 2, 5, 10, and maybe 1.')
    args, _ = parser.parse_known_args()

    main(args)
