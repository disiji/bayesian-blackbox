import random
from typing import Dict, Any

import matplotlib as mpl
import matplotlib.pyplot as plt

from data_utils import datafile_dict, prepare_data, num_classes_dict
from models import SumOfBetaEce

DATASET_NAMES = {
    'cifar100': 'CIFAR-100',
    'imagenet': 'ImageNet',
    'svhn': 'SVHN',
    '20newsgroup': '20 Newsgroups',
    'dbpedia': 'DBpedia',
}
DEFAULT_RC = {
    'lines.markersize': 2,
    'font.size': 6,
    'font.family': 'serif',
    'font.serif': ['Times'],
    # 'text.usetex': True,
    'axes.titlesize': 6,
    'axes.labelsize': 5,
    'legend.fontsize': 5,
    'legend.loc': 'upper right',
    'figure.titlesize': 6,
    'xtick.labelsize': 3,
    'ytick.labelsize': 3,
}
DEFAULT_PLOT_KWARGS = {
    'linewidth': 1,
    'linestyle': "--",
}
num_bins = 10
COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875

from figure_reliability_diagrams import plot_bayesian_reliability_diagram

pseudocount = 2
N_list = [100, 1000, 10000]
num_samples = 1000


def plot_ece_samples(ax, samples_prior, samples_posterior, plot_kwargs: Dict[str, Any] = {}):
    _plot_kwargs = DEFAULT_PLOT_KWARGS.copy()
    _plot_kwargs.update(plot_kwargs)

    # ax.hist(samples_prior, color='blue', label='prior', **_plot_kwargs)
    ax.hist(samples_posterior, color='red', label='posterior', **_plot_kwargs)
    ax.set_xlim(0, 0.3)
    ax.legend()

    return ax


def main():
    dataset = 'cifar100'
    datafile = datafile_dict[dataset]

    categories, observations, confidences, idx2category, category2idx, labels = prepare_data(datafile, False)
    # tmp = list(zip(confidences, observations))
    # random.shuffle(tmp)
    # confidences, observations = zip(*tmp)

    ece_model = SumOfBetaEce(num_bins=10, pseudocount=pseudocount)
    samples_prior = ece_model.sample(num_samples)

    with mpl.rc_context(rc=DEFAULT_RC):
        fig, axes = plt.subplots(ncols=2, nrows=3, dpi=300)

        for i in range(len(N_list)):
            tmp = 0 if i == 0 else N_list[i - 1]
            ece_model.update_batch(confidences[tmp: N_list[i]], observations[tmp: N_list[i]])
            samples_posterior = ece_model.sample(num_samples)

            plot_kwargs = {}
            axes[i, 0] = plot_bayesian_reliability_diagram(axes[i, 0], ece_model, plot_kwargs=plot_kwargs)
            axes[i, 0].set_xlabel("Score(Model Confidence)")
            axes[i, 0].set_ylabel("Accuracy")
            axes[i, 1] = plot_ece_samples(axes[i, 1], samples_prior, samples_posterior, plot_kwargs=plot_kwargs)
            axes[i, 1].set_xlabel("ECE")
            axes[i, 1].tick_params(left=False)
            axes[i, 1].tick_params(labelleft=False)
            # axes[i, 1].set_ylabel("Histogram")

        axes[0, 0].text(-4, 0.5, "N=100", verticalalignment='center', rotation=90)
        axes[1, 0].text(-4, 0.5, "N=1000", verticalalignment='center', rotation=90)
        axes[2, 0].text(-4, 0.5, "Label all data", verticalalignment='center', rotation=90)

        fig.set_size_inches(COLUMN_WIDTH, 3.5)
        fig.subplots_adjust(bottom=0.05, wspace=0.05)
        fig.tight_layout()

    fig.savefig('../figures/cifar100_ece_posterior.pdf', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    # plot posterior of ECE along with posterior Bayesian reliability diagram for cifar100
    main()
