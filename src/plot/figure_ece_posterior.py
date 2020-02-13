######################################CONSTANTS######################################
DEFAULT_RC = {
    'lines.markersize': 2,
    'font.size': 6,
    'font.family': 'serif',
    'font.serif': ['Times'],
    # 'text.usetex': True,
    'axes.titlesize': 6,
    'axes.labelsize': 5,
    'legend.fontsize': 4,
    'legend.loc': 'upper right',
    'figure.titlesize': 6,
    'xtick.labelsize': 4,
    'ytick.labelsize': 4,
}
DEFAULT_PLOT_KWARGS = {
    'linewidth': 1,
    'linestyle': "--",
}
PRECOMPUTED_GROUND_TRUTH_ECE = {
    'cifar100': 0.09869599923480106,
    'imagenet': 0.05007696763512674,
    'svhn': 0.01095047338003833,
    '20newsgroup': 0.09242892818137771,
    'dbpedia': 0.002903918643656106,
}

COLUMN_WIDTH = 3.25  # Inches

num_bins = 10
pseudocount = 2
N_list = [100, 1000, 10000]
num_samples = 1000

######################################CONSTANTS######################################
import argparse
import pathlib
import random
import sys
from typing import Dict, Any, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from figure_reliability_diagrams import plot_bayesian_reliability_diagram

sys.path.insert(0, '..')
from data_utils import DATAFILE_LIST, prepare_data, FIGURE_DIR
from models import SumOfBetaEce


def plot_ece_samples(ax: mpl.axes.Axes,
                     ground_truth_ece: float,
                     frequentist_ece,
                     samples_posterior: np.ndarray,
                     plot_kwargs: Dict[str, Any] = {}) -> mpl.axes.Axes:
    """

    :param ax:
    :param ground_truth_ece: float
    :param frequentist_ece: float or np.ndarray

    :param samples_posterior:
    :param plot_kwargs:
    :return:
    """
    _plot_kwargs = DEFAULT_PLOT_KWARGS.copy()
    _plot_kwargs.update(plot_kwargs)
    if isinstance(frequentist_ece, float):
        ax.axvline(x=frequentist_ece, label='Frequentist', color='blue', **_plot_kwargs)
    else:
        ax.hist(frequentist_ece, color='blue', alpha=0.7, label='Frequentist', **_plot_kwargs)
    ax.hist(samples_posterior, color='red', label='Bayesian', alpha=0.7, **_plot_kwargs)
    ax.axvline(x=ground_truth_ece, label='Ground truth', color='black', **_plot_kwargs)

    ax.set_xlim(0, 0.3)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])

    return ax


def frequentist_bootstrap_ece(confidences: List[int], observations: List[bool], num_bootstrap_samples: int):
    """
    Draw bootstrap samples of ECE. At each bootstrap step, we resample (num_datapoints, ) instances from
    (confidences, observations) with replacement, and compute and estiamtion of ECE with these samples.
    :param confidences: (num_datapoints, )
    :param observations: (num_datapoints, )
    :return: frequentist_ece: np.ndarray(num_bootstrap_samples, )
        bootstrap samples of ECE
    """
    frequentist_ece = np.zeros(num_bootstrap_samples, )
    num_datapoints = len(confidences)
    for idx in range(num_bootstrap_samples):
        indices = np.random.choice(np.arange(num_datapoints), size=num_datapoints, replace=True)
        confidences_resampled = [confidences[i] for i in indices]
        observations_resampled = [observations[i] for i in indices]
        ece_model = SumOfBetaEce(num_bins=10, pseudocount=1e-6)
        ece_model.update_batch(confidences_resampled, observations_resampled)
        frequentist_ece[idx] = ece_model.frequentist_eval
    return frequentist_ece


def main(args):
    dataset = 'cifar100'
    datafile = DATAFILE_LIST[dataset]
    categories, observations, confidences, idx2category, category2idx, labels = prepare_data(datafile, False)

    tmp = list(zip(confidences, observations))
    random.shuffle(tmp)
    confidences, observations = zip(*tmp)

    ece_model = SumOfBetaEce(num_bins=10, pseudocount=pseudocount)

    with mpl.rc_context(rc=DEFAULT_RC):
        fig, axes = plt.subplots(ncols=2, nrows=3, dpi=300)

        for i in range(len(N_list)):
            tmp = 0 if i == 0 else N_list[i - 1]

            ece_model.update_batch(confidences[tmp: N_list[i]], observations[tmp: N_list[i]])
            samples_posterior = ece_model.sample(num_samples)

            print(args.frequentist_bootstrap)

            if args.frequentist_bootstrap:
                file = pathlib.Path(FIGURE_DIR + 'frequentist_ece_%d.csv' % i)
                if file.exists():
                    frequentist_ece = np.genfromtxt(file)
                else:
                    frequentist_ece = frequentist_bootstrap_ece(confidences[:N_list[i]], observations[:N_list[i]],
                                                                num_samples)
                    np.savetxt(file, frequentist_ece, delimiter=',')

            else:
                frequentist_ece = ece_model.frequentist_eval

            ground_truth_ece = PRECOMPUTED_GROUND_TRUTH_ECE[dataset]

            plot_kwargs = {}
            axes[i, 0] = plot_bayesian_reliability_diagram(axes[i, 0], ece_model, plot_kwargs=plot_kwargs)
            axes[i, 0].set_ylabel("Accuracy")
            axes[i, 1] = plot_ece_samples(axes[i, 1], ground_truth_ece, frequentist_ece, samples_posterior,
                                          plot_kwargs=plot_kwargs)
            axes[i, 1].tick_params(left=False)
            axes[i, 1].tick_params(labelleft=False)
            # axes[i, 1].set_ylabel("Histogram")

        axes[-1, 0].set_xlabel("Score(Model Confidence)")
        axes[-1, 1].set_xlabel("ECE")
        axes[-1, 1].legend()

        axes[0, 0].text(-4, 0.5, "N=%d" % N_list[0], verticalalignment='center', rotation=90)
        axes[1, 0].text(-4, 0.5, "N=%d" % N_list[1], verticalalignment='center', rotation=90)
        axes[2, 0].text(-4, 0.5, "N=%d" % N_list[2], verticalalignment='center', rotation=90)

        fig.set_size_inches(COLUMN_WIDTH, 3.0)
        fig.subplots_adjust(bottom=0.05, wspace=0.05)
        fig.tight_layout()

    fig.savefig(FIGURE_DIR + 'cifar100_ece_posterior.pdf', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    # plot posterior of ECE along with posterior Bayesian reliability diagram for cifar100
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequentist_bootstrap', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='true or false, bootstrap to get uncertainty about frequentist estimation or not')
    args, _ = parser.parse_known_args()
    main(args)
