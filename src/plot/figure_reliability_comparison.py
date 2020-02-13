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
    'legend.loc': 'upper right',
    'figure.titlesize': 6,
    'xtick.labelsize': 4,
    'ytick.labelsize': 4,
}
DEFAULT_PLOT_KWARGS = {
    'linewidth': 1,
    'linestyle': "-",
}
PRECOMPUTED_GROUND_TRUTH_ECE = {
    'cifar100': 0.09869599923480106,
    'imagenet': 0.05007696763512674,
    'svhn': 0.01095047338003833,
    '20newsgroup': 0.09242892818137771,
    'dbpedia': 0.002903918643656106,
}
num_bins = 10
TEXT_WIDTH = 6.299213  # Inches

ylims = [30, 100, 150, 16, 200]
######################################CONSTANTS######################################
import sys

sys.path.insert(0, '..')

import argparse
from typing import Dict, Any, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from data_utils import FIGURE_DIR, DATASET_NAMES, RESULTS_DIR

RESULTS_DIR = RESULTS_DIR + 'bayesian_reliability_comparison/online_weights/'


def plot_reliability_comparison(ax: mpl.axes.Axes,
                                N_list: List[int],
                                bayesian_ece: np.ndarray,
                                frequentist_ece: np.ndarray,
                                bayesian_ece_std: np.ndarray,
                                frequentist_ece_std: np.ndarray,
                                ece_true: float,
                                ylim: float,
                                plot_kwargs: Dict[str, Any] = {},
                                plot_errorbar: bool = False) -> mpl.axes.Axes:
    """
    Plot comparison of calibration estimation error obtained with the Bayesian or frequentist method.
    :param ax:
    :param N_list: List[int]
    :param bayesian_ece: (len(N_list), ) as type float.
            Estimation of ECE obtained with the Bayesian method.
    :param frequentist_ece: (len(N_list), ) as type float.
            Estimation of ECE obtained with the frequentist method.
    :param bayesian_ece_std: (len(N_list), ) as type float.
            Standard deviation of estimation of ECE obtained with the Bayesian method.
    :param frequentist_ece_std: (len(N_list), ) as type float.
            Standard deviation of estimation of ECE obtained with the frequentist method.
    :param ece_true: float.
            Ground truth ECE
    :param plot_kwargs: dict.
        Keyword arguments passed to the plot.
    :return: ax: the generated matplotlib ax
    """
    _plot_kwargs = DEFAULT_PLOT_KWARGS.copy()
    _plot_kwargs.update(plot_kwargs)
    if plot_errorbar:
        # todo: debug
        ax.errorbar(N_list, (bayesian_ece - ece_true) / ece_true * 100, bayesian_ece_std / ece_true * 100,
                    '-*', **_plot_kwargs, label='Bayesian', color='tab:red')
        ax.errorbar(N_list, (frequentist_ece - ece_true) / ece_true * 100, frequentist_ece_std / ece_true * 100,
                    '-o', **_plot_kwargs, label='Frequentist',
                    color='tab:blue')
    else:
        ax.plot(N_list, (bayesian_ece - ece_true) / ece_true * 100,
                '-*', **_plot_kwargs, label='Bayesian', color='tab:red')
        ax.plot(N_list, (frequentist_ece - ece_true) / ece_true * 100,
                '-o', **_plot_kwargs, label='Frequentist',
                color='tab:blue')
    ax.set_xscale('log')
    ax.set_xlabel('#queries', labelpad=0.2)
    ax.xaxis.set_ticks(N_list)
    ax.set_ylim(ymin=0, ymax=ylim)
    ax.set_yticks((0, ylim / 2, ylim))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.tick_params(pad=0.25, length=1.5)
    return ax


def main(args: argparse.Namespace) -> None:
    with mpl.rc_context(rc=DEFAULT_RC):
        fig, axes = plt.subplots(figsize=(TEXT_WIDTH, 1), ncols=len(DATASET_NAMES), dpi=300)
        idx = 0

        for dataset in DATASET_NAMES:
            # load result files
            df_mean = pd.read_csv(
                RESULTS_DIR + 'frequentist_ground_truth_%s_pseudocount%d.csv' % (
                    dataset, args.pseudocount),
                header=0)
            N_list = df_mean['# N']
            bayesian_ece = df_mean[" bayesian_ece"]
            frequentist_ece = df_mean[" frequentist_ece"]

            df_std = pd.read_csv(
                RESULTS_DIR + 'frequentist_ground_truth_%s_pseudocount%d_std.csv' % (
                    dataset, args.pseudocount),
                header=0)
            bayesian_ece_std = df_std[" bayesian_ece"]
            frequentist_ece_std = df_std[" frequentist_ece"]

            # uncomment if PRECOMPUTED_GROUND_TRUTH_ECE does not exist to compute ground truth ece.
            # datafile = datafile_dict[dataset]
            # categories, observations, confidences, idx2category, category2idx, labels = prepare_data(datafile, False)
            # ground_truth_model = SumOfBetaEce(num_bins=10, pseudocount=args.pseudocount)
            # ground_truth_model.update_batch(confidences, observations)
            # ece_true = ground_truth_model.frequentist_eval  # if we use Bayesian baseline
            ece_true = PRECOMPUTED_GROUND_TRUTH_ECE[dataset]

            plot_kwargs = {}
            axes[idx] = plot_reliability_comparison(axes[idx],
                                                    N_list,
                                                    bayesian_ece,
                                                    frequentist_ece,
                                                    bayesian_ece_std,
                                                    frequentist_ece_std,
                                                    ece_true,
                                                    ylims[idx],
                                                    plot_kwargs=plot_kwargs)
            axes[idx].set_title(DATASET_NAMES[dataset])
            idx += 1

        axes[-1].legend()
        axes[0].set_ylabel('ECE estimation error', labelpad=0.2)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2, wspace=0.2)

    fig.savefig(FIGURE_DIR + 'reliability_comparison_pseudocount%d.pdf' % args.pseudocount, bbox_inches='tight',
                pad_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pseudocount', type=int, default=1, help='Takes value from 2, 5, 10, and maybe 1.')
    args, _ = parser.parse_known_args()

    main(args)
