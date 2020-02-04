###################CONSTANTS
METRIC = 'accuracy'
MODE = 'min'

RESULTS_DIR = '/Volumes/deepdata/bayesian_blackbox/output_from_datalab_20200201/output/active_learning_topk/'
RUNS = 100
LOG_FREQ = 100
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
EVAL_METRIC_NAMES = {
    'avg_num_agreement': '#agreements',
    'mrr': 'MRR'
}
COLOR = {'non-active_no_prior': '#1f77b4',
         'ts_uniform': '#ff7f0e',
         'ts_informed': 'green'
         }
METHOD_NAME_DICT = {'non-active_no_prior': 'Non-active',
                    #                         'non-active_uniform': 'non-active_uniform',
                    #                         'non-active_informed': 'non-active_informed',
                    'ts_uniform': 'TS',
                    'ts_informed': 'TS (informative)'
                    }
TOPK_METHOD_NAME_DICT = {'non-active_no_prior': 'Non-active',
                         #                         'non-active_uniform': 'non-active_uniform',
                         #                         'non-active_informed': 'non-active_informed',
                         'ts_uniform': 'MP-TS',
                         'ts_informed': 'MP-TS (informative)'
                         }
DEFAULT_RC = {
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
    'linewidth': 1.2
}

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875

import argparse
from typing import Dict, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from data_utils import datasize_dict


def plot_topk_accuracy(ax: mpl.axes.Axes,
                       experiment_name: str,
                       topk: int,
                       eval_metric: str,
                       pool_size: int,
                       threshold: float,
                       plot_kwargs: Dict[str, Any] = {},
                       plot_informed: bool = False) -> None:
    """
    Replicates Figure 2 in [CITE PAPER].

    Parameters
    ===
    experiment_name: str.
        Experimental results were written to files under a directory named using experiment_name.
    eval_metric: str.
        Takes value from ['avg_num_agreement', 'mrr']
    pool_size: int.
        Total size of pool from which samples were drawn.
    plot_kwargs : dict.
        Keyword arguments passed to the plot.
    Returns
    ===
    fig, axes : The generated matplotlib Figure and Axes.
    """

    _plot_kwargs = DEFAULT_PLOT_KWARGS.copy()
    _plot_kwargs.update(plot_kwargs)

    if plot_informed:
        benchmark = 'ts_informed'
        method_list = {'ts_informed': 'TS (informative)',
                       'ts_uniform': 'TS (non-informative)'}
    else:
        benchmark = 'ts_uniform'
        method_list = {'non-active_no_prior', 'ts_uniform'}

    for method in method_list:
        metric_eval = np.load(
            RESULTS_DIR + experiment_name + ('%s_%s.npy' % (eval_metric, method))).mean(axis=0)
        x = np.arange(len(metric_eval)) * LOG_FREQ / pool_size
        if topk == 1:
            if plot_informed:
                label = method_list[method]
            else:
                label = METHOD_NAME_DICT[method]
        else:
            label = TOPK_METHOD_NAME_DICT[method]
        ax.plot(x, metric_eval, label=label, color=COLOR[method], **_plot_kwargs)

        if method == benchmark:
            if method == benchmark:
                if max(metric_eval) > threshold:
                    cutoff = list(map(lambda i: i > threshold, metric_eval.tolist()[10:])).index(True) + 10
                    cutoff = min(int(cutoff * 1.2), len(metric_eval) - 1)
                else:
                    cutoff = len(metric_eval) - 1

    ax.set_xlim(0, cutoff * LOG_FREQ / pool_size)
    ax.set_ylim(0, 1.0)
    xmin, xmax = ax.get_xlim()
    step = ((xmax - xmin) / 4.0001)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.xaxis.set_ticks(np.arange(xmin, xmax + 0.001, step))
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.20))
    ax.tick_params(pad=0.25, length=1.5)

    return ax


def main(eval_metric: str, top1: bool, pseudocount: int, threshold: float) -> None:
    with mpl.rc_context(rc=DEFAULT_RC):
        fig, axes = plt.subplots(ncols=len(TOPK_DICT), dpi=300, sharey=True)
        idx = 0
        for dataset in TOPK_DICT:
            if top1:
                topk = 1
            else:
                topk = TOPK_DICT[dataset]
            experiment_name = '%s_%s_%s_top%d_runs%d_pseudocount%.2f/' % \
                              (dataset, METRIC, MODE, topk, RUNS, pseudocount)
            plot_kwargs = {}
            plot_topk_accuracy(axes[idx],
                               experiment_name,
                               topk,
                               eval_metric,
                               datasize_dict[dataset],
                               threshold=threshold,
                               plot_kwargs=plot_kwargs)
            if topk == 1:
                axes[idx].set_title(DATASET_NAMES[dataset])
            else:
                axes[idx].set_xlabel("#queries")
            if idx > 0:
                axes[idx].tick_params(left=False)
            idx += 1

        axes[-1].legend()
        if topk == 1:
            axes[0].set_ylabel("MRR, top1")
        else:
            axes[0].set_ylabel("MRR, topK")
        fig.tight_layout()
        fig.set_size_inches(TEXT_WIDTH, 0.8)
        fig.subplots_adjust(bottom=0.05, wspace=0.20)

    if top1:
        figname = '../figures/%s_%s_%s_top1_pseudocount%d.pdf' % (METRIC, MODE, eval_metric, pseudocount)
    else:
        figname = '../figures/%s_%s_%s_topk_pseudocount%d.pdf' % (METRIC, MODE, eval_metric, pseudocount)
    fig.savefig(figname, bbox_inches='tight', pad_inches=0)


def main_informed(eval_metric: str, pseudocount: int, threshold: float) -> None:
    with mpl.rc_context(rc=DEFAULT_RC):
        # fig, axes = plt.subplots(ncols=2, nrows=2, dpi=300, sharey=True)
        fig, axes = plt.subplots(ncols=2, nrows=1, dpi=300, sharey=True)
        for idx, dataset in enumerate(['imagenet', 'svhn']):
            plot_kwargs = {}

            experiment_name = '%s_%s_%s_top%d_runs%d_pseudocount%.2f/' % \
                              (dataset, METRIC, MODE, 1, RUNS, pseudocount)
            plot_topk_accuracy(axes[idx],
                               experiment_name,
                               1,
                               eval_metric,
                               datasize_dict[dataset],
                               threshold=threshold,
                               plot_kwargs=plot_kwargs,
                               plot_informed=True)
        axes[1].legend()
        axes[0].set_ylabel("MRR, top1")
        axes[0].set_xlabel("#queries")
        axes[1].set_xlabel("#queries")
        axes[0].set_title(DATASET_NAMES['imagenet'])
        axes[1].set_title(DATASET_NAMES['svhn'])
        fig.tight_layout()
        fig.set_size_inches(COLUMN_WIDTH * 0.9, 1.0)
        fig.subplots_adjust(wspace=0.20)

    figname = '../figures/informed_%s_%s_%s_pseudocount%d.pdf' % (METRIC, MODE, eval_metric, pseudocount)
    fig.savefig(figname, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    threshold = 0.98  # threshold for x-axis cutoff
    parser = argparse.ArgumentParser()
    parser.add_argument('-pseudocount', type=int, default=2, help='Takes value from 2, 10, 100, and maybe 1.')

    args, _ = parser.parse_known_args()
    for eval_metric in ['avg_num_agreement', 'mrr']:
        for top1 in [True, False]:
            main(eval_metric, top1, args.pseudocount, threshold)
        main_informed(eval_metric, args.pseudocount, threshold)
