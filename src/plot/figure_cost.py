######################################CONSTANTS######################################
METHOD_NAME_DICT = {'random_no_prior': 'Non-active',
                    #                         'random_uniform': 'non-active_uniform',
                    #                         'random_informed': 'non-active_informed',
                    'active': 'TS (uninformative)',
                    'active_informed': 'TS (informative)'}
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
}

COLUMN_WIDTH = 3.25  # Inche
LOG_FREQ = 10

FIGURE_DIR = '../../figures/'
DATA_DIR = '/Volumes/deepdata/bayesian_blackbox/output_from_datalab_20200201/output/cost_result_matrices/'
RESULTS_DIR = '/Volumes/deepdata/bayesian_blackbox/output_from_datalab_20200201/output/costs/cifar100/'
######################################CONSTANTS######################################
import sys

sys.path.insert(0, '..')
from typing import Dict, Any
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from data_utils import *


def plot_topk_cost(ax: mpl.axes.Axes,
                   experiment_name: str,
                   eval_metric: str,
                   pool_size: int,
                   plot_kwargs: Dict[str, Any] = {}) -> None:
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

    for method in METHOD_NAME_DICT:
        metric_eval = np.load(
            RESULTS_DIR + experiment_name + ('/%s_%s_top1_pseudocount1.0.npy' % (method, eval_metric)))
        x = np.arange(len(metric_eval)) * LOG_FREQ / pool_size
        ax.plot(x, metric_eval, label=METHOD_NAME_DICT[method], **_plot_kwargs)

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


def plot_cost_matrix(new_idx):
    plot_kwargs = {}
    with mpl.rc_context(rc=DEFAULT_RC):
        _plot_kwargs = DEFAULT_PLOT_KWARGS.copy()
        _plot_kwargs.update(plot_kwargs)
        fig, axes = plt.subplots(2, 1, sharex=True)

        axes[0].imshow(np.load(DATA_DIR + 'cifar100_people_full/costs.npy')[:, new_idx][new_idx, :].T,
                       **_plot_kwargs)
        axes[1].imshow(np.load(DATA_DIR + 'cifar100_superclass_full/costs.npy')[:, new_idx][new_idx, :].T,
                       **_plot_kwargs)

        axes[0].set_title('Human')
        axes[1].set_title('Superclass')
        axes[0].set_ylabel("True label", labelpad=0.3)
        axes[1].set_ylabel("True label", labelpad=0.3)
        axes[1].set_xlabel("Predicted label", labelpad=0.8)
        axes[0].xaxis.set_ticks(np.arange(0, 101, 10))
        axes[0].yaxis.set_ticks(np.arange(0, 101, 10))
        axes[1].xaxis.set_ticks(np.arange(0, 101, 10))
        axes[1].yaxis.set_ticks(np.arange(0, 101, 10))

        axes[0].tick_params(bottom=False, labelbottom=False, pad=0.2, length=1)
        axes[1].tick_params(pad=0.2, length=1)
        fig.subplots_adjust(bottom=0.3, wspace=0.01)
        fig.set_size_inches(COLUMN_WIDTH * 0.42, 2.4)
        fig.tight_layout()

    fig.savefig(FIGURE_DIR + 'cost_matrix.pdf', bbox_inches='tight', pad_inches=0)


def plot_comparison():
    plot_kwargs = {}
    with mpl.rc_context(rc=DEFAULT_RC):
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0] = plot_topk_cost(axes[0], experiment_name='human', eval_metric='mrr', pool_size=10000,
                                 plot_kwargs=plot_kwargs)
        axes[1] = plot_topk_cost(axes[1], experiment_name='superclass', eval_metric='mrr', pool_size=10000,
                                 plot_kwargs=plot_kwargs)
        axes[1].legend()
        axes[0].set_ylabel("MRR, top-1", labelpad=0.5)
        axes[1].set_ylabel("MRR, top-1", labelpad=0.5)
        axes[1].set_xlabel("#queries", labelpad=0.5)
        axes[0].set_title('Human')
        axes[1].set_title('Superclass')

        axes[0].tick_params(pad=0.2, length=2)
        axes[1].tick_params(pad=0.2, length=2)

        fig.subplots_adjust(bottom=0.3, wspace=0.01)
        fig.set_size_inches(COLUMN_WIDTH * 0.6, 2.4)
        fig.tight_layout()

    fig.savefig(FIGURE_DIR + 'cost_comparison.pdf', bbox_inches='tight', pad_inches=0)


def plot_confusion(new_idx):
    plot_kwargs = {}
    with mpl.rc_context(rc=DEFAULT_RC):
        _plot_kwargs = DEFAULT_PLOT_KWARGS.copy()
        _plot_kwargs.update(plot_kwargs)
        fig, axes = plt.subplots(2, 3, sharey=True, sharex=True)

        for idx, method_name in enumerate(['random_uniform', 'active_informed']):
            matrices = (
                np.load(RESULTS_DIR + 'superclass/%s_confusion_log_top1_pseudocount1.0.npy' % method_name)[:, new_idx,
                :][:, :, new_idx])
            if method_name == 'active_informed':
                prior = (np.load(DATA_DIR + 'cifar100_superclass_full/informed_prior.npy'))[new_idx, :][:, new_idx]
            else:
                prior = np.ones((100, 100)) * 1. / 10

            for (i, num_samples) in enumerate([0, 99, 999]):
                matrix = matrices[num_samples]  # + prior
                matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]
                axes[idx][i].imshow(np.log(matrix).T, vmin=-20, vmax=0, **_plot_kwargs)

                axes[idx][i].xaxis.set_ticks(np.arange(0, 101, 10))
                axes[idx][i].yaxis.set_ticks(np.arange(0, 101, 10))
                if i == 0:
                    axes[idx, i].set_ylabel("True label", labelpad=0.5)
                if idx == 1:
                    axes[idx, i].set_xlabel("Predicted label", labelpad=0.5)

                if i > 0:
                    axes[idx, i].tick_params(left=False)
                    axes[idx, i].tick_params(labelleft=False)
                axes[0, i].tick_params(bottom=False)
                axes[0, i].tick_params(labelbottom=False)
                axes[idx, i].tick_params(pad=0.2, length=2)

        axes[0, 0].text(-35, 50, "Uninformative prior", verticalalignment='center', rotation=90)
        axes[1, 0].text(-35, 50, "Informative prior", verticalalignment='center', rotation=90)
        axes[0, 0].set_title("N = 10")
        axes[0, 1].set_title("N = 1000")
        axes[0, 2].set_title("All labeled data")
        fig.subplots_adjust(bottom=-0.3, wspace=0.01)
        fig.tight_layout()
        fig.set_size_inches(COLUMN_WIDTH * 0.9, 2.4)

    fig.savefig(FIGURE_DIR + 'cost_confusion_matrix.pdf', bbox_inches='tight', pad_inches=0)


def main():
    class_idx = {s: i for i, s in enumerate(cifar100_classes)}
    superclass_idx = {s: i for i, s in enumerate(cifar100_superclasses)}
    superclass_lookup = {}
    for superclass, class_list in cifar100_reverse_superclass_lookup.items():
        for _class in class_list:
            superclass_lookup[class_idx[_class]] = superclass_idx[superclass]

    new_idx = []
    for superclass_name in cifar100_reverse_superclass_lookup:
        for class_name in cifar100_reverse_superclass_lookup[superclass_name]:
            new_idx.append(class_idx[class_name])
    new_idx = np.array(new_idx)

    plot_cost_matrix(new_idx)
    plot_comparison()
    plot_confusion(new_idx)


if __name__ == "__main__":
    main()
