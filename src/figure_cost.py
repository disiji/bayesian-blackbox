from typing import Dict, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from cifar100meta import *

DATA_DIR = '/Volumes/deepdata/bayesian_blackbox/output_from_datalab_20200201/output/cost_result_matrices/'
RESULTS_DIR = '/Volumes/deepdata/bayesian_blackbox/output_from_datalab_20200201/output/costs/cifar100/'
METHOD_NAME_DICT = {'random_no_prior': 'Non-active',
                    #                         'random_uniform': 'non-active_uniform',
                    #                         'random_informed': 'non-active_informed',
                    'active': 'TS (uniform)',
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
    'xtick.labelsize': 3,
    'ytick.labelsize': 3,
}
DEFAULT_PLOT_KWARGS = {
}

COLUMN_WIDTH = 3.25  # Inche
LOG_FREQ = 100


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

    benchmark = 'active_informed'
    threshold = 0.99

    for method in METHOD_NAME_DICT:
        metric_eval = np.load(
            RESULTS_DIR + experiment_name + ('/%s_%s_top1_pseudocount1.0.npy' % (method, eval_metric)))
        x = np.arange(len(metric_eval)) * LOG_FREQ / pool_size
        ax.plot(x, metric_eval, label=METHOD_NAME_DICT[method], **_plot_kwargs)

        if method == benchmark:
            cutoff = len(metric_eval) - 1
            if max(metric_eval) > threshold:
                cutoff = list(map(lambda i: i > threshold, metric_eval.tolist()[10:])).index(True) + 10
                cutoff = min(int(cutoff * 1.5), len(metric_eval) - 1)

    cutoff = len(metric_eval) - 1
    print(cutoff * LOG_FREQ / pool_size)
    ax.set_xlim(0, cutoff * LOG_FREQ / pool_size)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_ticks(np.arange(0, cutoff * LOG_FREQ / pool_size, 0.10))
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.20))

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

        axes[0].text(-26, 50, "Human", verticalalignment='center', rotation=90)
        axes[1].text(-26, 50, "Superclass", verticalalignment='center', rotation=90)
        axes[0].tick_params(bottom=False, pad=0.1)
        axes[0].tick_params(labelbottom=False, pad=0.1)
        axes[0].set_ylabel("Predicted label", labelpad=0.2)
        axes[1].set_ylabel("Predicted label", labelpad=0.2)
        axes[1].set_xlabel("True label", labelpad=0.8)
        axes[0].tick_params(pad=0.2, length=1)
        axes[1].tick_params(pad=0.2, length=1)
        axes[0].xaxis.set_ticks(np.arange(0, 101, 10))
        axes[0].yaxis.set_ticks(np.arange(0, 101, 10))
        axes[1].xaxis.set_ticks(np.arange(0, 101, 10))
        axes[1].yaxis.set_ticks(np.arange(0, 101, 10))

        fig.tight_layout()
        fig.set_size_inches(COLUMN_WIDTH * 0.4, 2.4)

    fig.savefig('../figures/cost/cost_matrix.pdf', bbox_inches='tight')


def plot_comparison():
    plot_kwargs = {}
    with mpl.rc_context(rc=DEFAULT_RC):
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0] = plot_topk_cost(axes[0], experiment_name='human', eval_metric='mrr', pool_size=10000,
                                 plot_kwargs=plot_kwargs)
        axes[1] = plot_topk_cost(axes[1], experiment_name='superclass', eval_metric='mrr', pool_size=10000,
                                 plot_kwargs=plot_kwargs)
        axes[1].legend()
        axes[0].set_ylabel("MRR, top1", labelpad=0.5)
        axes[1].set_ylabel("MRR, top1", labelpad=0.5)
        axes[1].set_xlabel("#queries", labelpad=0.5)
        axes[0].tick_params(pad=0.2, length=2)
        axes[1].tick_params(pad=0.2, length=2)
        axes[0].text(-0.18, 0.5, "Human", verticalalignment='center', rotation=90)
        axes[1].text(-0.18, 0.5, "Superclass", verticalalignment='center', rotation=90)

        fig.tight_layout()
        fig.set_size_inches(COLUMN_WIDTH * 0.6, 2.4)

    fig.savefig('../figures/cost/cost_comparison.pdf', bbox_inches='tight')


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

            for (i, num_samples) in enumerate([9, 99, 99]):

                matrix = matrices[num_samples] + prior
                matrix = matrix / matrix.sum(axis=1)[:, np.newaxis]
                axes[idx][i].imshow(np.log(matrix).T, vmin=-10, vmax=0, **_plot_kwargs)
                axes[idx][i].xaxis.set_ticks(np.arange(0, 101, 10))
                axes[idx][i].yaxis.set_ticks(np.arange(0, 101, 10))
                if i == 0:
                    axes[idx, i].set_ylabel("Predicted label", labelpad=0.5)
                if idx == 1:
                    axes[idx, i].set_xlabel("True label", labelpad=0.5)
                if i > 0:
                    axes[idx, i].tick_params(left=False)
                    axes[idx, i].tick_params(labelleft=False)
                axes[0, i].tick_params(bottom=False)
                axes[0, i].tick_params(labelbottom=False)
                axes[idx, i].tick_params(pad=0.2, length=2)

        axes[0, 0].text(-35, 50, "Uninformative prior", verticalalignment='center', rotation=90)
        axes[1, 0].text(-35, 50, "Informative prior", verticalalignment='center', rotation=90)
        axes[0, 0].text(50, -5, "N = 100", horizontalalignment='center')
        axes[0, 1].text(50, -5, "N = 1000", horizontalalignment='center')
        axes[0, 2].text(50, -5, "N = 10000", horizontalalignment='center')
        fig.subplots_adjust(bottom=-0.3, wspace=0.01)
        fig.tight_layout()
        fig.set_size_inches(COLUMN_WIDTH * 0.9, 2.4)

    fig.savefig('../figures/cost/cost_confusion_matrix.pdf', bbox_inches='tight')


def main():
    class_idx = {s: i for i, s in enumerate(classes)}
    superclass_idx = {s: i for i, s in enumerate(superclasses)}
    superclass_lookup = {}
    for superclass, class_list in reverse_superclass_lookup.items():
        for _class in class_list:
            superclass_lookup[class_idx[_class]] = superclass_idx[superclass]

    new_idx = []
    for superclass_name in reverse_superclass_lookup:
        for class_name in reverse_superclass_lookup[superclass_name]:
            new_idx.append(class_idx[class_name])
    new_idx = np.array(new_idx)

    plot_cost_matrix(new_idx)
    plot_comparison()
    plot_confusion(new_idx)


if __name__ == "__main__":
    main()
