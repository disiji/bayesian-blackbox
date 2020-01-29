import argparse
import csv
import multiprocessing
import os
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from data_utils import datafile_dict, datasize_dict, output_str_dict, DATASET_LIST
from scipy import stats

num_cores = multiprocessing.cpu_count()
NUM_BINS = 10
PRIORTYPE = 'pseudocount'  #
NUM_RUNS = 100
FIG_OUTPUT = '../figures/reliability_diagram/'
OUTPUT_DIR = "../output/"


def BetaParams(mu: float, var: float) -> Tuple[float, float]:
    """
    Compute parameters of Beta distribution given mean and variance.
    """
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
    beta = alpha * (1 / mu - 1)
    return alpha, beta


def prepare_data(filename: str, N: int, random_seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(random_seed)
    data = np.genfromtxt(filename)
    np.random.shuffle(data)
    data = data[0:N]
    probabilities = data[:, 1:]
    confidence = np.max(probabilities, axis=1)
    Y_predict = np.argmax(probabilities, axis=1)
    Y_true = data[:, 0].astype(int)
    return confidence, Y_predict, Y_true


def get_ground_truth(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(datafile_dict[args.dataset])
    Y_true = data[:, 0]
    probabilities = data[:, 1:]
    confidence = np.max(probabilities, axis=1)
    Y_predict = np.argmax(probabilities, axis=1)
    bins = np.linspace(0, 1, args.num_bins + 1)
    digitized = np.digitize(confidence, bins[1:-1])

    accuracy_bins = np.array([(Y_predict[digitized == i] == Y_true[digitized == i]).mean()
                              for i in range(args.num_bins)])
    accuracy_bins[np.isnan(accuracy_bins)] = 0.0
    density_bins = np.array([(digitized == i).mean() for i in range(0, args.num_bins)])
    diag_bins = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
    calibration_bias = np.abs(accuracy_bins - diag_bins)
    weighted_calibration_error = np.dot(density_bins, calibration_bias)
    unweighted_calibration_error = calibration_bias.mean()

    return {
        'accuracy_bins': accuracy_bins,
        'density_bins': density_bins,
        'weighted_calibration_error': weighted_calibration_error,
        'unweighted_calibration_error': unweighted_calibration_error,
    }


def bayesian_assessment(args: argparse.Namespace,
                        confidence: np.ndarray,
                        Y_predict: np.ndarray,
                        Y_true: np.ndarray,
                        var: int = None,
                        pseudocount: int = None) -> Dict[str, np.ndarray]:
    bins = np.linspace(0, 1, args.num_bins + 1)
    digitized = np.digitize(confidence, bins[1:-1])
    density_bins = [(digitized == i).mean() for i in range(0, args.num_bins)]
    diagonal_bins = [(i + 0.5) / args.num_bins for i in range(0, args.num_bins)]

    # compute prior, update prior
    if args.prior_type == 'fixed_var':
        beta_priors = np.array([BetaParams((i + 0.5) / args.num_bins, var) for i in range(args.num_bins)])
    elif args.prior_type == 'pseudocount':
        beta_priors = np.array([[(i + 0.5) * pseudocount / 10, (9.5 - i) * pseudocount / 10] for i in range(args.num_bins)])

    counts = np.array([(((Y_predict[digitized == i]) == (Y_true[digitized == i])).sum(),
                        ((Y_predict[digitized == i]) != (Y_true[digitized == i])).sum())
                       for i in range(NUM_BINS)])
    empirical_accuracy = counts[:, 0] / (counts[:, 0] + counts[:, 1])
    empirical_accuracy[np.isnan(empirical_accuracy)] = 0
    beta_posteriors = beta_priors + counts

    # compute mean and credible interval of posterior
    beta_posteriors_mean = np.array([beta_posteriors[i, 0] / (beta_posteriors[i].sum())
                                     for i in range(args.num_bins)])
    beta_posterior_p025 = np.array([stats.beta.ppf(0.025, beta_posteriors[i, 0], beta_posteriors[i, 1])
                                    for i in range(args.num_bins)])
    beta_posterior_p975 = np.array([stats.beta.ppf(0.975, beta_posteriors[i, 0], beta_posteriors[i, 1])
                                    for i in range(args.num_bins)])
    return {
        'diagonal_bins': np.array(diagonal_bins),
        'density_bins': np.array(density_bins),
        'empirical_accuracy': empirical_accuracy,
        'beta_posteriors_mean': beta_posteriors_mean,
        'beta_posterior_p025': beta_posterior_p025,
        'beta_posterior_p975': beta_posterior_p975,
    }


def plot_bayesian_reliability_diagram(args: argparse.Namespace,
                                      confidence: np.ndarray,
                                      Y_predict: np.ndarray,
                                      Y_true: np.ndarray,
                                      figname: str,
                                      var: int = None,
                                      pseudocount: int = None) -> None:
    output = bayesian_assessment(args, confidence, Y_predict, Y_true, var, pseudocount)

    fig, ax1 = plt.subplots(figsize=(4.3, 3))
    color = 'tab:red'
    # ax1.grid(True)
    ax1.scatter([i + 0.5 for i in range(args.num_bins)], output['empirical_accuracy'], label="Frequentist", marker="^",
                s=100)
    ax1.plot([i + 0.5 for i in range(args.num_bins)], output['beta_posteriors_mean'], c="r", linestyle="--")
    ax1.errorbar([i + 0.5 for i in range(args.num_bins)],
                 output['beta_posteriors_mean'],
                 yerr=(output['beta_posteriors_mean'] - output['beta_posterior_p025'],
                       output['beta_posterior_p975'] - output['beta_posteriors_mean']),
                 fmt='o', color='r', label="Bayesian")
    ax1.plot(np.linspace(0, 1, args.num_bins + 1), linestyle="--", linewidth=3, c="gray")
    ax1.fill_between([i + 0.5 for i in range(args.num_bins)], output['beta_posteriors_mean'], \
                     np.linspace(0, 1, args.num_bins + 1)[:-1] + 0.05, color="gray", alpha=0.3)
    # ax1.legend(loc='upper left', prop={'size': 10})
    ax1.set_xlim((0.0, args.num_bins))
    ax1.set_xlabel("Score(Model Confidence)", fontsize=14)
    ax1.set_xticks(range(args.num_bins + 1))
    ax1.set_xticklabels(["%.1f" % i for i in np.linspace(0, 1, args.num_bins + 1)])
    ax1.set_ylim((0.0, 1.0))
    ax1.set_ylabel("Estimated Accuracy", fontsize=14)

    # add histogram to the reliability diagram
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.bar([i + 0.5 for i in range(args.num_bins)], output['density_bins'], color=color, alpha=0.5, label="Histogram",
            width=1.0)
    ax2.set_ylabel('Histogram', color=color, fontsize=12)
    ax2.set_ylim((0.0, 2.0))
    ax2.set_yticks([0, 1.0])
    ax2.set_yticklabels([0, 1.0], color=color)
    ax2.yaxis.set_label_coords(1.01, 0.25)
    plt.tight_layout()
    plt.savefig(figname)


def compute_estimation_error(args: argparse.Namespace,
                             N_list: List[int],
                             var: int = None,
                             pseudocount: int = None):
    ground_truth = get_ground_truth(args)

    weighted_pool_bayesian_estimation_error = np.zeros(shape=(args.num_runs, len(N_list)))
    weighted_pool_frequentist_estimation_error = np.zeros(shape=(args.num_runs, len(N_list)))
    weighted_online_bayesian_estimation_error = np.zeros(shape=(args.num_runs, len(N_list)))
    weighted_online_frequentist_estimation_error = np.zeros(shape=(args.num_runs, len(N_list)))
    unweighted_bayesian_estimation_error = np.zeros(shape=(args.num_runs, len(N_list)))
    unweighted_frequentist_estimation_error = np.zeros(shape=(args.num_runs, len(N_list)))
    pool_bayesian_ece = np.zeros(shape=(args.num_runs, len(N_list)))
    pool_frequentist_ece = np.zeros(shape=(args.num_runs, len(N_list)))
    online_bayesian_ece = np.zeros(shape=(args.num_runs, len(N_list)))
    online_frequentist_ece = np.zeros(shape=(args.num_runs, len(N_list)))
    bayesian_mce = np.zeros(shape=(args.num_runs, len(N_list)))
    frequentist_mce = np.zeros(shape=(args.num_runs, len(N_list)))

    for run_idx in range(args.num_runs):
        for i, N in enumerate(N_list):
            confidence, Y_predict, Y_true = prepare_data(datafile_dict[args.dataset], N, random_seed=run_idx)
            output = bayesian_assessment(args, confidence, Y_predict, Y_true, var, pseudocount)

            #### compute metrics
            bayesian_error = np.abs(ground_truth['accuracy_bins'] - output['beta_posteriors_mean'])
            frequentist_error = np.abs(ground_truth['accuracy_bins'] - output['empirical_accuracy'])
            bayesian_calibration_bias = np.abs(output['diagonal_bins'] - output['beta_posteriors_mean'])
            frequentist_calibration_bias = np.abs(output['diagonal_bins'] - output['empirical_accuracy'])
            # empty bins
            bayesian_error[np.isnan(ground_truth['accuracy_bins'])] = 0.0
            frequentist_error[np.isnan(ground_truth['accuracy_bins'])] = 0.0
            bayesian_calibration_bias[np.isnan(ground_truth['accuracy_bins'])] = 0.0
            frequentist_calibration_bias[np.isnan(ground_truth['accuracy_bins'])] = 0.0
            # compute metrics
            weighted_pool_bayesian_estimation_error[run_idx, i] = np.dot(bayesian_error, ground_truth['density_bins'])
            weighted_pool_frequentist_estimation_error[run_idx, i] = np.dot(frequentist_error,
                                                                            ground_truth['density_bins'])
            weighted_online_bayesian_estimation_error[run_idx, i] = np.dot(bayesian_error, output['density_bins'])
            weighted_online_frequentist_estimation_error[run_idx, i] = np.dot(frequentist_error, output['density_bins'])
            unweighted_bayesian_estimation_error[run_idx, i] = bayesian_error.mean()
            unweighted_frequentist_estimation_error[run_idx, i] = frequentist_error.mean()
            pool_bayesian_ece[run_idx, i] = np.dot(bayesian_calibration_bias, ground_truth['density_bins'])
            pool_frequentist_ece[run_idx, i] = np.dot(frequentist_calibration_bias, ground_truth['density_bins'])
            online_bayesian_ece[run_idx, i] = np.dot(bayesian_calibration_bias, output['density_bins'])
            online_frequentist_ece[run_idx, i] = np.dot(frequentist_calibration_bias, output['density_bins'])
            bayesian_mce[run_idx, i] = bayesian_calibration_bias.max()
            frequentist_mce[run_idx, i] = frequentist_calibration_bias.max()

    return {
        "weighted_pool_bayesian_estimation_error": weighted_pool_bayesian_estimation_error,
        "weighted_pool_frequentist_estimation_error": weighted_pool_frequentist_estimation_error,
        "weighted_online_bayesian_estimation_error": weighted_online_bayesian_estimation_error,
        "weighted_online_frequentist_estimation_error": weighted_online_frequentist_estimation_error,
        "unweighted_bayesian_estimation_error": unweighted_bayesian_estimation_error,
        "unweighted_frequentist_estimation_error": unweighted_frequentist_estimation_error,
        "pool_bayesian_ece": pool_bayesian_ece,
        "pool_frequentist_ece": pool_frequentist_ece,
        "online_bayesian_ece": online_bayesian_ece,
        "online_frequentist_ece": online_frequentist_ece,
        "bayesian_mce": bayesian_mce,
        "frequentist_mce": frequentist_mce}


def run_calibration_error(args: argparse.Namespace) -> None:
    PSEUDOCOUNT = [0.1, 1, 10]
    N_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    # N_list = [100, 200, 2000, 10000]

    if args.dataset == '20newsgroup':  # 5607
        N_list = N_list[:-1]

    if PRIORTYPE == 'fixed_var':
        pass
    elif PRIORTYPE == 'pseudocount':
        for pseudo_n in PSEUDOCOUNT:
            estimation_error_output = compute_estimation_error(args, N_list,
                                                               pseudocount=pseudo_n)
            print(estimation_error_output)

            for metric in estimation_error_output:
                np.savetxt(args.output_dir + 'accuracy_estimation_error/' + output_str_dict[metric] % (
                    args.dataset, pseudo_n, NUM_RUNS),
                           estimation_error_output[metric],
                           delimiter=',')


def run_true_calibration_error(args: argparse.Namespace) -> None:
    results = get_ground_truth(args)

    calibration_error_ground_truth_filename = output_str_dict + "calibration_error_ground_truth.csv"

    if not os.path.exists(calibration_error_ground_truth_filename):
        with open(calibration_error_ground_truth_filename, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(['dataset', 'weighted', 'calibration_error'])

    with open(calibration_error_ground_truth_filename, 'a+') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows([[args.dataset, 'weighted', results['weighted_calibration_error']],
                          [args.dataset, 'unweighted', results['unweighted_calibration_error']]])


def run_reliability_diagrams(args: argparse.Namespace) -> None:
    VAR_list = [0.01, 0.05, 0.10, 0.25]  # takes value from (0, 0.25); variance of beta prior
    PSEUDOCOUNT = [0.1, 1, 10]
    N_list = [100, 1000, datasize_dict[args.dataset]]

    for N in N_list:
        confidence, Y_predict, Y_true = prepare_data(datafile_dict[args.dataset], N)
        if PRIORTYPE == 'fixed_var':
            for var in VAR_list:
                figname = args.fig_output + "reliability_plot_%s_N%d_VAR%.2f.pdf" % (args.dataset, N, var)
                plot_bayesian_reliability_diagram(args, confidence, Y_predict, Y_true, figname, var=var)
        elif PRIORTYPE == 'pseudocount':
            for pseudo_n in PSEUDOCOUNT:
                figname = args.fig_output + "reliability_plot_%s_N%d_PseudoCount%.1f.pdf" % (
                    args.dataset, N, pseudo_n)
                plot_bayesian_reliability_diagram(args, confidence, Y_predict, Y_true, figname, pseudocount=pseudo_n)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default='cifar100', help='input dataset')
    parser.add_argument('--num_runs', type=int, default=NUM_RUNS, help='number of runs')
    parser.add_argument('--num_bins', type=int, default=NUM_BINS, help='number of bins in reliability diagram')
    parser.add_argument('--prior_type', type=str, default=PRIORTYPE, help='fixed_var or pseudocount')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='output prefix')
    parser.add_argument('--fig_output', type=str, default=FIG_OUTPUT, help='figure output prefix')
    args, _ = parser.parse_known_args()

    if args.dataset not in DATASET_LIST:
        raise ValueError("%s is not in DATASET_LIST." % args.dataset)

    run_reliability_diagrams(args)

    # for DATASET in DATASET_LIST:
    # run_reliability_diagrams(DATASET, PRIORTYPE)
    # run_calibration_error(DATASET, PRIORTYPE, NUM_RUNS)
    # run_true_calibration_error(DATASET)
    # results = Parallel(n_jobs=num_cores)(delayed(run_calibration_error(DATASET, PRIORTYPE, NUM_RUNS))
    #                                      for DATASET in DATASET_LIST)
