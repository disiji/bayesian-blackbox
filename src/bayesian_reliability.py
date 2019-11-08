import csv
import multiprocessing
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

num_cores = multiprocessing.cpu_count()


def BetaParams(mu, var):
    """
    Compute parameters of Beta distribution given mean and variance.
    """
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
    beta = alpha * (1 / mu - 1)
    return alpha, beta


def prepare_data(filename, N, random_seed=1):
    np.random.seed(random_seed)
    data = np.genfromtxt(filename)
    np.random.shuffle(data)
    data = data[0:N]
    probabilities = data[:, 1:]
    confidence = np.max(probabilities, axis=1)
    Y_predict = np.argmax(probabilities, axis=1)
    Y_true = data[:, 0]
    return confidence, Y_predict, Y_true


def get_ground_truth(filename):
    data = np.genfromtxt(filename)
    Y_true = data[:, 0]
    probabilities = data[:, 1:]
    confidence = np.max(probabilities, axis=1)
    Y_predict = np.argmax(probabilities, axis=1)
    bins = np.linspace(0, 1, NUM_BINS + 1)
    digitized = np.digitize(confidence, bins[1:-1])

    accuracy_bins = np.array([(Y_predict[digitized == i] == Y_true[digitized == i]).mean()
                              for i in range(NUM_BINS)])
    accuracy_bins[np.isnan(accuracy_bins)] = 0.0
    density_bins = np.array([(digitized == i).mean() for i in range(0, NUM_BINS)])
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


def bayesian_assessment(confidence, Y_predict, Y_true, prior_type, VAR=None, pseudocount=None):
    bins = np.linspace(0, 1, NUM_BINS + 1)
    digitized = np.digitize(confidence, bins[1:-1])
    density_bins = [(digitized == i).mean() for i in range(0, NUM_BINS)]
    diagonal_bins = [(i + 0.5) / NUM_BINS for i in range(0, NUM_BINS)]

    # compute prior, update prior
    if prior_type == 'fixed_var':
        beta_priors = np.array([BetaParams((i + 0.5) / NUM_BINS, VAR) for i in range(NUM_BINS)])
    elif prior_type == 'pseudocount':
        beta_priors = np.array([[(i + 0.5) * pseudocount / 10, (9.5 - i) * pseudocount / 10] for i in range(NUM_BINS)])

    counts = np.array([(((Y_predict[digitized == i]) == (Y_true[digitized == i])).sum(),
                        ((Y_predict[digitized == i]) != (Y_true[digitized == i])).sum())
                       for i in range(NUM_BINS)])
    empirical_accuracy = counts[:, 0] / (counts[:, 0] + counts[:, 1])
    empirical_accuracy[np.isnan(empirical_accuracy)] = 0
    beta_posteriors = beta_priors + counts

    # compute mean and credible interval of posterior
    beta_posteriors_mean = np.array([beta_posteriors[i, 0] / (beta_posteriors[i].sum())
                                     for i in range(NUM_BINS)])
    beta_posterior_p025 = np.array([stats.beta.ppf(0.025, beta_posteriors[i, 0], beta_posteriors[i, 1])
                                    for i in range(NUM_BINS)])
    beta_posterior_p975 = np.array([stats.beta.ppf(0.975, beta_posteriors[i, 0], beta_posteriors[i, 1])
                                    for i in range(NUM_BINS)])
    return {
        'diagonal_bins': np.array(diagonal_bins),
        'density_bins': np.array(density_bins),
        'empirical_accuracy': empirical_accuracy,
        'beta_posteriors_mean': beta_posteriors_mean,
        'beta_posterior_p025': beta_posterior_p025,
        'beta_posterior_p975': beta_posterior_p975,
    }


def bayesian_reliability_diagram(confidence, Y_predict, Y_true, prior_type, figname, VAR=None,
                                 pseudocount=None) -> None:
    output = bayesian_assessment(confidence, Y_predict, Y_true, prior_type, VAR, pseudocount)

    fig, ax1 = plt.subplots(figsize=(4.3, 3))
    color = 'tab:red'
    # ax1.grid(True)
    ax1.scatter([i + 0.5 for i in range(NUM_BINS)], output['empirical_accuracy'], label="Frequentist", marker="^",
                s=100)
    ax1.plot([i + 0.5 for i in range(NUM_BINS)], output['beta_posteriors_mean'], c="r", linestyle="--")
    ax1.errorbar([i + 0.5 for i in range(NUM_BINS)],
                 output['beta_posteriors_mean'],
                 yerr=(output['beta_posteriors_mean'] - output['beta_posterior_p025'],
                       output['beta_posterior_p975'] - output['beta_posteriors_mean']),
                 fmt='o', color='r', label="Bayesian")
    ax1.plot(np.linspace(0, 1, 11), linestyle="--", linewidth=3, c="gray")
    ax1.fill_between([i + 0.5 for i in range(NUM_BINS)], output['beta_posteriors_mean'], \
                     np.linspace(0, 1, 11)[:-1] + 0.05, color="gray", alpha=0.3)
    # ax1.legend(loc='upper left', prop={'size': 10})
    ax1.set_xlim((0.0, NUM_BINS))
    ax1.set_xlabel("Score(Model Confidence)", fontsize=14)
    ax1.set_xticks(range(NUM_BINS + 1))
    ax1.set_xticklabels(["%.1f" % i for i in np.linspace(0, 1, NUM_BINS + 1)])
    ax1.set_ylim((0.0, 1.0))
    ax1.set_ylabel("Estimated Accuracy", fontsize=14)

    # add histogram to the reliability diagram
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.bar([i + 0.5 for i in range(NUM_BINS)], output['density_bins'], color=color, alpha=0.5, label="Histogram",
            width=1.0)
    ax2.set_ylabel('Histogram', color=color, fontsize=12)
    ax2.set_ylim((0.0, 2.0))
    ax2.set_yticks([0, 1.0])
    ax2.set_yticklabels([0, 1.0], color=color)
    ax2.yaxis.set_label_coords(1.01, 0.25)
    plt.tight_layout()
    plt.savefig(figname)


def compute_estimation_error(datafile, N_list, num_runs, prior_type, VAR=None, pseudocount=None):
    ground_truth = get_ground_truth(datafile)

    print(datafile, prior_type, VAR, pseudocount, "...")

    weighted_pool_bayesian_estimation_error = np.zeros(shape=(num_runs, len(N_list)))
    weighted_pool_frequentist_estimation_error = np.zeros(shape=(num_runs, len(N_list)))
    weighted_online_bayesian_estimation_error = np.zeros(shape=(num_runs, len(N_list)))
    weighted_online_frequentist_estimation_error = np.zeros(shape=(num_runs, len(N_list)))
    unweighted_bayesian_estimation_error = np.zeros(shape=(num_runs, len(N_list)))
    unweighted_frequentist_estimation_error = np.zeros(shape=(num_runs, len(N_list)))
    pool_bayesian_ece = np.zeros(shape=(num_runs, len(N_list)))
    pool_frequentist_ece = np.zeros(shape=(num_runs, len(N_list)))
    online_bayesian_ece = np.zeros(shape=(num_runs, len(N_list)))
    online_frequentist_ece = np.zeros(shape=(num_runs, len(N_list)))
    bayesian_mce = np.zeros(shape=(num_runs, len(N_list)))
    frequentist_mce = np.zeros(shape=(num_runs, len(N_list)))

    for run_idx in range(num_runs):
        for i, N in enumerate(N_list):
            confidence, Y_predict, Y_true = prepare_data(datafile, N, random_seed=run_idx)
            output = bayesian_assessment(confidence, Y_predict, Y_true, prior_type, VAR, pseudocount)

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

            # print("=================", N)
            # print("ground_truth['accuracy_bins']:", ground_truth['accuracy_bins'])
            # print("ground_truth['density_bins']:", ground_truth['density_bins'])
            # print("output['density_bins']:", output['density_bins'])
            # print("output['beta_posteriors_mean']:", output['beta_posteriors_mean'])
            # print("output['diagonal_bins']:", output['diagonal_bins'])
            # print("bayesian_calibration_bias:", bayesian_calibration_bias)
            # print("frequentist_calibration_bias:", frequentist_calibration_bias)

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


def run_calibration_error(DATASET, PRIORTYPE, NUM_RUNS):
    PSEUDOCOUNT = [0.1, 1, 10]
    N_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    # N_list = [100, 200, 2000, 10000]
    output_dir = "../output/accuracy_estimation_error/"

    if DATASET == "cifar100":  # 10,000
        datafile = "../data/cifar100/cifar100_predictions_dropout.txt"
    elif DATASET == 'imagenet':  # 50,000
        datafile = '../data/imagenet/resnet152_imagenet_outputs.txt'
    elif DATASET == 'imagenet2_topimages':  # 10,000
        datafile = '../data/imagenet/resnet152_imagenetv2_topimages_outputs.txt'
    elif DATASET == '20newsgroup':  # 5607
        datafile = "../data/20newsgroup/bert_20_newsgroups_outputs.txt"
        N_list = N_list[:-1]
    elif DATASET == 'svhn':  # 26032
        datafile = '../data/svhn/svhn_predictions.txt'
    elif DATASET == 'dbpedia':  # 70000
        datafile = '../data/dbpedia/bert_dbpedia_outputs.txt'

    if PRIORTYPE == 'fixed_var':
        pass

    elif PRIORTYPE == 'pseudocount':
        for pseudo_n in PSEUDOCOUNT:
            estimation_error_output = compute_estimation_error(datafile, N_list, NUM_RUNS, PRIORTYPE,
                                                               pseudocount=pseudo_n)
            print(estimation_error_output)

            ## weighted estimation error, weight of each bin is estiamted by pooling all unlabeled data
            np.savetxt(output_dir + "weighted_pool_error_%s_PseudoCount%.1f_runs%d_bayesian.csv" % (
                DATASET, pseudo_n, NUM_RUNS),
                       estimation_error_output['weighted_pool_bayesian_estimation_error'],
                       delimiter=',')
            np.savetxt(output_dir + "weighted_pool_error_%s_PseudoCount%.1f_runs%d_frequentist.csv" % (
                DATASET, pseudo_n, NUM_RUNS),
                       estimation_error_output['weighted_pool_frequentist_estimation_error'], delimiter=',')

            ## weighted estimation error, weight of each bin is estiamted with observed labeled data
            np.savetxt(output_dir + "weighted_online_error_%s_PseudoCount%.1f_runs%d_bayesian.csv" % (
                DATASET, pseudo_n, NUM_RUNS),
                       estimation_error_output['weighted_online_bayesian_estimation_error'],
                       delimiter=',')
            np.savetxt(output_dir + "weighted_online_error_%s_PseudoCount%.1f_runs%d_frequentist.csv" % (
                DATASET, pseudo_n, NUM_RUNS),
                       estimation_error_output['weighted_online_frequentist_estimation_error'], delimiter=',')

            ## unweighted estimation error
            np.savetxt(output_dir + "unweighted_error_%s_PseudoCount%.1f_runs%d_bayesian.csv" % (
                DATASET, pseudo_n, NUM_RUNS),
                       estimation_error_output['unweighted_bayesian_estimation_error'],
                       delimiter=',')
            np.savetxt(output_dir + "unweighted_error_%s_PseudoCount%.1f_runs%d_frequentist.csv" % (
                DATASET, pseudo_n, NUM_RUNS),
                       estimation_error_output['unweighted_frequentist_estimation_error'],
                       delimiter=',')

            ## ece, weight of each bin is estiamted by pooling all unlabeled data
            np.savetxt(output_dir + "pool_ece_%s_PseudoCount%.1f_runs%d_bayesian.csv" % (DATASET, pseudo_n, NUM_RUNS),
                       estimation_error_output['pool_bayesian_ece'],
                       delimiter=',')
            np.savetxt(
                output_dir + "pool_ece_%s_PseudoCount%.1f_runs%d_frequentist.csv" % (DATASET, pseudo_n, NUM_RUNS),
                estimation_error_output['pool_frequentist_ece'],
                delimiter=',')

            ## ece, weight of each bin is estiamted with observed labeled data
            np.savetxt(output_dir + "online_ece_%s_PseudoCount%.1f_runs%d_bayesian.csv" % (DATASET, pseudo_n, NUM_RUNS),
                       estimation_error_output['online_bayesian_ece'],
                       delimiter=',')
            np.savetxt(
                output_dir + "online_ece_%s_PseudoCount%.1f_runs%d_frequentist.csv" % (DATASET, pseudo_n, NUM_RUNS),
                estimation_error_output['online_frequentist_ece'],
                delimiter=',')

            ## mce
            np.savetxt(output_dir + "mce_%s_PseudoCount%.1f_runs%d_bayesian.csv" % (DATASET, pseudo_n, NUM_RUNS),
                       estimation_error_output['bayesian_mce'],
                       delimiter=',')
            np.savetxt(
                output_dir + "mce_%s_PseudoCount%.1f_runs%d_frequentist.csv" % (DATASET, pseudo_n, NUM_RUNS),
                estimation_error_output['frequentist_mce'],
                delimiter=',')


def run_true_calibration_error(DATASET):
    if DATASET == "cifar100":  # 10,000
        datafile = "../data/cifar100/cifar100_predictions_dropout.txt"
    elif DATASET == 'imagenet':  # 50,000
        datafile = '../data/imagenet/resnet152_imagenet_outputs.txt'
    elif DATASET == 'imagenet2_topimages':  # 10,000
        datafile = '../data/imagenet/resnet152_imagenetv2_topimages_outputs.txt'
    elif DATASET == '20newsgroup':  # 5607
        datafile = "../data/20newsgroup/bert_20_newsgroups_outputs.txt"
    elif DATASET == 'svhn':
        datafile = '../data/svhn/svhn_predictions.txt'
    elif DATASET == 'dbpedia':
        datafile = '../data/dbpedia/bert_dbpedia_outputs.txt'

    results = get_ground_truth(datafile)

    calibration_error_ground_truth_filename = "../output/calibration_error_ground_truth.csv"
    if not os.path.exists(calibration_error_ground_truth_filename):
        with open(calibration_error_ground_truth_filename, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(['dataset', 'weighted', 'calibration_error'])

    with open(calibration_error_ground_truth_filename, 'a+') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows([[DATASET, 'weighted', results['weighted_calibration_error']],
                          [DATASET, 'unweighted', results['unweighted_calibration_error']]])


def run_reliability_diagrams(DATASET, PRIORTYPE):
    VAR_list = [0.01, 0.05, 0.10, 0.25]  # takes value from (0, 0.25); variance of beta prior
    PSEUDOCOUNT = [0.1, 1, 10]

    if DATASET == "cifar100":  # 10,000
        N_list = [100, 1000, 10000]
        datafile = "../data/cifar100/cifar100_predictions_dropout.txt"
    elif DATASET == 'imagenet':  # 50,000
        N_list = [100, 1000, 50000]
        datafile = '../data/imagenet/resnet152_imagenet_outputs.txt'
    elif DATASET == 'imagenet2_topimages':  # 10,000
        N_list = [100, 1000, 10000]
        datafile = '../data/imagenet/resnet152_imagenetv2_topimages_outputs.txt'
    elif DATASET == '20newsgroup':  # 5607
        N_list = [100, 1000, 5607]
        datafile = "../data/20newsgroup/bert_20_newsgroups_outputs.txt"
    elif DATASET == 'svhn':
        N_list = [100, 1000, 26032]
        datafile = '../data/svhn/svhn_predictions.txt'
    elif DATASET == 'dbpedia':
        N_list = [100, 1000, 70000]
        datafile = '../data/dbpedia/bert_dbpedia_outputs.txt'

    for N in N_list:
        confidence, Y_predict, Y_true = prepare_data(datafile, N)
        if PRIORTYPE == 'fixed_var':
            for VAR in VAR_list:
                figname = "../figures/reliability_diagram/reliability_plot_%s_N%d_VAR%.2f.pdf" % (DATASET, N, VAR)
                bayesian_reliability_diagram(confidence, Y_predict, Y_true, PRIORTYPE, figname, VAR=VAR)
        elif PRIORTYPE == 'pseudocount':
            for pseudo_n in PSEUDOCOUNT:
                figname = "../figures/reliability_diagram/reliability_plot_%s_N%d_PseudoCount%.1f.pdf" % (
                    DATASET, N, pseudo_n)
                bayesian_reliability_diagram(confidence, Y_predict, Y_true, PRIORTYPE, figname, pseudocount=pseudo_n)


if __name__ == "__main__":
    NUM_BINS = 10
    PRIORTYPE = 'pseudocount'  #
    NUM_RUNS = 100

    DATASET_LIST = ['imagenet', 'dbpedia', 'cifar100', '20newsgroup', 'svhn', 'imagenet2_topimages']
    # DATASET_LIST = ['cifar100']

    dataset = str(sys.argv[1])
    if dataset not in DATASET_LIST:
        raise ValueError("%s is not in DATASET_LIST." % dataset)

    run_calibration_error(dataset, PRIORTYPE, NUM_RUNS)

    # for DATASET in DATASET_LIST:
    # run_reliability_diagrams(DATASET, PRIORTYPE)
    # run_calibration_error(DATASET, PRIORTYPE, NUM_RUNS)
    # run_true_calibration_error(DATASET)
    # results = Parallel(n_jobs=num_cores)(delayed(run_calibration_error(DATASET, PRIORTYPE, NUM_RUNS))
    #                                      for DATASET in DATASET_LIST)
