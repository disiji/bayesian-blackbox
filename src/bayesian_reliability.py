import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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
    accuracy_bins = [(Y_predict[digitized == i] == Y_true[digitized == i]).mean()
                     for i in range(NUM_BINS)]
    density_bins = [(digitized == i).mean() for i in range(0, NUM_BINS)]
    return {
        'accuracy_bins': np.array(accuracy_bins),
        'density_bins': np.array(density_bins),
    }


def bayesian_assessment(confidence, Y_predict, Y_true, prior_type, VAR=None, pseudocount=None):
    bins = np.linspace(0, 1, NUM_BINS + 1)
    digitized = np.digitize(confidence, bins[1:-1])
    accuracy_bins = [(Y_predict[digitized == i] == Y_true[digitized == i]).mean()
                     for i in range(NUM_BINS)]
    density_bins = [(digitized == i).mean() for i in range(0, NUM_BINS)]

    # compute prior, update prior
    if prior_type == 'fixed_var':
        beta_priors = np.array([BetaParams((i + 0.5) / NUM_BINS, VAR) for i in range(NUM_BINS)])
    elif prior_type == 'pseudocount':
        beta_priors = np.array([[(i + 0.5) * pseudocount / 10, (9.5 - i) * pseudocount / 10] for i in range(NUM_BINS)])

    counts = np.array([(((Y_predict[digitized == i]) == (Y_true[digitized == i])).sum(),
                        ((Y_predict[digitized == i]) != (Y_true[digitized == i])).sum())
                       for i in range(NUM_BINS)])
    empirical_accuracy = counts[:,0] / (counts[:,0] + counts[:, 1])
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
        'accuracy_bins': np.array(accuracy_bins),
        'density_bins': np.array(density_bins),
        'empirical_accuracy': empirical_accuracy,
        'beta_posteriors_mean': beta_posteriors_mean,
        'beta_posterior_p025': beta_posterior_p025,
        'beta_posterior_p975': beta_posterior_p975,
    }


def bayesian_reliability_diagram(confidence, Y_predict, Y_true, prior_type, figname, VAR=None,
                                 pseudocount=None) -> None:
    output = bayesian_assessment(confidence, Y_predict, Y_true, prior_type, VAR, pseudocount)

    fig, ax1 = plt.subplots(figsize=(4.3, 4))
    color = 'tab:red'
    # ax1.grid(True)
    ax1.scatter([i + 0.5 for i in range(NUM_BINS)], output['accuracy_bins'], label="Frequentist", marker="^", s=100)
    ax1.plot([i + 0.5 for i in range(NUM_BINS)], output['beta_posteriors_mean'], c="r", linestyle="--")
    ax1.errorbar([i + 0.5 for i in range(NUM_BINS)],
                 output['beta_posteriors_mean'],
                 yerr=(output['beta_posteriors_mean'] - output['beta_posterior_p025'],
                       output['beta_posterior_p975'] - output['beta_posteriors_mean']),
                 fmt='o', color='r', label="Bayesian")
    ax1.plot(np.linspace(0, 1, 11), linestyle="--", linewidth=3, c="gray")
    ax1.fill_between([i + 0.5 for i in range(NUM_BINS)], output['beta_posteriors_mean'], \
                     np.linspace(0, 1, 11)[:-1] + 0.05, color="gray", alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_xlim((0.0, NUM_BINS))
    ax1.set_xlabel("Score(Model Confidence)")
    ax1.set_xticks(range(NUM_BINS + 1))
    ax1.set_xticklabels(["%.1f" % i for i in np.linspace(0, 1, NUM_BINS + 1)])
    ax1.set_ylim((0.0, 1.0))
    ax1.set_ylabel("Estimated Accuracy")

    # add histogram to the reliability diagram
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.bar([i + 0.5 for i in range(NUM_BINS)], output['density_bins'], color=color, alpha=0.5, label="Histogram",
            width=1.0)
    ax2.set_ylabel('Histogram', color=color)
    ax2.set_ylim((0.0, 2.0))
    ax2.set_yticks([0, 1.0])
    ax2.set_yticklabels([0, 1.0], color=color)
    ax2.yaxis.set_label_coords(1.01, 0.25)
    plt.tight_layout()
    plt.savefig(figname)


def compute_estimation_error(datafile, figname, N_list, num_runs, prior_type, VAR=None, pseudocount=None,
                             weighted=False):
    ground_truth = get_ground_truth(datafile)
    bayesian_estimation_error = np.zeros(shape=(num_runs, len(N_list)))
    frequentist_estimation_error = np.zeros(shape=(num_runs, len(N_list)))
    print(datafile, prior_type, VAR, pseudocount, "...")

    for run_idx in range(num_runs):
        for i, N in enumerate(N_list):
            confidence, Y_predict, Y_true = prepare_data(datafile, N, random_seed=run_idx)
            output = bayesian_assessment(confidence, Y_predict, Y_true, prior_type, VAR, pseudocount)

            bayesian_bias = np.abs(ground_truth['accuracy_bins'] - output['beta_posteriors_mean'])
            frequentist_bias = np.abs(ground_truth['accuracy_bins'] - output['beta_posteriors_mean'])
            # empty bins
            bayesian_bias[np.isnan(ground_truth['accuracy_bins'])] = 0.0
            frequentist_bias[np.isnan(ground_truth['accuracy_bins'])] = 0.0

            if weighted:
                bayesian_estimation_error[run_idx, i] = np.dot(bayesian_bias, ground_truth['density_bins'])
                frequentist_estimation_error[run_idx, i] = np.dot(frequentist_bias, ground_truth['density_bins'])
            else:
                bayesian_estimation_error[run_idx, i] = bayesian_bias.mean()
                frequentist_estimation_error[run_idx, i] = frequentist_bias.mean()

    fig, ax = plt.subplots(figsize=(4.3, 4))
    ax.errorbar(N_list, bayesian_estimation_error.mean(axis=0), bayesian_estimation_error.std(axis=0), linestyle='None',
                marker='^', label='Bayesian')
    ax.errorbar(N_list, frequentist_estimation_error.mean(axis=0), frequentist_estimation_error.std(axis=0),
                linestyle='None', marker='*', label='Frequentist')
    plt.tight_layout()
    plt.legend()
    plt.savefig(figname)
    plt.close()
    return {"bayesian_estimation_error": bayesian_estimation_error,
            "frequentist_estimation_error": frequentist_estimation_error, }


def run_calibration_error(DATASET, PRIORTYPE, NUM_RUNS):
    VAR_list = [0.01, 0.05, 0.10, 0.25]  # takes value from (0, 0.25); variance of beta prior
    PSEUDOCOUNT = [0.1, 1, 10]

    if DATASET == "cifar100":  # 10,000
        N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                  10000]
        datafile = "../data/cifar100/cifar100_predictions_dropout.txt"
    elif DATASET == 'imagenet':  # 50,000
        N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                  10000, 20000, 30000, 40000, 50000]
        datafile = '../data/imagenet/resnet152_imagenet_outputs.txt'
    elif DATASET == 'imagenet2_topimages':  # 10,000
        N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                  10000]
        datafile = '../data/imagenet/resnet152_imagenetv2_topimages_outputs.txt'
    elif DATASET == '20newsgroup':  # 5607
        N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 5607]
        datafile = "../data/20newsgroup/20newsgroups_in_domain.txt"
    elif DATASET == 'svhn':
        N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                  10000, 20000, 26032]
        datafile = '../data/svhn/svhn_predictions.txt'

    if PRIORTYPE == 'fixed_var':
        for VAR in VAR_list:
            figname = "../figures/accuracy_estimation_error/unweighted_error_%s_VAR%.2f_runs%d.pdf" % (
                DATASET, VAR, NUM_RUNS)
            bayesian_output_name = "../output/accuracy_estimation_error/unweighted_error_%s_VAR%.2f_runs%d_bayesian.csv" % (
                DATASET, VAR, NUM_RUNS)
            frequentist_output_name = "../output/accuracy_estimation_error/unweighted_error_%s_VAR%.2f_runs%d_frequentist.csv" % (
                DATASET, VAR, NUM_RUNS)
            estimation_error_output = compute_estimation_error(datafile, figname, N_list, NUM_RUNS, PRIORTYPE, VAR=VAR,
                                                         weighted=False)
            baysian_estimation_error = estimation_error_output['bayesian_estimation_error']
            frequentist_estimation_error = estimation_error_output['frequentist_estimation_error']
            np.savetxt(bayesian_output_name, baysian_estimation_error, delimiter=',')
            np.savetxt(frequentist_output_name, frequentist_estimation_error, delimiter=',')


            figname = "../figures/accuracy_estimation_error/weighted_error_%s_VAR%.2f_runs%d.pdf" % (
                DATASET, VAR, NUM_RUNS)
            bayesian_output_name = "../output/accuracy_estimation_error/weighted_error_%s_VAR%.2f_runs%d_bayesian.csv" % (
                DATASET, VAR, NUM_RUNS)
            frequentist_output_name = "../output/accuracy_estimation_error/weighted_error_%s_VAR%.2f_runs%d_frequentist.csv" % (
                DATASET, VAR, NUM_RUNS)
            estimation_error_output = compute_estimation_error(datafile, figname, N_list, NUM_RUNS, PRIORTYPE, VAR=VAR,
                                                         weighted=True)
            baysian_estimation_error = estimation_error_output['bayesian_estimation_error']
            frequentist_estimation_error = estimation_error_output['frequentist_estimation_error']
            np.savetxt(bayesian_output_name, baysian_estimation_error, delimiter=',')
            np.savetxt(frequentist_output_name, frequentist_estimation_error, delimiter=',')

    elif PRIORTYPE == 'pseudocount':
        for pseudo_n in PSEUDOCOUNT:
            figname = "../figures/accuracy_estimation_error/unweighted_error_%s_PseudoCount%.1f_runs%d.pdf" % (
                DATASET, pseudo_n, NUM_RUNS)
            bayesian_output_name = "../output/accuracy_estimation_error/unweighted_error_%s_PseudoCount%.1f_runs%d_bayesian.csv" % (
                DATASET, pseudo_n, NUM_RUNS)
            frequentist_output_name = "../output/accuracy_estimation_error/unweighted_error_%s_PseudoCount%.1f_runs%d_frequentist.csv" % (
                DATASET, pseudo_n, NUM_RUNS)
            estimation_error_output = compute_estimation_error(datafile, figname, N_list, NUM_RUNS, PRIORTYPE,
                                                         pseudocount=pseudo_n,
                                                         weighted=False)
            baysian_estimation_error = estimation_error_output['bayesian_estimation_error']
            frequentist_estimation_error = estimation_error_output['frequentist_estimation_error']
            np.savetxt(bayesian_output_name, baysian_estimation_error, delimiter=',')
            np.savetxt(frequentist_output_name, frequentist_estimation_error, delimiter=',')

            figname = "../figures/accuracy_estimation_error/weighted_error_%s_PseudoCount%.1f_runs%d.pdf" % (
                DATASET, pseudo_n, NUM_RUNS)
            bayesian_output_name = "../output/accuracy_estimation_error/weighted_error_%s_PseudoCount%.1f_runs%d_bayesian.csv" % (
                DATASET, pseudo_n, NUM_RUNS)
            frequentist_output_name = "../output/accuracy_estimation_error/weighted_error_%s_PseudoCount%.1f_runs%d_frequentist.csv" % (
                DATASET, pseudo_n, NUM_RUNS)
            estimation_error_output = compute_estimation_error(datafile, figname, N_list, NUM_RUNS, PRIORTYPE,
                                                         pseudocount=pseudo_n,
                                                         weighted=True)
            baysian_estimation_error = estimation_error_output['bayesian_estimation_error']
            frequentist_estimation_error = estimation_error_output['frequentist_estimation_error']
            np.savetxt(bayesian_output_name, baysian_estimation_error, delimiter=',')
            np.savetxt(frequentist_output_name, frequentist_estimation_error, delimiter=',')


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
        datafile = "../data/20newsgroup/20newsgroups_in_domain.txt"
    elif DATASET == 'svhn':  # 5607
        N_list = [100, 1000, 26032]
        datafile = '../data/svhn/svhn_predictions.txt'

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
    DATASET = 'cifar100'  # cifar100, imagenet, imagenet2_topimages, 20newsgroup, svhn
    PRIORTYPE = 'pseudocount'  #
    NUM_RUNS = 10

    #DATASET_LIST = ['cifar100', 'svhn', '20newsgroup', 'imagenet2_topimages', 'imagenet']
    DATASET_LIST = ['svhn']
    for DATASET in DATASET_LIST:
        run_reliability_diagrams(DATASET, PRIORTYPE)
        #run_calibration_error(DATASET, PRIORTYPE, NUM_RUNS)
