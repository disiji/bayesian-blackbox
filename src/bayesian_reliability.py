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



def prepare_data(filename, N):
    data = np.genfromtxt(filename)
    np.random.shuffle(data)
    data = data[0:N]
    probabilities = data[:, 1:]
    confidence = np.max(probabilities, axis=1)
    Y_predict = np.argmax(probabilities, axis=1)
    Y_true = data[:, 0]
    return confidence, Y_predict, Y_true


def bayesian_reliability_diagram(confidence, Y_predict, Y_true, prior_type, figname, VAR=None, pseudocount=None) -> None:
    bins = np.linspace(0, 1, NUM_BINS + 1)
    digitized = np.digitize(confidence, bins[1:-1])
    accuracy_bins = [(Y_predict[digitized == i] == Y_true[digitized == i]).mean()
                     for i in range(NUM_BINS)]

    # compute prior, update prior
    if prior_type == 'fixed_var':
        BetaPriors = np.array([BetaParams((i + 0.5) / NUM_BINS, VAR) for i in range(NUM_BINS)])
    elif prior_type == 'pseudocount':
        BetaPriors = np.array([[(i + 0.5) * pseudocount / 10, (9.5 - i) * pseudocount / 10] for i in range(NUM_BINS)])

    counts = np.array([(((Y_predict[digitized == i]) == (Y_true[digitized == i])).sum(),
                        ((Y_predict[digitized == i]) != (Y_true[digitized == i])).sum())
                       for i in range(NUM_BINS)])
    BetaPosteriors = BetaPriors + counts

    # compute mean and credible interval of posterior
    BetaPosteriorsMean = np.array([BetaPosteriors[i, 0] / (BetaPosteriors[i].sum())
                                   for i in range(NUM_BINS)])
    BetaPosteriorP025 = np.array([stats.beta.ppf(0.025, BetaPosteriors[i, 0], BetaPosteriors[i, 1])
                                  for i in range(NUM_BINS)])
    BetaPosteriorP975 = np.array([stats.beta.ppf(0.975, BetaPosteriors[i, 0], BetaPosteriors[i, 1])
                                  for i in range(NUM_BINS)])

    plt.figure(figsize=(4, 4))
    plt.grid(True)
    plt.xticks(range(NUM_BINS + 1), ["%.1f" % i for i in bins])
    plt.scatter([i + 0.5 for i in range(NUM_BINS)], accuracy_bins, label="Frequentist", marker="^", s=100)
    plt.plot([i + 0.5 for i in range(NUM_BINS)], BetaPosteriorsMean, c="r", linestyle="--")
    plt.errorbar([i + 0.5 for i in range(NUM_BINS)],
                 BetaPosteriorsMean,
                 yerr=(BetaPosteriorsMean - BetaPosteriorP025, BetaPosteriorP975 - BetaPosteriorsMean),
                 fmt='o', color='r', label="Bayesian")
    plt.plot(np.linspace(0, 1, 11), linestyle="--", linewidth=3, c="gray")
    plt.fill_between([i + 0.5 for i in range(NUM_BINS)], BetaPosteriorsMean, \
                     np.linspace(0, 1, 11)[:-1] + 0.05, color="gray", alpha=0.3)
    plt.legend(loc='upper left')
    plt.ylim((0.0, 1.0))
    plt.xlim((0.0, NUM_BINS))
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.savefig(figname)


if __name__ == "__main__":
    NUM_BINS = 10
    VAR_list = [0.01, 0.05, 0.10, 0.25]  # takes value from (0, 0.25); variance of beta prior
    PSEUDOCOUNT = [0.1, 1, 10]
    N_list = [100, 1000, 10000]
    DATASET = 'imagenet2_topimages'  # cifar100, imagenet, imagenet2_topimages
    PRIORTYPE = 'pseudocount'  #

    if PRIORTYPE == 'fixed_var':
        for N in N_list:
            for VAR in VAR_list:
                if DATASET == 'cifar100':
                    datafile = "../data/cifar100/cifar100_predictions_dropout.txt"
                    figname = "../figures/reliability_diagram/reliability_plot_cifar100_N%d_VAR%.2f.pdf" % (N, VAR)
                elif DATASET == 'imagenet':
                    datafile = '../data/imagenet/resnet152_imagenet_outputs.txt'
                    figname = "../figures/reliability_diagram/reliability_plot_imagenet_N%d_VAR%.2f.pdf" % (N, VAR)
                elif DATASET == "imagenet2_topimages":
                    datafile = '../data/imagenet/resnet152_imagenetv2_topimages_outputs.txt'
                    figname = "../figures/reliability_diagram/reliability_plot_imagenetv2_topimages_N%d_VAR%.2f.pdf" % (N, VAR)
                confidence, Y_predict, Y_true = prepare_data(datafile, N)
                bayesian_reliability_diagram(confidence, Y_predict, Y_true, PRIORTYPE, figname, VAR=VAR)

    elif PRIORTYPE == 'pseudocount':
        for N in N_list:
            for pseudo_n in PSEUDOCOUNT:
                if DATASET == 'cifar100':
                    datafile = "../data/cifar100/cifar100_predictions_dropout.txt"
                    figname = "../figures/reliability_diagram/reliability_plot_cifar100_N%d_PseudoCount%.1f.pdf" % (N, pseudo_n)
                elif DATASET == 'imagenet':
                    datafile = '../data/imagenet/resnet152_imagenet_outputs.txt'
                    figname = "../figures/reliability_diagram/reliability_plot_imagenet_N%d_PseudoCount%.1f.pdf" % (N, pseudo_n)
                elif DATASET == "imagenet2_topimages":
                    datafile = '../data/imagenet/resnet152_imagenetv2_topimages_outputs.txt'
                    figname = "../figures/reliability_diagram/reliability_plot_imagenetv2_topimages_N%d_PseudoCount%.1f.pdf" % (N, pseudo_n)
                confidence, Y_predict, Y_true = prepare_data(datafile, N)
                bayesian_reliability_diagram(confidence, Y_predict, Y_true, PRIORTYPE, figname, pseudocount=pseudo_n)
