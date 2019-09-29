from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from active_utils import prepare_data, thompson_sampling, top_two_thompson_sampling, random_sampling
from bbutils.utils import *
from tqdm import tqdm
from utils import BetaBernoulli
%matplotlib inline

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875
DPI = 300
FONT_SIZE = 8

mpl.rcParams['font.size'] = FONT_SIZE
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times']
# mpl.rcParams['text.usetex'] = True

mpl.rcParams['text.usetex'] = False
from bbutils import BetaBernoulli




def estimate_overall_accuracy(categories: List[int], observations: List[bool], confidences: List[float],
                num_classes: int, n: int, sample_method: str, mode: str, metric: str, prior=None, ttts_beta=0.5,
                max_ttts_trial=50, random_seed=0) -> Tuple[np.ndarray, np.ndarray]:
    # prepare model, deques, thetas, choices
    random.seed(random_seed)
    model = BetaBernoulli(num_classes, prior)
    deques = [deque() for _ in range(num_classes)]
    for category, observation in zip(categories, observations):
        deques[category].append(observation)
    for _deque in deques:
        random.shuffle(_deque)
    success = np.zeros((n,))

    ground_truth = get_ground_truth(categories, observations, confidences, num_classes, metric, mode)
    confidence_k = _get_confidence_k(categories, confidences, num_classes)

    for i in range(n):
    # get sample
    if sample_method == "ts":
        category = thompson_sampling(model, deques, mode, metric, confidence_k)
    elif sample_method == "random":
        category = random_sampling(deques)
    elif sample_method == "ttts":
        category = top_two_thompson_sampling(model, deques, mode, metric, confidence_k, max_ttts_trial, ttts_beta)

    # update model, deques, thetas, choices
    model.update(category, deques[category].pop())

    if metric == "accuracy":
        metric_val = model.theta
    elif metric == 'calibration_bias':
        metric_val = confidence_k - model.theta
    if mode == 'min':
        success[i] = (np.argmin(metric_val) == ground_truth) * 1.0
    elif mode == 'max':
        success[i] = (np.argmax(metric_val) == ground_truth) * 1.0


def main(RUNS, DATASET):
    if DATASET == 'cifar100':
        datafile = '../data/cifar100/predictions.txt'
        FOUR_COLUMN = True  # format of input
        NUM_CLASSES = 100
    elif DATASET == 'svhn':
        datafile = '../data/svhn/svhn_predictions.txt'
        FOUR_COLUMN = False  # format of input
        NUM_CLASSES = 10
    elif DATASET == 'imagenet':
        datafile = '../data/imagenet/resnet152_imagenet_outputs.txt'
        NUM_CLASSES = 1000
        FOUR_COLUMN = False
    elif DATASET == 'imagenet2_topimages':
        datafile = '../data/imagenet/resnet152_imagenetv2_topimages_outputs.txt'
        NUM_CLASSES = 1000
        FOUR_COLUMN = False

    PRIOR = np.ones((NUM_CLASSES, 2))
    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile, FOUR_COLUMN)
    N = len(observations)

    overall_accuracy_random = np.zeros((RUNS, N))
    overall_accuracy_ts = np.zeros((RUNS, N))
    overall_accuracy_ttts = np.zeros((RUNS, N))

    for r in range(RUNS):
        overall_accuracy_random[r:] += estimate_overall_accuracy(categories,
                                      observations,
                                      confidences,
                                      NUM_CLASSES, N,
                                      sample_method='random',
                                      mode=MODE,
                                      metric=METRIC,
                                      prior=PRIOR,
                                      random_seed=r)
        overall_accuracy_ts[r:] += estimate_overall_accuracy(categories,
                                         observations,
                                         confidences,
                                         NUM_CLASSES,
                                         N,
                                         sample_method='ts',
                                         mode=MODE,
                                         metric=METRIC,
                                         prior=PRIOR, ttts_beta=TTTS_BETA,
                                         max_ttts_trial=MAX_TTTS_TRIAL,
                                         random_seed=r)
        overall_accuracy_ttts += estimate_overall_accuracy(categories,
                                           observations,
                                           confidences,
                                           NUM_CLASSES,
                                           N,
                                           sample_method='ttts',
                                           mode=MODE,
                                           metric=METRIC,
                                           prior=PRIOR, ttts_beta=TTTS_BETA,
                                           max_ttts_trial=MAX_TTTS_TRIAL,
                                           random_seed=r)

    success_rate_dict = {
        'random': random_success / RUNS,
        'TS': active_ts_success / RUNS,
        'TTTS': active_ttts_success / RUNS,
    }
    print(success_rate_dict)
    output_name = "../output/active_learning/%s_%s_%s_runs_%d.pkl" % (DATASET, METRIC, MODE, RUNS)
    pickle.dump(success_rate_dict, open(output_name, "wb" ) )

    # evaluation
    figname = "../figures/active_learning/%s_%s_%s_runs_%d.pdf" % (DATASET, METRIC, MODE, RUNS)
    comparison_plot(success_rate_dict, figname)


if __name__ == "__main__":

    # configs
    RUNS = 100
    DATASET = 'cifar100'  # 'cifar100', 'svhn', 'imagenet', 'imagenet2_topimages
    main(RUNS, DATASET)
