import random
from collections import deque
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import BetaBernoulli
import pickle

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875
DPI = 300
FONT_SIZE = 8


# this function is different from bayesian_reliabiitly.prepare_data
def prepare_data(filename, four_column=False):
    """

    :param filename: str
    :param four_column: indicates whether the dataformat is "index, correct class, predicted class, confidence"
                        or true label followed by a vector of scores for each class
    :return:
            categories: List[int], predicted class
            observations: List[bool], whether predicted class is the same as truth class
            confidence: List[float]
            idx2category: Dict[int, str] or None
            category2idx: Dict[str, int] or None

    """
    if four_column:
        # when file is in 4 column format: index, correct class, predicted class, confidence
        with open(filename, 'r') as f:
            category2idx = dict()
            idx2category = []
            categories = []
            observations = []
            confidences = []
            next(f)
            for line in f:
                _, correct, predicted, confidence = line.split()
                if predicted not in category2idx:
                    category2idx[predicted] = len(category2idx)
                    idx2category.append(predicted)
                idx = category2idx[predicted]
                categories.append(idx)
                observations.append(correct == predicted)
                confidences.append(float(confidence))

    else:
        data = np.genfromtxt(filename)
        categories = np.argmax(data[:, 1:], axis=1).astype(int)
        confidences = list(np.max(data[:, 1:], axis=1).astype(float))
        observations = list((categories == data[:, 0]))
        categories = list(categories)
        idx2category = None
        category2idx = None
        print("Accuracy: %.3f" % (len([_ for _ in observations if _ == True]) * 1.0 / len(observations)))
    return categories, observations, confidences, idx2category, category2idx


def thompson_sampling(model: BetaBernoulli, deques: List[deque], mode: str, metric: str,
                      confidence_k: np.ndarray = None) -> int:
    theta_hat = model.sample()
    if metric == 'accuracy':
        metric_val = theta_hat
    elif metric == 'calibration_bias':
        metric_val = confidence_k - theta_hat
    if mode == 'max':
        ranked = np.argsort(metric_val)[::-1]
    elif mode == 'min':
        ranked = np.argsort(metric_val)
    for j in range(len(deques)):
        category = ranked[j]
        if len(deques[category]) != 0:
            return category


def top_two_thompson_sampling(model: BetaBernoulli, deques: List[deque], mode: str, metric: str,
                              confidence_k: np.ndarray = None, max_ttts_trial=50, beta: float = 0.5) -> int:
    category_1 = thompson_sampling(model, deques, mode, metric, confidence_k)
    # toss a coin with probability beta
    B = np.random.binomial(1, beta)
    if B == 1:
        return category_1
    else:
        count = 0
        while True:
            category_2 = thompson_sampling(model, deques, mode, metric, confidence_k)
            if category_2 != category_1:
                return category_2
            else:
                count += 1
                if count == max_ttts_trial:
                    return category_1


def random_sampling(deques: List[deque]) -> int:
    while True:
        # select each class randomly
        category = random.randrange(len(deques))
        if len(deques[category]) != 0:
            return category


def get_samples(categories: List[int], observations: List[bool], confidences: List[float],
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

    return success


def _get_confidence_k(categories: List[int], confidences: List[float], num_classes: int) -> np.ndarray:
    """

    :param categories:
    :param confidences:
    :param num_classes:
    :return: confidence_k: (num_classes, )
    """
    df = pd.DataFrame(list(zip(categories, confidences)), columns=['Predicted', 'Confidence'])
    confidence_k = np.array([df[(df['Predicted'] == id)]['Confidence'].mean()
                             for id in range(num_classes)])
    return confidence_k


def _get_accuracy_k(categories: List[int], observations: List[bool], num_classes: int) -> np.ndarray:
    observations = np.array(observations) * 1.0
    df = pd.DataFrame(list(zip(categories, observations)), columns=['Predicted', 'Observations'])
    accuracy_k = np.array([df[(df['Predicted'] == id)]['Observations'].mean()
                           for id in range(num_classes)])
    return accuracy_k


def get_ground_truth(categories: List[int], observations: List[bool], confidences: List[float], num_classes: int,
                     metric: str, mode: str) -> int:
    """
    Compute ground truth given metric and mode with all data points.
    :param categories:
    :param observations:
    :param confidences:
    :param metric:
    :param mode:
    :return:
    """
    if metric == 'accuracy':
        metric_val = _get_accuracy_k(categories, observations, num_classes)
    elif metric == 'calibration_bias':
        accuracy_k = _get_accuracy_k(categories, observations, num_classes)
        confidence_k = _get_confidence_k(categories, confidences, num_classes)
        metric_val = confidence_k - accuracy_k
    if mode == 'max':
        return np.argmax(metric_val)
    else:
        return np.argmin(metric_val)


def comparison_plot(success_rate_dict, figname) -> None:
    # If labels are getting cut off make the figsize smaller
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO), dpi=300)
    for method_name, success_rate in success_rate_dict.items():
        plt.plot(success_rate, label=method_name)
    plt.xlabel('Time')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.ylim(0.0, 1.0)
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')


def main(RUNS, MODE, METRIC, DATASET, TTTS_BETA, MAX_TTTS_TRIAL):
    if DATASET == 'cifar100':
        # datafile = "../data/cifar100/cifar100_predictions_dropout.txt"
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

    # get samples for multiple runs
    # returns one thing: success or not
    active_ts_success = np.zeros((N,))
    active_ttts_success = np.zeros((N,))
    random_success = np.zeros((N,))
    for r in range(RUNS):
        random_success += get_samples(categories,
                                      observations,
                                      confidences,
                                      NUM_CLASSES, N,
                                      sample_method='random',
                                      mode=MODE,
                                      metric=METRIC,
                                      prior=PRIOR,
                                      random_seed=r)
        active_ts_success += get_samples(categories,
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
        active_ttts_success += get_samples(categories,
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
    # DATASET = 'cifar100'  # 'cifar100', 'svhn', 'imagenet', 'imagenet2_topimages
    MAX_TTTS_TRIAL = 50
    TTTS_BETA = 0.5
    # main(RUNS, MODE, METRIC, DATASET, ACTIVE_TYPE)

    for DATASET in ['cifar100', 'svhn', 'imagenet2_topimages', 'imagenet']:
        for METRIC in ['accuracy', 'calibration_bias']:
            for MODE in ['min', 'max']:
                print(DATASET, METRIC, MODE, '...')
                main(RUNS, MODE, METRIC, DATASET, TTTS_BETA, MAX_TTTS_TRIAL)
