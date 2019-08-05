import random
from collections import deque
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import BetaBernoulli

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
                              confidence_k: np.ndarray = None, beta: float = 0.5) -> int:
    category_1 = thompson_sampling(model, deques, mode, metric, confidence_k)
    # toss a coin with probability beta
    B = np.random.binomial(1, beta)
    if B == 1:
        return category_1
    else:
        while True:
            category_2 = thompson_sampling(model, deques, mode, metric, confidence_k)
            print(category_1, category_2)
            if category_2 != category_1:
                return category_2


def random_sampling(model: BetaBernoulli, deques: List[deque]) -> int:
    while True:
        # select each class randomly
        category = random.randrange(len(deques))
        if len(deques[category]) != 0:
            return category


def get_samples(categories: List[int], observations: List[bool], confidences: List[float], num_classes: int, n: int,
                sample_method: str, mode: str, metric: str, prior=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param num_classes:
    :param n:
    :param categories:
    :param observations:
    :param sample_method: "random", "ts", "ttts"
    :param mode: str, "min" or "max"
    :param metric: str, "accuracy" and "calibration bias"
    :return:
    """

    # prepare model, deques, thetas, choices
    model = BetaBernoulli(num_classes, prior)
    deques = [deque() for _ in range(num_classes)]
    for category, observation in zip(categories, observations):
        deques[category].append(observation)
    for _deque in deques:
        random.shuffle(_deque)
    choices = np.zeros((num_classes, n))
    thetas = np.zeros((num_classes, n))

    confidence_k = _get_confidence_k(categories, confidences, num_classes)

    for i in range(n):

        # get sample
        if sample_method == "ts":
            category = thompson_sampling(model, deques, mode, metric, confidence_k)
        elif sample_method == "random":
            category = random_sampling(model, deques)
        elif sample_method == "ttts":
            category = top_two_thompson_sampling(model, deques, mode, metric, confidence_k, beta=0.5)

        # update model, deques, thetas, choices
        model.update(category, deques[category].pop())
        thetas[:, i] = model._params[:, 0] / (model._params[:, 0] + model._params[:, 1])
        if i > 0:
            choices[:, i] = choices[:, i - 1]
        choices[category, i] += 1
    return choices, thetas


def comparison_plot(random_thetas: np.ndarray, active_thetas: np.ndarray, ground_truth: int, mode: str, metric: str,
                    num_runs: int, dataset: str, active_type: str) -> None:
    """

    :param random_thetas: (num_runs, num_classes, n)
    :param active_thetas: (num_runs, num_classes, n)
    :param ground_truth:
    :param mode:
    :param metric:
    :param num_runs:
    :param dataset:
    :param active_type:
    :return:
    """
    figname = "../figures/active_learning/%s_%s_%s_runs_%d_%s.pdf" % (dataset, metric, mode, num_runs, active_type)
    if mode == 'min':
        random_success = np.mean(np.argmin(random_thetas, axis=1) == ground_truth, axis=0)
        active_success = np.mean(np.argmin(active_thetas, axis=1) == ground_truth, axis=0)
    elif mode == 'max':
        random_success = np.mean(np.argmax(random_thetas, axis=1) == ground_truth, axis=0)
        active_success = np.mean(np.argmax(active_thetas, axis=1) == ground_truth, axis=0)

    # If labels are getting cut off make the figsize smaller
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO), dpi=300)
    plt.plot(active_success, label='active')
    plt.plot(random_success, label='random')
    plt.xlabel('Time')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')


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
    observations = [1 if _ is True else 0 for _ in observations]
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


if __name__ == "__main__":

    # configs
    RUNS = 2
    MODE = 'min'  # 'min' or 'max'
    METRIC = 'accuracy'  # 'accuracy' or 'calibration_bias'
    DATASET = 'cifar100'  # 'cifar100', 'svhn', 'imagenet', or 'imagenet2_topimages'
    ACTIVE_TYPE = 'ttts'  # 'ts' or 'ttts'

    if DATASET == 'cifar100':
        # datafile = "../data/cifar100/cifar100_predictions_dropout.txt"
        datafile = '../data/cifar100/predictions.txt'
        FOUR_COLUMN = True  # format of input
        NUM_CLASSES = 100
        PRIOR = np.ones((NUM_CLASSES, 2))
    elif DATASET == 'svhn':
        datafile = '../data/svhn/svhn_predictions.txt'
        FOUR_COLUMN = False  # format of input
        NUM_CLASSES = 10
        PRIOR = np.ones((NUM_CLASSES, 2))
    elif DATASET == 'imagenet':
        datafile = '../data/imagenet/resnet152_imagenet_outputs.txt'
        NUM_CLASSES = 1000
        FOUR_COLUMN = False
        PRIOR = np.ones((NUM_CLASSES, 2))
    elif DATASET == 'imagenet2_topimages':
        datafile = '../data/imagenet/resnet152_imagenetv2_topimages_outputs.txt'
        NUM_CLASSES = 1000
        FOUR_COLUMN = False
        PRIOR = np.ones((NUM_CLASSES, 2))

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile, FOUR_COLUMN)
    N = len(observations)

    # get samples for multiple runs
    active_choices = np.zeros((RUNS, NUM_CLASSES, N))
    active_thetas = np.zeros((RUNS, NUM_CLASSES, N))
    random_choices = np.zeros((RUNS, NUM_CLASSES, N))
    random_thetas = np.zeros((RUNS, NUM_CLASSES, N))
    for r in range(RUNS):
        active_choices[r, :, :], active_thetas[r, :, :] = get_samples(categories, observations, confidences,
                                                                      NUM_CLASSES, N,
                                                                      sample_method=ACTIVE_TYPE, mode=MODE,
                                                                      metric=METRIC,
                                                                      prior=PRIOR)
        random_choices[r, :, :], random_thetas[r, :, :] = get_samples(categories, observations, confidences,
                                                                      NUM_CLASSES, N,
                                                                      sample_method='random', mode=MODE, metric=METRIC,
                                                                      prior=PRIOR)

    # evaluation
    ground_truth = get_ground_truth(categories, observations, confidences, NUM_CLASSES, METRIC, MODE)
    comparison_plot(random_thetas, active_thetas, ground_truth, MODE, METRIC, RUNS, DATASET, ACTIVE_TYPE)
