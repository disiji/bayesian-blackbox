import random
from collections import deque
from typing import List

import numpy as np
from models import BetaBernoulli


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


def random_sampling(deques: List[deque], **kwargs) -> int:
    while True:
        # select each class randomly
        category = random.randrange(len(deques))
        if len(deques[category]) != 0:
            return category


def thompson_sampling(deques: List[deque],
                      model: BetaBernoulli,
                      mode: str,
                      metric: str,
                      confidence_k: np.ndarray = None,
                      **kwargs) -> int:
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


def top_two_thompson_sampling(deques: List[deque],
                              model: BetaBernoulli,
                              mode: str,
                              metric: str,
                              confidence_k: np.ndarray = None,
                              max_ttts_trial=50,
                              ttts_beta: float = 0.5,
                              **kwargs) -> int:
    category_1 = thompson_sampling(deques, model, mode, metric, confidence_k)
    # toss a coin with probability beta
    B = np.random.binomial(1, ttts_beta)
    if B == 1:
        return category_1
    else:
        count = 0
        while True:
            category_2 = thompson_sampling(deques, model, mode, metric, confidence_k)
            if category_2 != category_1:
                return category_2
            else:
                count += 1
                if count == max_ttts_trial:
                    return category_1


def epsilon_greedy(deques: List[deque],
                   model: BetaBernoulli,
                   mode: str,
                   metric: str,
                   confidence_k: np.ndarray = None,
                   epsilon: float = 0.1,
                   **kwargs) -> int:
    if random.random() < epsilon:
        return random_sampling(deques)
    else:
        theta_hat = model.theta
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


def bayesian_UCB(deques: List[deque],
                 model: BetaBernoulli,
                 mode: str,
                 metric: str,
                 confidence_k: np.ndarray = None,
                 ucb_c: int = 1,
                 **kwargs) -> int:
    # get mean of metric_val
    if metric == 'accuracy':
        metric_val = model.theta
    elif metric == 'calibration_bias':
        metric_val = confidence_k - model.theta
    if mode == 'max':
        metric_val += ucb_c * model.get_variance()
        ranked = np.argsort(metric_val)[::-1]
    elif mode == 'min':
        metric_val -= ucb_c * model.get_variance()
        ranked = np.argsort(metric_val)
    for j in range(len(deques)):
        category = ranked[j]
        if len(deques[category]) != 0:
            return category


SAMPLE_CATEGORY = {
    'random': random_sampling,
    'ts': thompson_sampling,
    'ttts': top_two_thompson_sampling,
    'epsilon_greedy': epsilon_greedy,
    'bayesian_ucb': bayesian_UCB
}


