import random
from collections import deque
from typing import List

import numpy as np
import pandas as pd
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


def eval_ece(confidences: List[float], observations: List[bool], num_bins=10):
    """

    :param confidences:
    :param observations:
    :param num_bins:
    :return:
    """
    confidences = np.array(confidences)
    observations = np.array(observations) * 1.0
    bins = np.linspace(0, 1, num_bins + 1)
    digitized = np.digitize(confidences, bins[1:-1])

    w = np.array([(digitized == i).sum() for i in range(num_bins)])
    w = w / sum(w)

    confidence_bins = np.array([confidences[digitized == i].mean() for i in range(num_bins)])
    accuracy_bins = np.array([observations[digitized == i].mean() for i in range(num_bins)])
    confidence_bins[np.isnan(confidence_bins)] = 0
    accuracy_bins[np.isnan(accuracy_bins)] = 0
    diff = np.absolute(confidence_bins - accuracy_bins)
    ece = np.inner(diff, w)
    return ece


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
    accuracy_k = np.array([df[(df['Predicted'] == class_idx)]['Observations'].mean()
                           for class_idx in range(num_classes)])
    return accuracy_k


def _get_ece_k(categories: List[int], observations: List[bool], confidences: List[float], num_classes: int,
               num_bins=10) -> np.ndarray:
    """

    :param categories:
    :param observations:
    :param confidences:
    :param num_classes:
    :param num_bins:
    :return:
    """
    ece_k = np.zeros((num_classes,))

    for class_idx in range(num_classes):
        mask_idx = [i for i in range(len(observations)) if categories[i] == class_idx]
        observations_sublist = [observations[i] for i in mask_idx]
        confidences_sublist = [confidences[i] for i in mask_idx]
        ece_k[class_idx] = eval_ece(confidences_sublist, observations_sublist, num_bins)

    return ece_k


def _get_ground_truth(categories: List[int], observations: List[bool], confidences: List[float], num_classes: int,
                      metric: str, mode: str, topk: int = 1) -> int:
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
    elif metric == 'calibration_error':
        metric_val = _get_ece_k(categories, observations, confidences, num_classes, num_bins=10)

    output = np.zeros((num_classes,), dtype=np.bool_)

    if mode == 'max':
        indices = metric_val.argsort()[-topk:]
    else:
        indices = metric_val.argsort()[:topk]

    output[indices] = 1

    return output


def random_sampling(deques: List[deque], topk: int = 1, **kwargs) -> int:
    while True:
        # select each class randomly
        if topk == 1:
            category = random.randrange(len(deques))
            if len(deques[category]) != 0:
                return category
        else:
            # return a list of randomly selected categories:
            candidates = set([i for i in range(len(deques)) if len(deques[i]) > 0])
            if len(candidates) >= topk:
                return random.sample(candidates, topk)
            else:  # there are less than topk available arms to play
                return random_sampling(deques, topk= 1)


def thompson_sampling(deques: List[deque],
                      model: BetaBernoulli,
                      mode: str,
                      topk: int = 1,
                      **kwargs) -> int:
    samples = model.sample()
    if mode == 'max':
        ranked = np.argsort(samples)[::-1]
    elif mode == 'min':
        ranked = np.argsort(samples)
    if topk == 1:
        for category in ranked:
            if len(deques[category]) != 0:
                return category
    else:
        categories_list = []
        candidates = set([i for i in range(len(deques)) if len(deques[i]) > 0])
        # when we go through 'ranked' and len(categories_list) < topk, topk sampling is reduced to top 1
        if len(candidates) < topk:
            return thompson_sampling(deques, model, mode, topk=1)
        else:
            for category in ranked:
                if category in candidates:
                    categories_list.append(category)
                    if len(categories_list) == topk:
                        return categories_list


def top_two_thompson_sampling(deques: List[deque],
                              model: BetaBernoulli,
                              mode: str,
                              max_ttts_trial=50,
                              ttts_beta: float = 0.5,
                              **kwargs) -> int:
    category_1 = thompson_sampling(deques, model, mode)
    # toss a coin with probability beta
    B = np.random.binomial(1, ttts_beta)
    if B == 1:
        return category_1
    else:
        count = 0
        while True:
            category_2 = thompson_sampling(deques, model, mode)
            if category_2 != category_1:
                return category_2
            else:
                count += 1
                if count == max_ttts_trial:
                    return category_1


def epsilon_greedy(deques: List[deque],
                   model: BetaBernoulli,
                   mode: str,
                   epsilon: float = 0.1,
                   **kwargs) -> int:
    if random.random() < epsilon:
        return random_sampling(deques)
    else:
        samples = model.eval
        if mode == 'max':
            ranked = np.argsort(samples)[::-1]
        elif mode == 'min':
            ranked = np.argsort(samples)

        for j in range(len(deques)):
            category = ranked[j]
            if len(deques[category]) != 0:
                return category


def bayesian_UCB(deques: List[deque],
                 model: BetaBernoulli,
                 mode: str,
                 ucb_c: int = 1,
                 **kwargs) -> int:
    metric_val = model.eval
    if mode == 'max':
        metric_val += ucb_c * model.variance
        ranked = np.argsort(metric_val)[::-1]
    elif mode == 'min':
        metric_val -= ucb_c * model.variance
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
