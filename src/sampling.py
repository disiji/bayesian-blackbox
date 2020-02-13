import random
from collections import deque
from typing import List, Union

import numpy as np

from models import BetaBernoulli


def random_sampling(deques: List[deque], topk: int = 1, **kwargs) -> Union[int, List[int]]:
    """
    Draw topk samples with random sampling.
    :param deques: List[deque]
        A list of deques, each contains a deque of samples from one predicted class.
    :param topk: int
        The number of extreme classes to identify. Default: 1.
    :param kwargs:
    :return: Union[int, List[int]]
        A list of index if topk > 1 and topk < number of non-empty deques; else return one index.
    """
    while True:
        # select each class randomly
        if topk == 1:
            category = random.randrange(len(deques))
            if len(deques[category]) != 0:
                return category
        else:
            # return a list of randomly selected categories:
            candidates = set([i for i in range(len(deques)) if len(deques[i]) > 0])
            if len(candidates) < topk:
                return random_sampling(deques, topk=1)
            else:  # there are less than topk available arms to play
                return random.sample(candidates, topk)


def thompson_sampling(deques: List[deque],
                      model: BetaBernoulli,
                      mode: str,
                      topk: int = 1,
                      **kwargs) -> Union[int, List[int]]:
    """
    Draw topk samples with Thompson sampling.
    :param deques: List[deque]
        A list of deques, each contains a deque of samples from one predicted class.
    :param model: BetaBernoulli
        A model for classwise accuracy.
    :param mode: str
        'min' or 'max'
    :param topk: int
        The number of extreme classes to identify. Default: 1.
    :param kwargs:
    :return: Union[int, List[int]]
        A list of index if topk > 1 and topk < number of non-empty deques; else return one index.
    """
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
                              **kwargs) -> Union[int, List[int]]:
    """
    Draw topk samples with Top Two Thompson sampling.
        Russo, D.  Simple Bayesian algorithms for best arm iden-tification. InConference on Learning Theory,
            pp. 1417–1418, 2016.
    :param deques: List[deque]
        A list of deques, each contains a deque of samples from one predicted class.
    :param model: BetaBernoulli
        A model for classwise accuracy.
    :param mode: str
        'min' or 'max'
    :param max_ttts_trial: int
        The number of trials to draw a different arm. Default: 50.
    :param ttts_beta: float
        Between 0 and 1. The probability to play the best arm without further exploration.
    :param kwargs:
    :return: Union[int, List[int]]
        A list of index if topk > 1 and topk < number of non-empty deques; else return one index.
    """
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
                   topk: int = 1,
                   epsilon: float = 0.1,
                   **kwargs) -> Union[int, List[int]]:
    """
    Draw topk samples with epsilon greedy.
    :param deques: List[deque]
        A list of deques, each contains a deque of samples from one predicted class.
    :param model: BetaBernoulli
        A model for classwise accuracy.
    :param mode: str
        'min' or 'max'
    :param topk: int
        The number of extreme classes to identify. Default: 1.
    :param epsilon: float
        The probability to explore at each time step.
    :param kwargs:
    :return: Union[int, List[int]]
        A list of index if topk > 1 and topk < number of non-empty deques; else return one index.
    """
    if random.random() < epsilon:
        return random_sampling(deques, topk)
    else:
        samples = model.eval
        if mode == 'max':
            ranked = np.argsort(samples)[::-1]
        elif mode == 'min':
            ranked = np.argsort(samples)

        if topk == 1:
            for j in range(len(deques)):
                category = ranked[j]
                if len(deques[category]) != 0:
                    return category
        else:
            categories_list = []
            candidates = set([i for i in range(len(deques)) if len(deques[i]) > 0])
            # when we go through 'ranked' and len(categories_list) < topk, topk sampling is reduced to top 1
            if len(candidates) < topk:
                return epsilon_greedy(deques, model, mode, topk=1)
            else:
                for category in ranked:
                    if category in candidates:
                        categories_list.append(category)
                        if len(categories_list) == topk:
                            return categories_list


def bayesian_UCB(deques: List[deque],
                 model: BetaBernoulli,
                 mode: str,
                 topk: int = 1,
                 ucb_c: int = 1,
                 **kwargs) -> Union[int, List[int]]:
    """
    Draw topk samples with Bayesian Upper Confidence Bounds (UCB).
    :param deques: List[deque]
        A list of deques, each contains a deque of samples from one predicted class.
    :param model: BetaBernoulli
        A model for classwise accuracy.
    :param mode: str
        'min' or 'max'
    :param topk: int
        The number of extreme classes to identify. Default: 1.
    :param ucb_c: float
        How many standard dev to consider as upper confidence bound. Default: 1.
    :param kwargs:
    :return: Union[int, List[int]]
        A list of index if topk > 1 and topk < number of non-empty deques; else return one index.
    """
    metric_val = model.eval
    if mode == 'max':
        metric_val += ucb_c * model.variance
        ranked = np.argsort(metric_val)[::-1]
    elif mode == 'min':
        metric_val -= ucb_c * model.variance
        ranked = np.argsort(metric_val)

    if topk == 1:
        for j in range(len(deques)):
            category = ranked[j]
            if len(deques[category]) != 0:
                return category
    else:
        categories_list = []
        candidates = set([i for i in range(len(deques)) if len(deques[i]) > 0])
        # when we go through 'ranked' and len(categories_list) < topk, topk sampling is reduced to top 1
        if len(candidates) < topk:
            return bayesian_UCB(deques, model, mode, topk=1)
        else:
            for category in ranked:
                if category in candidates:
                    categories_list.append(category)
                    if len(categories_list) == topk:
                        return categories_list


SAMPLE_CATEGORY = {
    'random': random_sampling,
    'ts': thompson_sampling,
    'ttts': top_two_thompson_sampling,
    'epsilon_greedy': epsilon_greedy,
    'bayesian_ucb': bayesian_UCB
}
