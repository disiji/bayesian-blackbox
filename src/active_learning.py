import copy
import pickle
import random
from collections import deque
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from active_utils import prepare_data, SAMPLE_CATEGORY
from data_utils import datafile_dict, num_classes_dict
from models import BetaBernoulli

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875
DPI = 300
FONT_SIZE = 8


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


def _get_ground_truth(categories: List[int], observations: List[bool], confidences: List[float], num_classes: int,
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
        return np.argwhere(metric_val == np.amax(metric_val)).flatten().tolist()
    else:
        return np.argwhere(metric_val == np.amin(metric_val)).flatten().tolist()


def get_samples(categories: List[int],
                observations: List[bool],
                confidences: List[float],
                num_classes: int,
                num_samples: int,
                sample_method: str,
                mode: str,
                metric: str,
                prior=None,
                random_seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    # prepare model, deques, thetas, choices
    random.seed(random_seed)

    model = copy.deepcopy(BetaBernoulli(num_classes, prior))
    deques = [deque() for _ in range(num_classes)]
    for category, observation in zip(categories, observations):
        deques[category].append(observation)
    for _deque in deques:
        random.shuffle(_deque)

    ground_truth = _get_ground_truth(categories, observations, confidences, num_classes, metric, mode)
    confidence_k = _get_confidence_k(categories, confidences, num_classes)

    success = copy.deepcopy(np.zeros((num_samples,)))
    for i in range(num_samples):
        category = SAMPLE_CATEGORY[sample_method].__call__(deques=deques,
                                                           random_seed=random_seed,
                                                           model=model,
                                                           mode=mode,
                                                           metric=metric,
                                                           confidence_k=confidence_k,
                                                           max_ttts_trial=50,
                                                           ttts_beta=0.5,
                                                           epsilon=0.1,
                                                           ucb_c=1, )
        # update model, deques, thetas, choices
        model.update(category, deques[category].pop())

        if metric == "accuracy":
            metric_val = model.theta
        elif metric == 'calibration_bias':
            metric_val = confidence_k - model.theta
        if mode == 'min':
            success[i] = (np.argmin(metric_val) in ground_truth) * 1.0
        elif mode == 'max':
            success[i] = (np.argmax(metric_val) in ground_truth) * 1.0
    return success


def comparison_plot(success_rate_dict, figname) -> None:
    # If labels are getting cut off make the figsize smaller
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO), dpi=300)
    for method_name, success_rate in success_rate_dict.items():
        plt.plot(success_rate, label=method_name)
    plt.xlabel('Time')
    plt.ylabel('Success Rate')
    # plt.legend()
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.ylim(0.0, 1.0)
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')


def main(RUNS, MODE, METRIC, DATASET):
    if DATASET == 'cifar100':
        FOUR_COLUMN = True  # format of input
    else:
        FOUR_COLUMN = False  # format of input

    datafile = datafile_dict[DATASET]
    NUM_CLASSES = num_classes_dict[DATASET]

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile, FOUR_COLUMN)
    N = len(observations)

    UNIFORM_PRIOR = np.ones((NUM_CLASSES, 2)) / 2
    confidence = _get_confidence_k(categories, confidences, NUM_CLASSES)
    INFORMED_PRIOR = np.array([confidence, 1 - confidence]).T

    # get samples for multiple runs
    # returns one thing: success or not
    success_rate_dict = {
        'random': copy.deepcopy(np.zeros((N,))),
        'ts_uniform': copy.deepcopy(np.zeros((N,))),
        'ttts_uniform': copy.deepcopy(np.zeros((N,))),
        'ts_informed': copy.deepcopy(np.zeros((N,))),
        'ttts_informed': copy.deepcopy(np.zeros((N,))),
        'epsilon_greedy': copy.deepcopy(np.zeros((N,))),
        'bayesian_ucb': copy.deepcopy(np.zeros((N,))),
    }
    for r in range(RUNS):
        print(r, 'random')
        success_rate_dict['random'] += get_samples(categories,
                                                   observations,
                                                   confidences,
                                                   NUM_CLASSES,
                                                   N,
                                                   sample_method='random',
                                                   mode=MODE,
                                                   metric=METRIC,
                                                   prior=UNIFORM_PRIOR,
                                                   random_seed=r)
        print(r, 'ts_uniform')
        success_rate_dict['ts_uniform'] += get_samples(categories,
                                                       observations,
                                                       confidences,
                                                       NUM_CLASSES,
                                                       N,
                                                       sample_method='ts',
                                                       mode=MODE,
                                                       metric=METRIC,
                                                       prior=UNIFORM_PRIOR,
                                                       random_seed=r)
        print(r, 'ttts_uniform')
        success_rate_dict['ttts_uniform'] += get_samples(categories,
                                                         observations,
                                                         confidences,
                                                         NUM_CLASSES,
                                                         N,
                                                         sample_method='ttts',
                                                         mode=MODE,
                                                         metric=METRIC,
                                                         prior=UNIFORM_PRIOR,
                                                         random_seed=r)
        print(r, 'ts_informed')
        success_rate_dict['ts_informed'] += get_samples(categories,
                                                        observations,
                                                        confidences,
                                                        NUM_CLASSES,
                                                        N,
                                                        sample_method='ts',
                                                        mode=MODE,
                                                        metric=METRIC,
                                                        prior=INFORMED_PRIOR,
                                                        random_seed=r)
        print(r, 'ttts_informed')
        success_rate_dict['ttts_informed'] += get_samples(categories,
                                                          observations,
                                                          confidences,
                                                          NUM_CLASSES,
                                                          N,
                                                          sample_method='ttts',
                                                          mode=MODE,
                                                          metric=METRIC,
                                                          prior=INFORMED_PRIOR,
                                                          random_seed=r)
        print(r, 'epsilon_greedy')
        success_rate_dict['epsilon_greedy'] += get_samples(categories,
                                                           observations,
                                                           confidences,
                                                           NUM_CLASSES,
                                                           N,
                                                           sample_method='epsilon_greedy',
                                                           mode=MODE,
                                                           metric=METRIC,
                                                           prior=UNIFORM_PRIOR,
                                                           random_seed=r)
        print(r, 'bayesian_ucb')
        success_rate_dict['bayesian_ucb'] += get_samples(categories,
                                                         observations,
                                                         confidences,
                                                         NUM_CLASSES,
                                                         N,
                                                         sample_method='bayesian_ucb',
                                                         mode=MODE,
                                                         metric=METRIC,
                                                         prior=UNIFORM_PRIOR,
                                                         random_seed=r)

    for method in success_rate_dict:
        success_rate_dict[method] /= RUNS
        output_name = "../output/active_learning/%s_%s_%s_%s_runs_%d.pkl" % (DATASET, METRIC, MODE, method, RUNS)
        pickle.dump(success_rate_dict[method], open(output_name, "wb"))

    # evaluation
    figname = "../output/active_learning/%s_%s_%s_runs_%d.pdf" % (DATASET, METRIC, MODE, RUNS)
    comparison_plot(success_rate_dict, figname)


if __name__ == "__main__":

    RUNS = 100

    # for DATASET in ['cifar100', 'svhn', 'imagenet', 'imagenet2_topimages', '20newsgroup', 'dbpedia']:
    for DATASET in ['cifar100']:
        for METRIC in ['accuracy', 'calibration_bias']:
            for MODE in ['min', 'max']:
                print(DATASET, METRIC, MODE, '...')
                main(RUNS, MODE, METRIC, DATASET)
