import copy
import pickle
import random
import sys
from collections import deque
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from active_utils import prepare_data, SAMPLE_CATEGORY, eval_ece
from data_utils import datafile_dict, num_classes_dict, DATASET_LIST
from models import BetaBernoulli, ClasswiseEce

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
    elif metric == 'calibration_error':
        metric_val = _get_ece_k(categories, observations, confidences, num_classes, num_bins=10)
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
                weight=None,
                random_seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    # prepare model, deques, thetas, choices
    random.seed(random_seed)

    if metric == 'accuracy':
        model = copy.deepcopy(BetaBernoulli(num_classes, prior))
    elif metric == 'calibration_error':
        model = copy.deepcopy(ClasswiseEce(num_classes, num_bins=10, weight=weight, prior=None))

    deques = [deque() for _ in range(num_classes)]
    for (category, score, observation) in zip(categories, confidences, observations):
        if metric == 'accuracy':
            deques[category].append(observation)
        elif metric == 'calibration_error':
            deques[category].append((observation, score))
    for _deque in deques:
        random.shuffle(_deque)

    ground_truth = _get_ground_truth(categories, observations, confidences, num_classes, metric, mode)

    success = copy.deepcopy(np.zeros((num_samples,)))
    for i in range(num_samples):
        category = SAMPLE_CATEGORY[sample_method].__call__(deques=deques,
                                                           random_seed=random_seed,
                                                           model=model,
                                                           mode=mode,
                                                           max_ttts_trial=50,
                                                           ttts_beta=0.5,
                                                           epsilon=0.1,
                                                           ucb_c=1, )
        # update model, deques, thetas, choices
        if metric == 'accuracy':
            model.update(category, deques[category].pop())
        elif metric == 'calibration_error':
            observation, score = deques[category].pop()
            model.update(category, observation, score)

        metric_val = model.eval
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


def main_accuracy(RUNS, MODE, DATASET):
    datafile = datafile_dict[DATASET]
    NUM_CLASSES = num_classes_dict[DATASET]

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile, False)
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
                                                   metric='accuracy',
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
                                                       metric='accuracy',
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
                                                         metric='accuracy',
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
                                                        metric='accuracy',
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
                                                          metric='accuracy',
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
                                                           metric='accuracy',
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
                                                         metric='accuracy',
                                                         prior=UNIFORM_PRIOR,
                                                         random_seed=r)

    for method in success_rate_dict:
        success_rate_dict[method] /= RUNS
        output_name = "../output/active_learning/%s_%s_%s_%s_runs_%d.pkl" % (DATASET, 'accuracy', MODE, method, RUNS)
        pickle.dump(success_rate_dict[method], open(output_name, "wb"))

    # evaluation
    figname = "../output/active_learning/%s_%s_%s_runs_%d.pdf" % (DATASET, 'accurayc', MODE, RUNS)
    comparison_plot(success_rate_dict, figname)


def main_calibration_error(RUNS, MODE, DATASET):
    datafile = datafile_dict[DATASET]
    NUM_CLASSES = num_classes_dict[DATASET]

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile, False)
    N = len(observations)

    # prior is None
    # get samples for multiple runs
    # returns one thing: success or not
    success_rate_dict = {
        'random': copy.deepcopy(np.zeros((N,))),
        'ts': copy.deepcopy(np.zeros((N,))),
        'ttts': copy.deepcopy(np.zeros((N,))),
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
                                                   metric='calibration_error',
                                                   prior=None,
                                                   random_seed=r)
        print(r, 'ts')
        success_rate_dict['ts'] += get_samples(categories,
                                               observations,
                                               confidences,
                                               NUM_CLASSES,
                                               N,
                                               sample_method='ts',
                                               mode=MODE,
                                               metric='calibration_error',
                                               prior=None,
                                               random_seed=r)
        print(r, 'ttts')
        success_rate_dict['ttts'] += get_samples(categories,
                                                 observations,
                                                 confidences,
                                                 NUM_CLASSES,
                                                 N,
                                                 sample_method='ttts',
                                                 mode=MODE,
                                                 metric='calibration_error',
                                                 prior=None,
                                                 random_seed=r)
        print(r, 'epsilon_greedy')
        success_rate_dict['epsilon_greedy'] += get_samples(categories,
                                                           observations,
                                                           confidences,
                                                           NUM_CLASSES,
                                                           N,
                                                           sample_method='epsilon_greedy',
                                                           mode=MODE,
                                                           metric='calibration_error',
                                                           prior=None,
                                                           random_seed=r)
        print(r, 'bayesian_ucb')
        success_rate_dict['bayesian_ucb'] += get_samples(categories,
                                                         observations,
                                                         confidences,
                                                         NUM_CLASSES,
                                                         N,
                                                         sample_method='bayesian_ucb',
                                                         mode=MODE,
                                                         metric='calibration_error',
                                                         prior=None,
                                                         random_seed=r)

    for method in success_rate_dict:
        success_rate_dict[method] /= RUNS
        output_name = "../output/active_learning/%s_%s_%s_%s_runs_%d.pkl" % (DATASET, 'ece', MODE, method, RUNS)
        pickle.dump(success_rate_dict[method], open(output_name, "wb"))

    # evaluation
    figname = "../output/active_learning/%s_%s_%s_runs_%d.pdf" % (DATASET, 'ece', MODE, RUNS)
    comparison_plot(success_rate_dict, figname)


if __name__ == "__main__":

    RUNS = 100

    dataset = str(sys.argv[1])
    if dataset not in DATASET_LIST:
        raise ValueError("%s is not in DATASET_LIST." % dataset)

    for MODE in ['min', 'max']:
        print(dataset, MODE, '...')
        main_accuracy(RUNS, MODE, dataset)
        main_calibration_error(RUNS, MODE, dataset)
