import copy
import pickle
import random
from collections import deque
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from active_utils import prepare_data, SAMPLE_CATEGORY, _get_confidence_k, _get_ground_truth, _get_accuracy_k, \
    _get_ece_k
from data_utils import datafile_dict, num_classes_dict, DATASET_LIST
from models import BetaBernoulli, ClasswiseEce

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875
DPI = 300
FONT_SIZE = 8


def get_samples_topk(categories: List[int],
                     observations: List[bool],
                     confidences: List[float],
                     num_classes: int,
                     num_samples: int,
                     sample_method: str,
                     mode: str,
                     metric: str,
                     topk: int = 1,
                     prior=None,
                     weight=None,
                     random_seed: int = 0) -> Dict[str, np.ndarray]:
    """

    :param categories:
    :param observations:
    :param confidences:
    :param num_classes:
    :param num_samples:
    :param sample_method:
    :param mode:
    :param metric:
    :param topk:
    :param prior:
    :param weight:
    :param random_seed:

    :return avg_num_agreement: (num_samples, ) array. Average number of agreement between selected topk and ground truth topk at each step.
    :return cumulative_metric: (num_samples, ) array. Metric (accuracy or ece) measured on sampled_observations, sampled categories and sampled scores.
    :return non_cumulative_metric: (num_samples, ) array. Average metric (accuracy or ece) evaluated with model.eval of the selected topk arms at each step.
    """
    # prepare model, deques, thetas, choices

    random.seed(random_seed)
    TOPK = topk

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

    ground_truth = _get_ground_truth(categories, observations, confidences, num_classes, metric, mode, topk=TOPK)

    avg_num_agreement = [None] * num_samples
    cumulative_metric = [None] * num_samples
    non_cumulative_metric = [None] * num_samples

    sampled_categories = [None] * num_samples
    sampled_scores = [None] * num_samples
    sampled_observations = [None] * num_samples

    idx = 0

    while (idx < num_samples):

        if idx % (num_samples // 10) == 0:
            print (idx, '...')
        # sampling process:
        # if there are less than k available arms to play, switch to top 1, the sampling method has been switched to top1,
        # then the return 'category_list' is an int
        categories_list = SAMPLE_CATEGORY[sample_method].__call__(deques=deques,
                                                                  random_seed=random_seed,
                                                                  model=model,
                                                                  mode=mode,
                                                                  topk=topk,
                                                                  max_ttts_trial=50,
                                                                  ttts_beta=0.5,
                                                                  epsilon=0.1,
                                                                  ucb_c=1, )
        if type(categories_list) != list:
            categories_list = [categories_list]
            topk = 1

        # update model, deques, thetas, choices
        for category in categories_list:
            if metric == 'accuracy':
                observation = deques[category].pop()
                model.update(category, observation)

            elif metric == 'calibration_error':
                observation, score = deques[category].pop()
                model.update(category, observation, score)
                sampled_scores[idx] = score

            sampled_categories[idx] = category
            sampled_observations[idx] = observation

            # select TOPK arms
            metric_val = model.eval
            if mode == 'min':
                topk_arms = metric_val.argsort()[:TOPK].flatten().tolist()
            elif mode == 'max':
                topk_arms = metric_val.argsort()[-TOPK:][::-1].flatten().tolist()

            # evaluation
            avg_num_agreement[idx] = len([_ for _ in topk_arms if _ in ground_truth]) * 1.0 / TOPK
            if metric == 'accuracy':
                cumulative_metric[idx] = _get_accuracy_k(sampled_categories[:idx + 1], sampled_observations[:idx + 1],
                                                         num_classes).mean()
            elif metric == 'calibration_error':
                cumulative_metric[idx] = _get_ece_k(sampled_categories[:idx + 1], sampled_observations[:idx + 1],
                                                    sampled_scores[:idx + 1], num_classes, num_bins=10).mean()
            non_cumulative_metric[idx] = metric_val[topk_arms].mean()

            idx += 1

    return {'avg_num_agreement': np.array(avg_num_agreement),
            'cumulative_metric': np.array(cumulative_metric),
            'non_cumulative_metric': np.array(non_cumulative_metric)}


def comparison_plot(success_rate_dict, figname, ylabel) -> None:
    # If labels are getting cut off make the figsize smaller
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO), dpi=300)
    for method_name, success_rate in success_rate_dict.items():
        plt.plot(success_rate, label=method_name)
    plt.xlabel('#Queries')
    plt.ylabel(ylabel)
    plt.legend()
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.ylim(0.0, 1.0)
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')


def main_accuracy_topk_two_stage(RUNS: int, MODE: str, DATASET: str, topk=1) -> None:
    datafile = datafile_dict[DATASET]
    NUM_CLASSES = num_classes_dict[DATASET]

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile, False)
    N = len(observations) // 10

    UNIFORM_PRIOR = np.ones((NUM_CLASSES, 2)) / 2

    confidence = _get_confidence_k(categories, confidences, NUM_CLASSES)
    INFORMED_PRIOR = np.array([confidence, 1 - confidence]).T

    # get samples for multiple runs
    # returns one thing: success or not
    avg_num_agreement_dict = {
        'random': copy.deepcopy(np.zeros((N,))),
        'ts_uniform': copy.deepcopy(np.zeros((N,))),
        'ts_informed': copy.deepcopy(np.zeros((N,))),
    }
    cumulative_metric_dict = {
        'random': copy.deepcopy(np.zeros((N,))),
        'ts_uniform': copy.deepcopy(np.zeros((N,))),
        'ts_informed': copy.deepcopy(np.zeros((N,))),
    }
    non_cumulative_metric_dict = {
        'random': copy.deepcopy(np.zeros((N,))),
        'ts_uniform': copy.deepcopy(np.zeros((N,))),
        'ts_informed': copy.deepcopy(np.zeros((N,))),
    }
    for r in range(RUNS):
        print(r, 'random')
        output = get_samples_topk(categories,
                                  observations,
                                  confidences,
                                  NUM_CLASSES,
                                  N,
                                  sample_method='random',
                                  mode=MODE,
                                  metric='accuracy',
                                  topk=topk,
                                  prior=UNIFORM_PRIOR,
                                  random_seed=r)
        avg_num_agreement_dict['random'] += output['avg_num_agreement']
        cumulative_metric_dict['random'] += output['cumulative_metric']
        non_cumulative_metric_dict['random'] += output['non_cumulative_metric']

        print(r, 'ts_uniform')
        output = get_samples_topk(categories,
                                  observations,
                                  confidences,
                                  NUM_CLASSES,
                                  N,
                                  sample_method='ts',
                                  mode=MODE,
                                  metric='accuracy',
                                  topk=topk,
                                  prior=UNIFORM_PRIOR,
                                  random_seed=r)
        avg_num_agreement_dict['ts_uniform'] += output['avg_num_agreement']
        cumulative_metric_dict['ts_uniform'] += output['cumulative_metric']
        non_cumulative_metric_dict['ts_uniform'] += output['non_cumulative_metric']

        print(r, 'ts_informed')
        output = get_samples_topk(categories,
                                  observations,
                                  confidences,
                                  NUM_CLASSES,
                                  N,
                                  sample_method='ts',
                                  mode=MODE,
                                  metric='accuracy',
                                  topk=topk,
                                  prior=INFORMED_PRIOR,
                                  random_seed=r)
        avg_num_agreement_dict['ts_informed'] += output['avg_num_agreement']
        cumulative_metric_dict['ts_informed'] += output['cumulative_metric']
        non_cumulative_metric_dict['ts_informed'] += output['non_cumulative_metric']

    for method in ['random', 'ts_uniform', 'ts_informed']:
        pickle.dump(avg_num_agreement_dict[method],
                    open("../output/active_learning_topk/avg_num_agreement_%s_%s_%s_%s_runs_%d_topk_%d.pkl" % (
                        DATASET, 'acc', MODE, method, RUNS, topk), "wb"))

        pickle.dump(cumulative_metric_dict[method],
                    open("../output/active_learning_topk/cumulative_%s_%s_%s_%s_runs_%d_topk_%d.pkl" % (
                        DATASET, 'acc', MODE, method, RUNS, topk), "wb"))

        pickle.dump(non_cumulative_metric_dict[method],
                    open("../output/active_learning_topk/non_cumulative_%s_%s_%s_%s_runs_%d_topk_%d.pkl" % (
                        DATASET, 'acc', MODE, method, RUNS, topk), "wb"))

    # evaluation
    comparison_plot(avg_num_agreement_dict,
                    "../output/active_learning_topk/avg_num_agreement_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                        DATASET, 'acc', MODE, RUNS, topk),
                    'Avg number of agreement')

    comparison_plot(cumulative_metric_dict,
                    "../output/active_learning_topk/cumulative_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                        DATASET, 'acc', MODE, RUNS, topk),
                    'Cummlative accuracy')

    comparison_plot(non_cumulative_metric_dict,
                    "../output/active_learning_topk/non_cumulative_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                        DATASET, 'acc', MODE, RUNS, topk),
                    'Non cumulative accuracy')


def main_calibration_error_topk(RUNS: int, MODE: str, DATASET: str, topk=1) -> None:
    datafile = datafile_dict[DATASET]
    NUM_CLASSES = num_classes_dict[DATASET]

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile, False)
    N = len(observations) // 10

    avg_num_agreement_dict = {
        'random': copy.deepcopy(np.zeros((N,))),
        'ts': copy.deepcopy(np.zeros((N,))),
    }
    cumulative_metric_dict = {
        'random': copy.deepcopy(np.zeros((N,))),
        'ts': copy.deepcopy(np.zeros((N,))),
    }
    non_cumulative_metric_dict = {
        'random': copy.deepcopy(np.zeros((N,))),
        'ts': copy.deepcopy(np.zeros((N,))),
    }

    for r in range(RUNS):
        print(r, 'random')
        output = get_samples_topk(categories,
                                  observations,
                                  confidences,
                                  NUM_CLASSES,
                                  N,
                                  sample_method='random',
                                  mode=MODE,
                                  metric='calibration_error',
                                  topk=topk,
                                  prior=None,
                                  random_seed=r)
        avg_num_agreement_dict['random'] += output['avg_num_agreement']
        cumulative_metric_dict['random'] += output['cumulative_metric']
        non_cumulative_metric_dict['random'] += output['non_cumulative_metric']

        print(r, 'ts')
        output = get_samples_topk(categories,
                                  observations,
                                  confidences,
                                  NUM_CLASSES,
                                  N,
                                  sample_method='ts',
                                  mode=MODE,
                                  metric='calibration_error',
                                  topk=topk,
                                  prior=None,
                                  random_seed=r)
        avg_num_agreement_dict['ts'] += output['avg_num_agreement']
        cumulative_metric_dict['ts'] += output['cumulative_metric']
        non_cumulative_metric_dict['ts'] += output['non_cumulative_metric']

    for method in ['random', 'ts']:
        pickle.dump(avg_num_agreement_dict[method],
                    open("../output/active_learning_topk/avg_num_agreement_%s_%s_%s_%s_runs_%d_topk_%d.pkl" % (
                        DATASET, 'ece', MODE, method, RUNS, topk), "wb"))

        pickle.dump(cumulative_metric_dict[method],
                    open("../output/active_learning_topk/cumulative_%s_%s_%s_%s_runs_%d_topk_%d.pkl" % (
                        DATASET, 'ece', MODE, method, RUNS, topk), "wb"))

        pickle.dump(non_cumulative_metric_dict[method],
                    open("../output/active_learning_topk/non_cumulative_%s_%s_%s_%s_runs_%d_topk_%d.pkl" % (
                        DATASET, 'ece', MODE, method, RUNS, topk), "wb"))

    # evaluation
    comparison_plot(avg_num_agreement_dict,
                    "../output/active_learning_topk/avg_num_agreement_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                        DATASET, 'ece', MODE, RUNS, topk),
                    'Avg number of agreement')

    comparison_plot(cumulative_metric_dict,
                    "../output/active_learning_topk/cumulative_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                        DATASET, 'ece', MODE, RUNS, topk),
                    'Cummlative ECE')

    comparison_plot(non_cumulative_metric_dict,
                    "../output/active_learning_topk/non_cumulative_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                        DATASET, 'ece', MODE, RUNS, topk),
                    'Non cumulative ECE')


if __name__ == "__main__":

    RUNS = 100
    TOP_K = 2

    # dataset = str(sys.argv[1])
    dataset = 'cifar100'
    if dataset not in DATASET_LIST:
        raise ValueError("%s is not in DATASET_LIST." % dataset)

    for MODE in ['min', 'max']:
        print(dataset, MODE, '...')
        main_accuracy_topk_two_stage(RUNS, MODE, dataset, topk=TOP_K)
        # main_calibration_error_topk(RUNS, MODE, dataset, topk=TOP_K)
