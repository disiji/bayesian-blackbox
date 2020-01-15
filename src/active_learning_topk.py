import copy
import os
import pickle
import random
from collections import deque
from typing import List, Dict, Tuple

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
OUTPUT_DIR = "../output/active_learning_topk/"
FIGURE_DIR = "../output/figures/"


def get_samples_topk(categories: List[int],
                     observations: List[bool],
                     confidences: List[float],
                     num_classes: int,
                     num_samples: int,
                     sample_method: str,
                     experiment_name: str,
                     mode: str,
                     metric: str,
                     topk: int = 1,
                     prior=None,
                     weight=None,
                     random_seed: int = 0) -> Tuple[List, List, List]:
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
    """
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

    sampled_categories = [None] * num_samples
    sampled_scores = [None] * num_samples
    sampled_observations = [None] * num_samples

    idx = 0

    while (idx < num_samples):

        if idx % (num_samples // 10) == 0:
            print(idx, '...')
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
            if topk != 1:
                print("Switch to top 1 sampling at step %d..." % idx)
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

            idx += 1

    # write sampled_categories, sampled_observations, sampled_scores to file
    dir = OUTPUT_DIR + experiment_name
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)

    pickle.dump(sampled_categories, open(dir + '/sampled_categories.pkl', "wb"))
    pickle.dump(sampled_observations, open(dir + '/sampled_observations.pkl', "wb"))
    pickle.dump(sampled_scores, open(dir + '/sampled_scores.pkl', "wb"))

    return sampled_categories, sampled_observations, sampled_scores


def eval(experiment_name: str,
         num_classes: int,
         metric: str,
         mode: str,
         topk: int = 1,
         prior=None,
         weight=None) -> Dict[str, np.ndarray]:
    """
    :param num_classes:
    :param metric:
    :param mode:
    :param topk:
    :param prior:
    :param weight:
    :return avg_num_agreement: (num_samples, ) array.
            Average number of agreement between selected topk and ground truth topk at each step.
    :return cumulative_metric: (num_samples, ) array.
            Metric (accuracy or ece) measured on sampled_observations, sampled categories and sampled scores.
    :return non_cumulative_metric: (num_samples, ) array.
            Average metric (accuracy or ece) evaluated with model.eval of the selected topk arms at each step.
    """
    dir = OUTPUT_DIR + experiment_name
    categories = pickle.load(open(dir + '/sampled_categories.pkl', "rb"))
    observations = pickle.load(open(dir + '/sampled_observations.pkl', "rb"))
    confidences = pickle.load(open(dir + '/sampled_scores.pkl', "rb"))

    ground_truth = _get_ground_truth(categories, observations, confidences, num_classes, metric, mode, topk=topk)
    num_samples = len(categories)

    if metric == 'accuracy':
        model = copy.deepcopy(BetaBernoulli(num_classes, prior))

    elif metric == 'calibration_error':
        model = copy.deepcopy(ClasswiseEce(num_classes, num_bins=10, weight=weight, prior=None))

    avg_num_agreement = [None] * num_samples
    cumulative_metric = [None] * num_samples
    non_cumulative_metric = [None] * num_samples

    for idx, (category, observation, confidence) in enumerate(zip(categories, observations, confidences)):

        if metric == 'accuracy':
            model.update(category, observation)

        elif metric == 'calibration_error':
            model.update(category, observation, confidence)

        # select TOPK arms
        metric_val = model.eval
        if mode == 'min':
            topk_arms = metric_val.argsort()[:topk].flatten().tolist()
        elif mode == 'max':
            topk_arms = metric_val.argsort()[-topk:][::-1].flatten().tolist()

        # evaluation
        avg_num_agreement[idx] = len([_ for _ in topk_arms if _ in ground_truth]) * 1.0 / topk
        if metric == 'accuracy':
            cumulative_metric[idx] = _get_accuracy_k(categories[:idx + 1], observations[:idx + 1],
                                                     num_classes).mean()
        elif metric == 'calibration_error':
            cumulative_metric[idx] = _get_ece_k(categories[:idx + 1], observations[:idx + 1],
                                                confidences[:idx + 1], num_classes, num_bins=10).mean()
        non_cumulative_metric[idx] = metric_val[topk_arms].mean()

    # write eval results to file
    dir = OUTPUT_DIR + experiment_name
    avg_num_agreement = np.array(avg_num_agreement)
    cumulative_metric = np.array(cumulative_metric)
    non_cumulative_metric = np.array(non_cumulative_metric)

    pickle.dump(avg_num_agreement,
                open(dir + "/avg_num_agreement_%s_%s_top%d.pkl" % (
                    metric, MODE, topk), "wb"))

    pickle.dump(cumulative_metric,
                open(dir + "/cumulative_%s_%s_top%d.pkl" % (
                    metric, MODE, topk), "wb"))

    pickle.dump(non_cumulative_metric,
                open(dir + "/non_cumulative_%s_%s_top%d.pkl" % (
                    metric, MODE, topk), "wb"))

    return {'avg_num_agreement': avg_num_agreement,
            'cumulative_metric': cumulative_metric,
            'non_cumulative_metric': non_cumulative_metric}


def _comparison_plot(eval_result_dict, figname, ylabel) -> None:
    # If labels are getting cut off make the figsize smaller
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO), dpi=300)
    for method_name, success_rate in eval_result_dict.items():
        plt.plot(success_rate, label=method_name)
    plt.xlabel('#Queries')
    plt.ylabel(ylabel)
    plt.legend()
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.ylim(0.0, 1.0)
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')


def comparison_plot_accuracy(DATASET, MODE, topk, RUNS, N) -> None:
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

    for method in ['random', 'ts_uniform', 'ts_informed']:
        for r in range(RUNS):
            experiment_name = '_acc_%s_%s_run_idx_%d' % (DATASET, method, r)
            dir = OUTPUT_DIR + experiment_name

            print(pickle.load(open(dir + "/avg_num_agreement_%s_%s_top%d.pkl" % (
                'accuracy', MODE, topk), "rb")))

            avg_num_agreement_dict[method] += pickle.load(open(dir + "/avg_num_agreement_%s_%s_top%d.pkl" % (
                'accuracy', MODE, topk), "rb"))
            cumulative_metric_dict[method] += pickle.load(open(dir + "/cumulative_%s_%s_top%d.pkl" % (
                'accuracy', MODE, topk), "rb"))
            non_cumulative_metric_dict[method] += pickle.load(open(dir + "/non_cumulative_%s_%s_top%d.pkl" % (
                'accuracy', MODE, topk), "rb"))

    _comparison_plot(avg_num_agreement_dict,
                     FIGURE_DIR + "avg_num_agreement_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                         DATASET, 'acc', MODE, RUNS, topk),
                     'Avg number of agreement')

    _comparison_plot(cumulative_metric_dict,
                     FIGURE_DIR + "cumulative_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                         DATASET, 'acc', MODE, RUNS, topk),
                     'Cummlative accuracy')

    _comparison_plot(non_cumulative_metric_dict,
                     FIGURE_DIR + "non_cumulative_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                         DATASET, 'acc', MODE, RUNS, topk),
                     'Non cumulative accuracy')


def comparison_plot_calibration_error(DATASET, MODE, topk, RUNS, N) -> None:
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

    for method in ['random', 'ts']:
        for r in range(RUNS):
            experiment_name = '_ece_%s_%s_run_idx_%d' % (DATASET, method, r)
            dir = OUTPUT_DIR + experiment_name

            print(pickle.load(open(dir + "/avg_num_agreement_%s_%s_top%d.pkl" % (
                'accuracy', MODE, topk), "rb")))

            avg_num_agreement_dict[method] += pickle.load(open(dir + "/avg_num_agreement_%s_%s_top%d.pkl" % (
                'accuracy', MODE, topk), "rb"))
            cumulative_metric_dict[method] += pickle.load(open(dir + "/cumulative_%s_%s_top%d.pkl" % (
                'accuracy', MODE, topk), "rb"))
            non_cumulative_metric_dict[method] += pickle.load(open(dir + "/non_cumulative_%s_%s_top%d.pkl" % (
                'accuracy', MODE, topk), "rb"))

    _comparison_plot(avg_num_agreement_dict,
                     FIGURE_DIR + "avg_num_agreement_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                         DATASET, 'ece', MODE, RUNS, topk),
                     'Avg number of agreement')

    _comparison_plot(cumulative_metric_dict,
                     FIGURE_DIR + "cumulative_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                         DATASET, 'ece', MODE, RUNS, topk),
                     'Cummlative ECE')

    _comparison_plot(non_cumulative_metric_dict,
                     FIGURE_DIR + "non_cumulative_%s_%s_%s_runs_%d_topk_%d.pdf" % (
                         DATASET, 'ece', MODE, RUNS, topk),
                     'Non cumulative ECE')


def main_accuracy_topk_two_stage(RUNS: int, MODE: str, DATASET: str, topk=1, SAMPLE=True, EVAL=True, PLOT=True) -> None:
    datafile = datafile_dict[DATASET]
    NUM_CLASSES = num_classes_dict[DATASET]

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile, False)
    N = len(observations)

    UNIFORM_PRIOR = np.ones((NUM_CLASSES, 2)) / 2

    confidence = _get_confidence_k(categories, confidences, NUM_CLASSES)
    INFORMED_PRIOR = np.array([confidence, 1 - confidence]).T

    # sampling...
    if SAMPLE:
        for r in range(RUNS):
            print(r, 'random')
            get_samples_topk(categories,
                             observations,
                             confidences,
                             NUM_CLASSES,
                             N,
                             experiment_name=DATASET + '_acc_random_run_idx_' + str(r),
                             sample_method='random',
                             mode=MODE,
                             metric='accuracy',
                             topk=topk,
                             prior=UNIFORM_PRIOR,
                             random_seed=r)

            print(r, 'ts_uniform')
            get_samples_topk(categories,
                             observations,
                             confidences,
                             NUM_CLASSES,
                             N,
                             experiment_name=DATASET + '_acc_ts_uniform_run_idx_' + str(r),
                             sample_method='ts',
                             mode=MODE,
                             metric='accuracy',
                             topk=topk,
                             prior=UNIFORM_PRIOR,
                             random_seed=r)

            print(r, 'ts_informed')
            get_samples_topk(categories,
                             observations,
                             confidences,
                             NUM_CLASSES,
                             N,
                             experiment_name=DATASET + '_acc_ts_informed_run_idx_' + str(r),
                             sample_method='ts',
                             mode=MODE,
                             metric='accuracy',
                             topk=topk,
                             prior=INFORMED_PRIOR,
                             random_seed=r)

    # evaluating
    # get samples for multiple runs
    # returns one thing: success or not
    if EVAL:
        for r in range(RUNS):
            print(r, 'random')
            eval(experiment_name=DATASET + '_acc_random_run_idx_' + str(r),
                 num_classes=NUM_CLASSES,
                 metric='accuracy',
                 mode=MODE,
                 topk=topk,
                 prior=UNIFORM_PRIOR)

            print(r, 'ts_uniform')
            eval(experiment_name=DATASET + '_acc_ts_uniform_run_idx_' + str(r),
                 num_classes=NUM_CLASSES,
                 metric='accuracy',
                 mode=MODE,
                 topk=topk,
                 prior=UNIFORM_PRIOR)

            print(r, 'ts_informed')
            eval(experiment_name=DATASET + '_acc_ts_informed_run_idx_' + str(r),
                 num_classes=NUM_CLASSES,
                 metric='accuracy',
                 mode=MODE,
                 topk=topk,
                 prior=UNIFORM_PRIOR)

    if PLOT:
        comparison_plot_accuracy(DATASET, MODE, topk, RUNS, N)


def main_calibration_error_topk(RUNS: int, MODE: str, DATASET: str, topk=1, SAMPLE=True, EVAL=True, PLOT=True) -> None:
    datafile = datafile_dict[DATASET]
    NUM_CLASSES = num_classes_dict[DATASET]

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile, False)
    N = len(observations)

    if SAMPLE:
        for r in range(RUNS):
            print(r, 'random')
            get_samples_topk(categories,
                             observations,
                             confidences,
                             NUM_CLASSES,
                             N,
                             experiment_name=DATASET + '_ece_random_run_idx_' + str(r),
                             sample_method='random',
                             mode=MODE,
                             metric='calibration_error',
                             topk=topk,
                             prior=None,
                             random_seed=r)

            print(r, 'ts')
            get_samples_topk(categories,
                             observations,
                             confidences,
                             NUM_CLASSES,
                             N,
                             experiment_name=DATASET + '_ece_ts_idx_' + str(r),
                             sample_method='ts',
                             mode=MODE,
                             metric='calibration_error',
                             topk=topk,
                             prior=None,
                             random_seed=r)

    if EVAL:
        for r in range(RUNS):
            print(r, 'random')
            eval(experiment_name=DATASET + '_ece_random_run_idx_' + str(r),
                 num_classes=NUM_CLASSES,
                 metric='calibration_error',
                 mode=MODE,
                 topk=topk,
                 prior=None)

            print(r, 'ts')
            eval(experiment_name=DATASET + '_acc_ts_uniform_run_idx_' + str(r),
                 num_classes=NUM_CLASSES,
                 metric='calibration_error',
                 mode=MODE,
                 topk=topk,
                 prior=None)

    if PLOT:
        comparison_plot_calibration_error(DATASET, MODE, topk, RUNS, N)


if __name__ == "__main__":

    RUNS = 10
    TOP_K = 5

    # dataset = str(sys.argv[1])
    dataset = 'cifar100'
    if dataset not in DATASET_LIST:
        raise ValueError("%s is not in DATASET_LIST." % dataset)

    for MODE in ['min', 'max']:
        print(dataset, MODE, '...')
        # main_accuracy_topk_two_stage(RUNS, MODE, dataset, topk=TOP_K, SAMPLE=True, EVAL=True, PLOT=True
        main_calibration_error_topk(RUNS, MODE, dataset, topk=TOP_K, SAMPLE=True, EVAL=True, PLOT=True)
