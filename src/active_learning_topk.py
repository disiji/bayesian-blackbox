import argparse
import logging
import os
import pathlib
import pickle
import random
from collections import deque
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from active_utils import prepare_data, SAMPLE_CATEGORY, _get_confidence_k, _get_ground_truth
from data_utils import datafile_dict, num_classes_dict, DATASET_LIST
from models import BetaBernoulli, ClasswiseEce
from tqdm import tqdm

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875
DPI = 300
FONT_SIZE = 8
OUTPUT_DIR = "../output/active_learning_topk"
FIGURE_DIR = "../output/figures"
RUNS = 100


def get_samples_topk(args: argparse.Namespace,
                     categories: List[int],
                     observations: List[bool],
                     confidences: List[float],
                     num_classes: int,
                     num_samples: int,
                     sample_method: str,
                     experiment_name: str,
                     mode: str,
                     metric: str,
                     prior=None,
                     weight=None,
                     random_seed: int = 0) -> Tuple[List, List, List]:
    # prepare model, deques, thetas, choices

    random.seed(random_seed)

    if metric == 'accuracy':
        model = BetaBernoulli(num_classes, prior)
    elif metric == 'calibration_error':
        model = ClasswiseEce(num_classes, num_bins=10, weight=weight, prior=None)

    deques = [deque() for _ in range(num_classes)]
    for (category, score, observation) in zip(categories, confidences, observations):
        if metric == 'accuracy':
            deques[category].append(observation)
        elif metric == 'calibration_error':
            deques[category].append((observation, score))
    for _deque in deques:
        random.shuffle(_deque)

    sampled_categories = np.zeros((num_samples,), dtype=np.int)
    sampled_observations = np.zeros((num_samples,), dtype=np.int)
    sampled_scores = np.zeros((num_samples,), dtype=np.float)

    for idx in range(num_samples):
        # sampling process:
        # if there are less than k available arms to play, switch to top 1, the sampling method has been switched to top1,
        # then the return 'category_list' is an int
        categories_list = SAMPLE_CATEGORY[sample_method](deques=deques,
                                                         random_seed=random_seed,
                                                         model=model,
                                                         mode=mode,
                                                         topk=args.topk,
                                                         max_ttts_trial=50,
                                                         ttts_beta=0.5,
                                                         epsilon=0.1,
                                                         ucb_c=1, )
        if type(categories_list) != list:
            categories_list = [categories_list]
            if args.topk != 1:
                args.topk = 1

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

    # write sampled_categories, sampled_observations, sampled_scores to file
    dir = args.output / experiment_name
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)

    pickle.dump(sampled_categories, open(dir / 'sampled_categories.pkl', "wb"))
    pickle.dump(sampled_observations, open(dir / 'sampled_observations.pkl', "wb"))
    pickle.dump(sampled_scores, open(dir / 'sampled_scores.pkl', "wb"))

    return sampled_categories, sampled_observations, sampled_scores


def eval(args: argparse.Namespace,
         experiment_name: str,
         num_classes: int,
         metric: str,
         mode: str,
         prior=None,
         weight=None) -> Dict[str, np.ndarray]:
    """

    :param args:
    :param experiment_name:
    :param num_classes:
    :param metric:
    :param mode:
    :param prior:
    :param weight:
    :return avg_num_agreement: (num_samples, ) array.
            Average number of agreement between selected topk and ground truth topk at each step.
    :return cumulative_metric: (num_samples, ) array.
            Metric (accuracy or ece) measured on sampled_observations, sampled categories and sampled scores.
    :return non_cumulative_metric: (num_samples, ) array.
            Average metric (accuracy or ece) evaluated with model.eval of the selected topk arms at each step.
    """

    dir = args.output / experiment_name
    categories = pickle.load(open(dir / 'sampled_categories.pkl', "rb"))
    observations = pickle.load(open(dir / 'sampled_observations.pkl', "rb"))
    confidences = pickle.load(open(dir / 'sampled_scores.pkl', "rb"))

    ground_truth = _get_ground_truth(categories, observations, confidences, num_classes, metric, mode, topk=args.topk)

    evaluation_freq = args.evaluation_freq
    num_samples = len(categories)
    output_length = num_samples // evaluation_freq

    if metric == 'accuracy':
        model = BetaBernoulli(num_classes, prior)

    elif metric == 'calibration_error':
        model = ClasswiseEce(num_classes, num_bins=10, weight=weight, prior=None)

    avg_num_agreement = np.zeros((num_samples,))
    cumulative_metric = np.zeros((num_samples,))
    non_cumulative_metric = np.zeros((num_samples,))

    topk_arms = np.zeros((num_classes,), dtype=np.bool_)

    for idx, (category, observation, confidence) in enumerate(zip(categories, observations, confidences)):

        # Always update
        if metric == 'accuracy':
            model.update(category, observation)

        elif metric == 'calibration_error':
            model.update(category, observation, confidence)

        # To save time, evaluate less frequently
        if (idx % evaluation_freq) == 0:
            eval_idx = idx // evaluation_freq

            # select TOPK arms
            metric_val = model.eval

            topk_arms[:] = 0
            if mode == 'min':
                indices = metric_val.argsort()[:args.topk]
            elif mode == 'max':
                indices = metric_val.argsort()[-args.topk:]
            topk_arms[indices] = 1

            # evaluation
            avg_num_agreement[eval_idx] = (topk_arms == ground_truth).mean()
            # todo: each class is equally weighted by taking the mean. replace with frequency.(?)
            cumulative_metric[eval_idx] = model.frequentist_eval.mean()
            non_cumulative_metric[eval_idx] = metric_val[topk_arms].mean()

    # write eval results to file

    pickle.dump(avg_num_agreement,
                open(dir / ("avg_num_agreement_%s_%s_top%d.pkl" % (
                    metric, MODE, args.topk)), "wb"))

    pickle.dump(cumulative_metric,
                open(dir / ("cumulative_%s_%s_top%d.pkl" % (
                    metric, MODE, args.topk)), "wb"))

    pickle.dump(non_cumulative_metric,
                open(dir / ("non_cumulative_%s_%s_top%d.pkl" % (
                    metric, MODE, args.topk)), "wb"))

    return {'avg_num_agreement': avg_num_agreement,
            'cumulative_metric': cumulative_metric,
            'non_cumulative_metric': non_cumulative_metric}


def _comparison_plot(eval_result_dict: Dict[str, np.ndarray], figname: str, ylabel: str) -> None:
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


def comparison_plot_accuracy(args: argparse.Namespace, MODE: str, N: int) -> None:
    avg_num_agreement_dict = {
        'random': np.zeros((N,)),
        'ts_uniform': np.zeros((N,)),
        'ts_informed': np.zeros((N,)),
    }
    cumulative_metric_dict = {
        'random': np.zeros((N,)),
        'ts_uniform': np.zeros((N,)),
        'ts_informed': np.zeros((N,)),
    }
    non_cumulative_metric_dict = {
        'random': np.zeros((N,)),
        'ts_uniform': np.zeros((N,)),
        'ts_informed': np.zeros((N,)),
    }

    for method in ['random', 'ts_uniform', 'ts_informed']:
        for r in range(RUNS):
            experiment_name = '%s_acc_%s_run_idx_%d' % (args.dataset, method, r)
            dir = args.output / experiment_name

            avg_num_agreement_dict[method] += pickle.load(open(dir / ("avg_num_agreement_%s_%s_top%d.pkl" % (
                'accuracy', MODE, args.topk)), "rb"))
            cumulative_metric_dict[method] += pickle.load(open(dir / ("cumulative_%s_%s_top%d.pkl" % (
                'accuracy', MODE, args.topk)), "rb"))
            non_cumulative_metric_dict[method] += pickle.load(open(dir / ("non_cumulative_%s_%s_top%d.pkl" % (
                'accuracy', MODE, args.topk)), "rb"))

        avg_num_agreement_dict[method] /= RUNS
        cumulative_metric_dict[method] /= RUNS
        non_cumulative_metric_dict[method] /= RUNS

    _comparison_plot(avg_num_agreement_dict,
                     args.fig_dir / ("avg_num_agreement_%s_%s_%s_runs_%d_top%d.pdf" % (
                         args.dataset, 'acc', MODE, RUNS, args.topk)),
                     'Avg number of agreement')

    _comparison_plot(cumulative_metric_dict,
                     args.fig_dir / ("cumulative_%s_%s_%s_runs_%d_top%d.pdf" % (
                         args.dataset, 'acc', MODE, RUNS, args.topk)),
                     'Cumulative accuracy')

    _comparison_plot(non_cumulative_metric_dict,
                     args.fig_dir / ("non_cumulative_%s_%s_%s_runs_%d_top%d.pdf" % (
                         args.dataset, 'acc', MODE, RUNS, args.topk)),
                     'Non cumulative accuracy')


def comparison_plot_calibration_error(args: argparse.Namespace, MODE: str, N: int) -> None:
    avg_num_agreement_dict = {
        'random': np.zeros((N,)),
        'ts_uniform': np.zeros((N,)),
        'ts_informed': np.zeros((N,)),
    }
    cumulative_metric_dict = {
        'random': np.zeros((N,)),
        'ts_uniform': np.zeros((N,)),
        'ts_informed': np.zeros((N,)),
    }
    non_cumulative_metric_dict = {
        'random': np.zeros((N,)),
        'ts_uniform': np.zeros((N,)),
        'ts_informed': np.zeros((N,)),
    }

    for method in ['random', 'ts']:
        for r in range(RUNS):
            experiment_name = '%s_ece_%s_run_idx_%d' % (args.dataset, method, r)
            dir = args.output / experiment_name

            avg_num_agreement_dict[method] += pickle.load(open(dir / ("avg_num_agreement_%s_%s_top%d.pkl" % (
                'calibration_error', MODE, args.topk))), "rb")
            cumulative_metric_dict[method] += pickle.load(open(dir / ("cumulative_%s_%s_top%d.pkl" % (
                'calibration_error', MODE, args.topk))), "rb")
            non_cumulative_metric_dict[method] += pickle.load(open(dir / ("non_cumulative_%s_%s_top%d.pkl" % (
                'calibration_error', MODE, args.topk))), "rb")

        avg_num_agreement_dict[method] /= RUNS
        cumulative_metric_dict[method] /= RUNS
        non_cumulative_metric_dict[method] /= RUNS

    _comparison_plot(avg_num_agreement_dict,
                     args.fig_dir / ("avg_num_agreement_%s_%s_%s_runs_%d_top%d.pdf" % (
                         args.dataset, 'ece', MODE, args.topk)),
                     'Avg number of agreement')

    _comparison_plot(cumulative_metric_dict,
                     args.fig_dir / ("cumulative_%s_%s_%s_runs_%d_top%d.pdf" % (
                         args.dataset, 'ece', MODE, args.topk)),
                     'Cumulative ECE')

    _comparison_plot(non_cumulative_metric_dict,
                     args.fig_dir / ("non_cumulative_%s_%s_%s_runs_%d_top%d.pdf" % (
                         args.dataset, 'ece', MODE, args.topk)),
                     'Non cumulative ECE')


def main_accuracy_topk_two_stage(args: argparse.Namespace, MODE: str, SAMPLE=True, EVAL=True, PLOT=True) -> None:
    NUM_CLASSES = num_classes_dict[args.dataset]

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile_dict[args.dataset], False)
    N = len(observations)

    UNIFORM_PRIOR = np.ones((NUM_CLASSES, 2)) / 2

    confidence = _get_confidence_k(categories, confidences, NUM_CLASSES)
    INFORMED_PRIOR = np.array([confidence, 1 - confidence]).T

    # sampling...
    if SAMPLE:
        for r in tqdm(range(RUNS)):
            get_samples_topk(args,
                             categories,
                             observations,
                             confidences,
                             NUM_CLASSES,
                             N,
                             experiment_name=args.dataset + '_acc_random_run_idx_' + str(r),
                             sample_method='random',
                             mode=MODE,
                             metric='accuracy',
                             prior=UNIFORM_PRIOR,
                             random_seed=r)

            get_samples_topk(args,
                             categories,
                             observations,
                             confidences,
                             NUM_CLASSES,
                             N,
                             experiment_name=args.dataset + '_acc_ts_uniform_run_idx_' + str(r),
                             sample_method='ts',
                             mode=MODE,
                             metric='accuracy',
                             prior=UNIFORM_PRIOR,
                             random_seed=r)

            get_samples_topk(args,
                             categories,
                             observations,
                             confidences,
                             NUM_CLASSES,
                             N,
                             experiment_name=args.dataset + '_acc_ts_informed_run_idx_' + str(r),
                             sample_method='ts',
                             mode=MODE,
                             metric='accuracy',
                             prior=INFORMED_PRIOR,
                             random_seed=r)

    # evaluating
    # get samples for multiple runs
    # returns one thing: success or not
    if EVAL:
        for r in tqdm(range(RUNS)):
            eval(args,
                 experiment_name=args.dataset + '_acc_random_run_idx_' + str(r),
                 num_classes=NUM_CLASSES,
                 metric='accuracy',
                 mode=MODE,
                 prior=UNIFORM_PRIOR)

            eval(args,
                 experiment_name=args.dataset + '_acc_ts_uniform_run_idx_' + str(r),
                 num_classes=NUM_CLASSES,
                 metric='accuracy',
                 mode=MODE,
                 prior=UNIFORM_PRIOR)

            eval(args,
                 experiment_name=args.dataset + '_acc_ts_informed_run_idx_' + str(r),
                 num_classes=NUM_CLASSES,
                 metric='accuracy',
                 mode=MODE,
                 prior=UNIFORM_PRIOR)

    if PLOT:
        comparison_plot_accuracy(args, MODE, N)


def main_calibration_error_topk(args: argparse.Namespace, MODE: str, SAMPLE=True, EVAL=True, PLOT=True) -> None:
    NUM_CLASSES = num_classes_dict[args.dataset]

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile_dict[args.dataset], False)
    N = len(observations)

    if SAMPLE:
        for r in tqdm(range(RUNS)):
            get_samples_topk(args,
                             categories,
                             observations,
                             confidences,
                             NUM_CLASSES,
                             N,
                             experiment_name=args.dataset + '_ece_random_run_idx_' + str(r),
                             sample_method='random',
                             mode=MODE,
                             metric='calibration_error',
                             prior=None,
                             random_seed=r)

            get_samples_topk(args,
                             categories,
                             observations,
                             confidences,
                             NUM_CLASSES,
                             N,
                             experiment_name=args.dataset + '_ece_ts_run_idx_' + str(r),
                             sample_method='ts',
                             mode=MODE,
                             metric='calibration_error',
                             prior=None,
                             random_seed=r)

    if EVAL:
        for r in tqdm(range(RUNS)):
            eval(
                args,
                experiment_name=args.dataset + '_ece_random_run_idx_' + str(r),
                num_classes=NUM_CLASSES,
                metric='calibration_error',
                mode=MODE,
                prior=None)

            eval(
                args,
                experiment_name=args.dataset + '_ece_ts_run_idx_' + str(r),
                num_classes=NUM_CLASSES,
                metric='calibration_error',
                mode=MODE,
                prior=None)

    if PLOT:
        comparison_plot_calibration_error(args, MODE, N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default='cifar100', help='input dataset')
    parser.add_argument('--output', type=pathlib.Path, default=OUTPUT_DIR, help='output prefix')
    parser.add_argument('--fig_dir', type=pathlib.Path, default=FIGURE_DIR, help='figures output prefix')
    parser.add_argument('--topk', type=int, default=10, help='number of optimal arms to identify')
    parser.add_argument('--evaluation_freq', type=int,  default=1, help='Evaluation frequency. Set to a higher number to speed up evaluation on ImageNet and CIFAR.')
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    if args.dataset not in DATASET_LIST:
        raise ValueError("%s is not in DATASET_LIST." % args.dataset)

    for MODE in ['min', 'max']:
        print(args.dataset, MODE, '...')
        main_accuracy_topk_two_stage(args, MODE, SAMPLE=True, EVAL=True, PLOT=True)
        main_calibration_error_topk(args, MODE, SAMPLE=True, EVAL=True, PLOT=True)
