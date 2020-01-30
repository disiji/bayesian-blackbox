import argparse
import ctypes
import logging
import pathlib
import random
import warnings
from collections import deque
from functools import reduce
from multiprocessing import Array, Lock, Process, JoinableQueue
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from active_utils import SAMPLE_CATEGORY, _get_confidence_k, get_ground_truth, eval_ece
from calibration import CALIBRATION_MODELS
from data_utils import datafile_dict, num_classes_dict, logits_dict, datasize_dict, prepare_data, train_holdout_split, \
    DATASET_LIST
from models import BetaBernoulli, ClasswiseEce

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875
DPI = 300
FONT_SIZE = 8
OUTPUT_DIR = "../output/active_learning_topk"
RUNS = 100
LOG_FREQ = 100
CALIBRATION_FREQ = 100
PRIOR_STRENGTH = 3
CALIBRATION_MODEL = 'platt_scaling'
HOLDOUT_RATIO = 0.1

logger = logging.getLogger(__name__)
process_lock = Lock()


def get_samples_topk(args: argparse.Namespace,
                     categories: List[int],
                     observations: List[bool],
                     confidences: List[float],
                     labels: List[int],
                     indices: List[int],
                     num_classes: int,
                     num_samples: int,
                     sample_method: str,
                     prior=None,
                     weight=None,
                     random_seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # prepare model, deques, thetas, choices

    random.seed(random_seed)

    if args.metric == 'accuracy':
        model = BetaBernoulli(num_classes, prior)
    elif args.metric == 'calibration_error':
        model = ClasswiseEce(num_classes, num_bins=10, pseudocount=args.pseudocount, weight=weight, prior=None)

    deques = [deque() for _ in range(num_classes)]
    for category, score, observation, label, index in zip(categories, confidences, observations, labels, indices):
        if args.metric == 'accuracy':
            deques[category].append(observation)
        elif args.metric == 'calibration_error':
            deques[category].append((observation, score, label, index))
    for _deque in deques:
        random.shuffle(_deque)

    sampled_categories = np.zeros((num_samples,), dtype=np.int)
    sampled_observations = np.zeros((num_samples,), dtype=np.int)
    sampled_scores = np.zeros((num_samples,), dtype=np.float)
    sampled_labels = np.zeros((num_samples,), dtype=np.int)
    sampled_indices = np.zeros((num_samples,), dtype=np.int)
    sample_fct = SAMPLE_CATEGORY[sample_method]

    topk = args.topk
    idx = 0

    while idx < num_samples:
        # sampling process:
        # if there are less than k available arms to play, switch to top 1, the sampling method has been switched to top1,
        # then the return 'category_list' is an int
        categories_list = sample_fct(deques=deques,
                                     random_seed=random_seed,
                                     model=model,
                                     mode=args.mode,
                                     topk=topk,
                                     max_ttts_trial=50,
                                     ttts_beta=0.5,
                                     epsilon=0.1,
                                     ucb_c=1, )

        if type(categories_list) != list:
            categories_list = [categories_list]
            if topk != 1:
                topk = 1

        # update model, deques, thetas, choices
        for category in categories_list:
            if args.metric == 'accuracy':
                observation = deques[category].pop()
                model.update(category, observation)

            elif args.metric == 'calibration_error':
                observation, score, label, index = deques[category].pop()
                model.update(category, observation, score)
                sampled_scores[idx] = score
                sampled_labels[idx] = label
                sampled_indices[idx] = index

            sampled_categories[idx] = category
            sampled_observations[idx] = observation

            idx += 1

    return sampled_categories, sampled_observations, sampled_scores, sampled_labels, sampled_indices


def eval(args: argparse.Namespace,
         categories: List[int],
         observations: List[bool],
         confidences: List[float],
         labels: List[int],
         indices: List[int],
         ground_truth: np.ndarray,
         num_classes: int,
         holdout_categories: List[int] = None,  # will be used if train classwise calibration model
         holdout_observations: List[bool] = None,
         holdout_confidences: List[float] = None,
         holdout_labels: List[int] = None,
         holdout_indices: List[int] = None,
         prior=None,
         weight=None) -> Tuple[np.ndarray, ...]:
    """

    :param args:
    :param categories:
    :param observations:
    :param confidences:
    :param labels:
    :param indices:
    :param ground_truth:
    :param num_classes:
    :param holdout_categories:
    :param holdout_observations:
    :param holdout_confidences:
    :param holdout_labels:
    :param holdout_indices:
    :param prior:
    :param weight:
    :return avg_num_agreement: (num_samples, ) array.
            Average number of agreement between selected topk and ground truth topk at each step.
    :return cumulative_metric: (num_samples, ) array.
            Metric (accuracy or ece) measured on sampled_observations, sampled categories and sampled scores.
    :return non_cumulative_metric: (num_samples, ) array.
            Average metric (accuracy or ece) evaluated with model.eval of the selected topk arms at each step.
    """
    num_samples = len(categories)

    if args.metric == 'accuracy':
        model = BetaBernoulli(num_classes, prior)
    elif args.metric == 'calibration_error':
        model = ClasswiseEce(num_classes, num_bins=10, pseudocount=args.pseudocount, weight=weight, prior=None)

    avg_num_agreement = np.zeros((num_samples // LOG_FREQ + 1,))
    cumulative_metric = np.zeros((num_samples // LOG_FREQ + 1,))
    non_cumulative_metric = np.zeros((num_samples // LOG_FREQ + 1,))

    if args.metric == 'calibration_error':

        holdout_calibrated_ece = np.zeros((num_samples // CALIBRATION_FREQ + 1,))

        if args.calibration_model in ['histogram_binning', 'isotonic_regression', 'bayesian_binning_quantiles',
                                      'classwise_histogram_binning', 'two_group_histogram_binning']:
            holdout_X = np.array(holdout_confidences)
            holdout_X = np.array([1 - holdout_X, holdout_X]).T

        elif args.calibration_model in ['platt_scaling', 'temperature_scaling']:
            holdout_indices_array = np.array(holdout_indices, dtype=np.int)
            with process_lock:
                holdout_X = logits[holdout_indices_array]

    topk_arms = np.zeros((num_classes,), dtype=np.bool_)

    for idx, (category, observation, confidence, label, index) in enumerate(
            zip(categories, observations, confidences, labels, indices)):

        if args.metric == 'accuracy':
            model.update(category, observation)
        elif args.metric == 'calibration_error':
            model.update(category, observation, confidence)

        if idx % LOG_FREQ == 0:
            # select TOPK arms
            topk_arms[:] = 0
            metric_val = model.eval
            if args.mode == 'min':
                topk_indices = metric_val.argsort()[:args.topk]
            elif args.mode == 'max':
                topk_indices = metric_val.argsort()[-args.topk:]
            topk_arms[topk_indices] = 1
            # evaluation
            avg_num_agreement[idx // LOG_FREQ] = topk_arms[ground_truth == 1].mean()
            # todo: each class is equally weighted by taking the mean. replace with frequency.(?)
            cumulative_metric[idx // LOG_FREQ] = model.frequentist_eval.mean()
            non_cumulative_metric[idx // LOG_FREQ] = metric_val[topk_arms].mean()

        if args.metric == 'calibration_error' and idx % CALIBRATION_FREQ == 0:
            # before calibration
            if idx == 0:
                holdout_calibrated_ece[idx] = eval_ece(holdout_confidences, holdout_observations, num_bins=10)

            else:
                if args.calibration_model in ['histogram_binning', 'isotonic_regression', 'bayesian_binning_quantiles']:
                    calibration_model = CALIBRATION_MODELS[args.calibration_model]()
                    X = np.array(confidences[:idx])
                    X = np.array([1 - X, X]).T
                    y = np.array(observations[:idx]) * 1
                    calibration_model.fit(X, y)
                    calibrated_holdout_confidences = calibration_model.predict_proba(holdout_X)[:, 1].tolist()

                elif args.calibration_model in ['platt_scaling', 'temperature_scaling']:
                    calibration_model = CALIBRATION_MODELS[args.calibration_model]()
                    X = logits[indices[:idx]]
                    y = np.array(labels[:idx], dtype=np.int)
                    calibration_model.fit(X, y)

                    pred_array = np.array(holdout_categories).astype(int).reshape(-1, 1)
                    calibrated_holdout_confidences = calibration_model.predict_proba(holdout_X)
                    calibrated_holdout_confidences = np.take_along_axis(calibrated_holdout_confidences, pred_array,
                                                                        axis=1).squeeze().tolist()

                elif args.calibration_model in ['classwise_histogram_binning']:
                    # use the current MPE reliability diagram for calibration, no need to train a separate calibration model
                    calibration_mapping = model.beta_params_mpe
                    bin_idx = np.floor(np.array(holdout_confidences) * 10).astype(int)
                    bin_idx[bin_idx == 10] = 9
                    calibrated_holdout_confidences = calibration_mapping[holdout_categories, bin_idx].tolist()

                elif args.calibration_model in ['two_group_histogram_binning']:

                    calibrated_holdout_confidences = np.zeros(len(holdout_confidences))

                    calibration_model_less_calibrated = CALIBRATION_MODELS['histogram_binning']()
                    calibration_model_more_calibrated = CALIBRATION_MODELS['histogram_binning']()
                    X = np.array(confidences[:idx])
                    X = np.array([1 - X, X]).T
                    y = np.array(observations[:idx]) * 1

                    train_mask = np.array([ground_truth[val] for val in categories[:idx]])
                    holdout_mask = np.array([ground_truth[val] for val in holdout_categories])

                    calibration_model_less_calibrated.fit(X[train_mask], y[train_mask])
                    calibration_model_more_calibrated.fit(X[np.invert(train_mask)],
                                                          y[np.invert(train_mask)])

                    calibrated_holdout_confidences[holdout_mask] = calibration_model_less_calibrated.predict_proba(
                        holdout_X[holdout_mask])[:, 1]
                    calibrated_holdout_confidences[
                        np.invert(holdout_mask)] = calibration_model_more_calibrated.predict_proba(
                        holdout_X[np.invert(holdout_mask)])[:, 1]

                    calibrated_holdout_confidences = calibrated_holdout_confidences.tolist()
                else:
                    raise ValueError("%s is not an implemented calibration method." % args.calibration_model)

                holdout_calibrated_ece[idx // CALIBRATION_FREQ] = eval_ece(calibrated_holdout_confidences,
                                                                           holdout_observations, num_bins=10)

    if args.metric == 'accuracy':
        return avg_num_agreement, cumulative_metric, non_cumulative_metric
    elif args.metric == 'calibration_error':
        with process_lock:
            logger.debug(holdout_calibrated_ece)
        return avg_num_agreement, cumulative_metric, non_cumulative_metric, holdout_calibrated_ece


def _comparison_plot(eval_result_dict: Dict[str, np.ndarray], eval_freq: int, figname: str, ylabel: str) -> None:
    # If labels are getting cut off make the figsize smaller
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO), dpi=300)

    if args.metric == 'calibration_error':
        total_samples = datasize_dict[args.dataset] * (1 - HOLDOUT_RATIO)
    elif args.metric == 'accuracy':
        total_samples = datasize_dict[args.dataset]

    if args.metric == 'accuracy':
        method_list = ['non-active_no_prior', 'non-active_uniform', 'non-active_informed', 'ts_uniform', 'ts_informed']
    elif args.metric == 'calibration_error':
        method_list = ['non-active', 'ts']

    for method_name in method_list:
        metric_eval = eval_result_dict[method_name]
        metric_eval = metric_eval[: int(len(metric_eval) / 2)]
        x = np.arange(len(metric_eval)) * eval_freq / total_samples
        plt.plot(x, metric_eval, label=method_name)
    plt.xlabel('#Percentage')
    plt.ylabel(ylabel)
    plt.legend(fontsize=FONT_SIZE - 2)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    # plt.ylim(0.0, 1.0)
    # plt.xlim(0.0, 0.5)
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')


def comparison_plot(args: argparse.Namespace,
                    experiment_name: str,
                    avg_num_agreement_dict: Dict[str, np.ndarray],
                    cumulative_metric_dict: Dict[str, np.ndarray],
                    non_cumulative_metric_dict: Dict[str, np.ndarray],
                    holdout_ece_dict: Dict[str, np.ndarray] = None) -> None:
    """

    :param args:
    :param experiment_name:
    :param avg_num_agreement_dict:
        Dict maps str to np.ndarray of shape (RUNS, num_samples // LOG_FREQ + 1)
    :param cumulative_metric_dict:
        Dict maps str to np.ndarray of shape (RUNS, num_samples // LOG_FREQ + 1)
    :param non_cumulative_metric_dict:
        Dict maps str to np.ndarray of shape (RUNS, num_samples // LOG_FREQ + 1)
    :return:
    """

    # avg over runs
    if args.metric == 'accuracy':
        method_list = ['non-active_no_prior', 'non-active_uniform', 'non-active_informed', 'ts_uniform', 'ts_informed']
    elif args.metric == 'calibration_error':
        method_list = ['non-active', 'ts']

    for method in method_list:
        avg_num_agreement_dict[method] = avg_num_agreement_dict[method].mean(axis=0)
        cumulative_metric_dict[method] = cumulative_metric_dict[method].mean(axis=0)
        non_cumulative_metric_dict[method] = non_cumulative_metric_dict[method].mean(axis=0)
        if holdout_ece_dict:
            holdout_ece_dict[method] = holdout_ece_dict[method].mean(axis=0)

    _comparison_plot(avg_num_agreement_dict, LOG_FREQ,
                     args.output / experiment_name / "avg_num_agreement.pdf",
                     'Avg number of agreement')
    _comparison_plot(cumulative_metric_dict, LOG_FREQ,
                     args.output / experiment_name / "cumulative.pdf",
                     ('Cumulative %s' % args.metric))
    _comparison_plot(non_cumulative_metric_dict, LOG_FREQ,
                     args.output / experiment_name / "non_cumulative.pdf",
                     ('Non cumulative %s' % args.metric))
    if holdout_ece_dict:
        _comparison_plot(holdout_ece_dict, CALIBRATION_FREQ,
                         args.output / experiment_name / ("holdout_ece_%s.pdf" % args.calibration_model), 'ECE')


def main_accuracy_topk(args: argparse.Namespace, SAMPLE=True, EVAL=True, PLOT=True) -> None:
    num_classes = num_classes_dict[args.dataset]

    categories, observations, confidences, idx2category, category2idx, labels = prepare_data(
        datafile_dict[args.dataset], False)
    indices = np.arange(len(categories))

    num_samples = len(observations)

    UNIFORM_PRIOR = np.ones((num_classes, 2)) / 2 * args.pseudocount
    confidence = _get_confidence_k(categories, confidences, num_classes)
    INFORMED_PRIOR = np.array([confidence, 1 - confidence]).T * args.pseudocount

    experiment_name = '%s_%s_%s_top%d_runs%d_pseudocount%.2f' % (
        args.dataset, args.metric, args.mode, args.topk, RUNS, args.pseudocount)

    if not (args.output / experiment_name).is_dir():
        (args.output / experiment_name).mkdir()

    sampled_categories_dict = {
        'non-active': np.empty((RUNS, num_samples), dtype=int),
        'ts_uniform': np.empty((RUNS, num_samples), dtype=int),
        'ts_informed': np.empty((RUNS, num_samples), dtype=int),
    }
    sampled_observations_dict = {
        'non-active': np.empty((RUNS, num_samples), dtype=bool),
        'ts_uniform': np.empty((RUNS, num_samples), dtype=bool),
        'ts_informed': np.empty((RUNS, num_samples), dtype=bool),
    }
    sampled_scores_dict = {
        'non-active': np.empty((RUNS, num_samples), dtype=float),
        'ts_uniform': np.empty((RUNS, num_samples), dtype=float),
        'ts_informed': np.empty((RUNS, num_samples), dtype=float),
    }
    sampled_labels_dict = {
        'non-active': np.empty((RUNS, num_samples), dtype=int),
        'ts_uniform': np.empty((RUNS, num_samples), dtype=int),
        'ts_informed': np.empty((RUNS, num_samples), dtype=int),
    }
    sampled_indices_dict = {
        'non-active': np.empty((RUNS, num_samples), dtype=int),
        'ts_uniform': np.empty((RUNS, num_samples), dtype=int),
        'ts_informed': np.empty((RUNS, num_samples), dtype=int),
    }

    avg_num_agreement_dict = {
        'non-active_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'non-active_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'non-active_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }
    cumulative_metric_dict = {
        'non-active_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'non-active_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'non-active_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'non-active': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }
    non_cumulative_metric_dict = {
        'non-active_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'non-active_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'non-active_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'non-active': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }

    if SAMPLE:
        for r in tqdm(range(RUNS)):
            sampled_categories_dict['non-active'][r], sampled_observations_dict['non-active'][r], \
            sampled_scores_dict['non-active'][r], sampled_labels_dict['non-active'][r], \
            sampled_indices_dict['non-active'][r] = get_samples_topk(args,
                                                                     categories,
                                                                     observations,
                                                                     confidences,
                                                                     labels,
                                                                     indices,
                                                                     num_classes,
                                                                     num_samples,
                                                                     sample_method='random',
                                                                     prior=UNIFORM_PRIOR * 1e-6,
                                                                     random_seed=r)

            sampled_categories_dict['ts_uniform'][r], sampled_observations_dict['ts_uniform'][r], \
            sampled_scores_dict['ts_uniform'][r], sampled_labels_dict['ts_uniform'][r], \
            sampled_indices_dict['ts_uniform'][r] = get_samples_topk(args,
                                                                     categories,
                                                                     observations,
                                                                     confidences,
                                                                     labels,
                                                                     indices,
                                                                     num_classes,
                                                                     num_samples,
                                                                     sample_method='ts',
                                                                     prior=UNIFORM_PRIOR,
                                                                     random_seed=r)

            sampled_categories_dict['ts_informed'][r], sampled_observations_dict['ts_informed'][r], \
            sampled_scores_dict['ts_informed'][r], sampled_labels_dict['ts_informed'][r], \
            sampled_indices_dict['ts_informed'][r] = get_samples_topk(args,
                                                                      categories,
                                                                      observations,
                                                                      confidences,
                                                                      labels,
                                                                      indices,
                                                                      num_classes,
                                                                      num_samples,
                                                                      sample_method='ts',
                                                                      prior=INFORMED_PRIOR,
                                                                      random_seed=r)
        # write samples to file
        for method in ['non-active', 'ts_uniform', 'ts_informed']:
            np.save(args.output / experiment_name / ('sampled_categories_%s.npy' % method),
                    sampled_categories_dict[method])
            np.save(args.output / experiment_name / ('sampled_observations_%s.npy' % method),
                    sampled_observations_dict[method])
            np.save(args.output / experiment_name / ('sampled_scores_%s.npy' % method), sampled_scores_dict[method])
            np.save(args.output / experiment_name / ('sampled_labels_%s.npy' % method), sampled_labels_dict[method])
            np.save(args.output / experiment_name / ('sampled_indices_%s.npy' % method), sampled_indices_dict[method])
    else:
        # load sampled categories, scores and observations from file
        for method in ['non-active', 'ts_uniform', 'ts_informed']:
            sampled_categories_dict[method] = np.load(
                args.output / experiment_name / ('sampled_categories_%s.npy' % method))
            sampled_observations_dict[method] = np.load(
                args.output / experiment_name / ('sampled_observations_%s.npy' % method))
            sampled_scores_dict[method] = np.load(args.output / experiment_name / ('sampled_scores_%s.npy' % method))
            sampled_labels_dict[method] = np.load(args.output / experiment_name / ('sampled_labels_%s.npy' % method))
            sampled_indices_dict[method] = np.load(args.output / experiment_name / ('sampled_indices_%s.npy' % method))

    if EVAL:
        ground_truth = get_ground_truth(categories, observations, confidences, num_classes, args.metric, args.mode,
                                        topk=args.topk)

        for r in tqdm(range(RUNS)):
            avg_num_agreement_dict['non-active_no_prior'][r], cumulative_metric_dict['non-active_no_prior'][r], \
            non_cumulative_metric_dict['non-active_no_prior'][r] = eval(args,
                                                                        sampled_categories_dict['non-active'][
                                                                            r].tolist(),
                                                                        sampled_observations_dict['non-active'][
                                                                            r].tolist(),
                                                                        sampled_scores_dict['non-active'][r].tolist(),
                                                                        sampled_labels_dict['non-active'][r].tolist(),
                                                                        sampled_indices_dict['non-active'][r].tolist(),
                                                                        ground_truth,
                                                                        num_classes=num_classes,
                                                                        prior=UNIFORM_PRIOR * 1e-6)
            avg_num_agreement_dict['non-active_uniform'][r], cumulative_metric_dict['non-active_uniform'][r], \
            non_cumulative_metric_dict['non-active_uniform'][r] = eval(args,
                                                                       sampled_categories_dict['non-active'][
                                                                           r].tolist(),
                                                                       sampled_observations_dict['non-active'][
                                                                           r].tolist(),
                                                                       sampled_scores_dict['non-active'][r].tolist(),
                                                                       sampled_labels_dict['non-active'][r].tolist(),
                                                                       sampled_indices_dict['non-active'][r].tolist(),
                                                                       ground_truth,
                                                                       num_classes=num_classes,
                                                                       prior=UNIFORM_PRIOR)
            avg_num_agreement_dict['non-active_informed'][r], cumulative_metric_dict['non-active_informed'][r], \
            non_cumulative_metric_dict['non-active_informed'][r] = eval(args,
                                                                        sampled_categories_dict['non-active'][
                                                                            r].tolist(),
                                                                        sampled_observations_dict['non-active'][
                                                                            r].tolist(),
                                                                        sampled_scores_dict['non-active'][r].tolist(),
                                                                        sampled_labels_dict['non-active'][r].tolist(),
                                                                        sampled_indices_dict['non-active'][r].tolist(),
                                                                        ground_truth,
                                                                        num_classes=num_classes,
                                                                        prior=INFORMED_PRIOR)

            avg_num_agreement_dict['ts_uniform'][r], cumulative_metric_dict['ts_uniform'][r], \
            non_cumulative_metric_dict['ts_uniform'][r] = eval(args,
                                                               sampled_categories_dict['ts_uniform'][r].tolist(),
                                                               sampled_observations_dict['ts_uniform'][r].tolist(),
                                                               sampled_scores_dict['ts_uniform'][r].tolist(),
                                                               sampled_labels_dict['ts_uniform'][r].tolist(),
                                                               sampled_indices_dict['ts_uniform'][r].tolist(),
                                                               ground_truth,
                                                               num_classes=num_classes,
                                                               prior=UNIFORM_PRIOR)

            avg_num_agreement_dict['ts_informed'][r], cumulative_metric_dict['ts_informed'][r], \
            non_cumulative_metric_dict['ts_informed'][r] = eval(args,
                                                                sampled_categories_dict['ts_informed'][r].tolist(),
                                                                sampled_observations_dict['ts_informed'][r].tolist(),
                                                                sampled_scores_dict['ts_informed'][r].tolist(),
                                                                sampled_labels_dict['ts_informed'][r].tolist(),
                                                                sampled_indices_dict['ts_informed'][r].tolist(),
                                                                ground_truth,
                                                                num_classes=num_classes,
                                                                prior=INFORMED_PRIOR)

        for method in ['non-active_no_prior', 'non-active_uniform', 'non-active_informed', 'ts_uniform', 'ts_informed']:
            np.save(args.output / experiment_name / ('avg_num_agreement_%s.npy' % method),
                    avg_num_agreement_dict[method])
            np.save(args.output / experiment_name / ('cumulative_metric_%s.npy' % method),
                    cumulative_metric_dict[method])
            np.save(args.output / experiment_name / ('non_cumulative_metric_%s.npy' % method),
                    non_cumulative_metric_dict[method])
    else:
        for method in ['non-active_no_prior', 'non-active_uniform', 'non-active_informed', 'ts_uniform', 'ts_informed']:
            avg_num_agreement_dict[method] = np.load(
                args.output / experiment_name / ('avg_num_agreement_%s.npy' % method))
            cumulative_metric_dict[method] = np.load(
                args.output / experiment_name / ('cumulative_metric_%s.npy' % method))
            non_cumulative_metric_dict[method] = np.load(
                args.output / experiment_name / ('non_cumulative_metric_%s.npy' % method))

    if PLOT:
        comparison_plot(args, experiment_name, avg_num_agreement_dict, cumulative_metric_dict,
                        non_cumulative_metric_dict)


class MpSafeSharedArray:
    """Multiprocessing-safe shared array."""
    DTYPE_TO_CTYPE = {
        np.int: ctypes.c_long,
        np.bool: ctypes.c_bool,
        np.float: ctypes.c_double
    }

    def __init__(self, shape, dtype=np.float):
        warnings.warn('MpSafeSharedArray is experimental. Use at your own risk.')

        if dtype not in MpSafeSharedArray.DTYPE_TO_CTYPE:
            raise ValueError('Unsupported dtype')
        self._shape = shape
        self._dtype = dtype

        ctype = MpSafeSharedArray.DTYPE_TO_CTYPE[dtype]
        num_elements = reduce(lambda x, y: x * y, shape)
        self._mp_arr = Array(ctype, num_elements)

    def get_lock(self):
        return self._mp_arr.get_lock()

    def get_array(self):
        buffer = self._mp_arr.get_obj()
        arr = np.ndarray(shape=self._shape, dtype=self._dtype, buffer=buffer)
        return arr


def main_calibration_error_topk(args: argparse.Namespace, SAMPLE=True, EVAL=True, PLOT=True) -> None:
    num_classes = num_classes_dict[args.dataset]

    global logits
    logits_path = logits_dict.get(args.dataset,
                                  None)  # Since we haven't created all the logits yet, assign defaul value of None.
    if logits_path is not None:
        logits = np.genfromtxt(logits_path)[:, 1:]
    else:
        logits = None
    logits_lock = Lock()

    categories, observations, confidences, idx2category, category2idx, labels = prepare_data(
        datafile_dict[args.dataset], False)
    indices = np.arange(len(categories))

    categories, observations, confidences, labels, indices, \
    holdout_categories, holdout_observations, holdout_confidences, holdout_labels, holdout_indices = \
        train_holdout_split(categories, observations, confidences, labels, indices, holdout_ratio=HOLDOUT_RATIO)

    num_samples = len(observations)

    experiment_name = '%s_%s_%s_top%d_runs%d_pseudocount%.2f' % (
        args.dataset, args.metric, args.mode, args.topk, RUNS, args.pseudocount)

    if not (args.output / experiment_name).is_dir():
        (args.output / experiment_name).mkdir()

    sampled_categories_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
        'ts': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
    }
    sampled_observations_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples), dtype=np.bool),
        'ts': MpSafeSharedArray((RUNS, num_samples), dtype=np.bool),
    }
    sampled_scores_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples), dtype=np.float),
        'ts': MpSafeSharedArray((RUNS, num_samples), dtype=np.float),
    }
    sampled_labels_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
        'ts': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
    }
    sampled_indices_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
        'ts': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
    }

    avg_num_agreement_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
        'ts': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
    }
    cumulative_metric_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
        'ts': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
    }
    non_cumulative_metric_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
        'ts': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
    }
    holdout_ece_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples // CALIBRATION_FREQ + 1), dtype=np.float),
        'ts': MpSafeSharedArray((RUNS, num_samples // CALIBRATION_FREQ + 1), dtype=np.float),
    }

    if SAMPLE:
        logger.info('Starting sampling')

        def sampler_worker(queue):
            # Continue to work until queue is empty
            while not queue.empty():
                # Get a job (e.g. run index)
                run_idx, sample_method = queue.get()

                # @disiji - update if you change the name...
                if sample_method == 'non-active':
                    method = 'random'
                else:
                    method = sample_method

                with process_lock:
                    logger.debug(f'Working on sampling task :: Run: {run_idx} :: Method {method}')

                sampled_categories, sampled_observations, sampled_scores, sampled_labels, sampled_indices = \
                    get_samples_topk(args,
                                     categories,
                                     observations,
                                     confidences,
                                     labels,
                                     indices,
                                     num_classes,
                                     num_samples,
                                     sample_method=method,
                                     prior=None,
                                     random_seed=run_idx)

                category_array = sampled_categories_dict[sample_method]
                with category_array.get_lock():
                    arr = category_array.get_array()
                    arr[run_idx] = sampled_categories

                observation_array = sampled_observations_dict[sample_method]
                with observation_array.get_lock():
                    arr = observation_array.get_array()
                    arr[run_idx] = sampled_observations

                score_array = sampled_scores_dict[sample_method]
                with score_array.get_lock():
                    arr = score_array.get_array()
                    arr[run_idx] = sampled_scores

                label_array = sampled_labels_dict[sample_method]
                with label_array.get_lock():
                    arr = label_array.get_array()
                    arr[run_idx] = sampled_labels

                index_array = sampled_indices_dict[sample_method]
                with index_array.get_lock():
                    arr = index_array.get_array()
                    arr[run_idx] = sampled_indices

                queue.task_done()

        # Enqueue tasks
        logger.debug('Enqueueing sampling tasks')
        sampling_job_queue = JoinableQueue()
        for i in range(RUNS):
            sampling_job_queue.put((i, 'non-active'))
            sampling_job_queue.put((i, 'ts'))

        # Start tasks
        logger.debug('Running sampling tasks')
        for _ in range(args.processes):
            process = Process(target=sampler_worker, args=(sampling_job_queue,))
            process.start()

        # Make sure all work is done before proceeding
        sampling_job_queue.join()
        logger.debug('Sampling finished')

        # We want numpy arrays but instead have the weird custom process-safe arrays.
        # Since the rest of the code expects a numpy array we're going to replace the
        # elements in the dictionaries with their array counterparts.
        for method in ['non-active', 'ts']:
            with sampled_categories_dict[method].get_lock():
                sampled_categories_dict[method] = sampled_categories_dict[method].get_array()
            with sampled_observations_dict[method].get_lock():
                sampled_observations_dict[method] = sampled_observations_dict[method].get_array()
            with sampled_scores_dict[method].get_lock():
                sampled_scores_dict[method] = sampled_scores_dict[method].get_array()
            with sampled_labels_dict[method].get_lock():
                sampled_labels_dict[method] = sampled_labels_dict[method].get_array()
            with sampled_indices_dict[method].get_lock():
                sampled_indices_dict[method] = sampled_indices_dict[method].get_array()

        # Write to disk
        for method in ['non-active', 'ts']:
            np.save(args.output / experiment_name / ('sampled_categories_%s.npy' % method),
                    sampled_categories_dict[method])
            np.save(args.output / experiment_name / ('sampled_observations_%s.npy' % method),
                    sampled_observations_dict[method])
            np.save(args.output / experiment_name / ('sampled_scores_%s.npy' % method), sampled_scores_dict[method])
            np.save(args.output / experiment_name / ('sampled_labels_%s.npy' % method), sampled_labels_dict[method])
            np.save(args.output / experiment_name / ('sampled_indices_%s.npy' % method), sampled_indices_dict[method])
    else:
        # load sampled categories, scores and observations from file
        for method in ['non-active', 'ts']:
            sampled_categories_dict[method] = np.load(
                args.output / experiment_name / ('sampled_categories_%s.npy' % method))
            sampled_observations_dict[method] = np.load(
                args.output / experiment_name / ('sampled_observations_%s.npy' % method))
            sampled_scores_dict[method] = np.load(args.output / experiment_name / ('sampled_scores_%s.npy' % method))
            sampled_labels_dict[method] = np.load(args.output / experiment_name / ('sampled_labels_%s.npy' % method))
            sampled_indices_dict[method] = np.load(args.output / experiment_name / ('sampled_indices_%s.npy' % method))

    if EVAL:
        logger.info('Starting evaluation')
        ground_truth = get_ground_truth(categories, observations, confidences, num_classes, args.metric, args.mode,
                                        topk=args.topk)

        def eval_worker(queue):
            while not queue.empty():
                run_idx, method = queue.get()
                with process_lock:
                    logger.debug(f'Working on eval task :: Run: {run_idx} :: Method {method}')
                agreement, metric, noncum_metric, ece = eval(args,
                                                             sampled_categories_dict[method][run_idx].tolist(),
                                                             sampled_observations_dict[method][run_idx].tolist(),
                                                             sampled_scores_dict[method][run_idx].tolist(),
                                                             sampled_labels_dict[method][run_idx].tolist(),
                                                             sampled_indices_dict[method][run_idx].tolist(),
                                                             ground_truth,
                                                             num_classes=num_classes,
                                                             holdout_categories=holdout_categories,
                                                             holdout_observations=holdout_observations,
                                                             holdout_confidences=holdout_confidences,
                                                             holdout_labels=holdout_labels,
                                                             holdout_indices=holdout_indices,
                                                             prior=None)

                # Write outputs
                avg_num_agreement_array = avg_num_agreement_dict[method]
                with avg_num_agreement_array.get_lock():
                    arr = avg_num_agreement_array.get_array()
                    arr[run_idx] = agreement
                cumulative_metric_array = cumulative_metric_dict[method]
                with cumulative_metric_array.get_lock():
                    arr = cumulative_metric_array.get_array()
                    arr[run_idx] = metric
                non_cumulative_metric_array = non_cumulative_metric_dict[method]
                with non_cumulative_metric_array.get_lock():
                    arr = non_cumulative_metric_array.get_array()
                    arr[run_idx] = noncum_metric
                holdout_ece_array = holdout_ece_dict[method]
                with holdout_ece_array.get_lock():
                    arr = holdout_ece_array.get_array()
                    arr[run_idx] = ece

                queue.task_done()

        # Enqueue tasks
        logger.debug('Enqueueing evaluation tasks')
        eval_job_queue = JoinableQueue()
        for i in range(RUNS):
            eval_job_queue.put((i, 'non-active'))
            eval_job_queue.put((i, 'ts'))

        # Start tasks
        logger.debug('Running evaluation tasks')
        for _ in range(args.processes):
            process = Process(target=eval_worker, args=(eval_job_queue,))
            process.start()

        # Make sure all work is done before proceeding
        eval_job_queue.join()
        logger.debug('Evaluation tasks finished')

        # We want numpy arrays but instead have the weird custom process-safe arrays.
        # Since the rest of the code expects a numpy array we're going to replace the
        # elements in the dictionaries with their array counterparts.
        for method in ['non-active', 'ts']:
            with avg_num_agreement_dict[method].get_lock():
                avg_num_agreement_dict[method] = avg_num_agreement_dict[method].get_array()
            with cumulative_metric_dict[method].get_lock():
                cumulative_metric_dict[method] = cumulative_metric_dict[method].get_array()
            with non_cumulative_metric_dict[method].get_lock():
                non_cumulative_metric_dict[method] = non_cumulative_metric_dict[method].get_array()
            with holdout_ece_dict[method].get_lock():
                holdout_ece_dict[method] = holdout_ece_dict[method].get_array()

        for method in ['non-active', 'ts']:
            np.save(args.output / experiment_name / ('avg_num_agreement_%s.npy' % method),
                    avg_num_agreement_dict[method])
            np.save(args.output / experiment_name / ('cumulative_metric_%s.npy' % method),
                    cumulative_metric_dict[method])
            np.save(args.output / experiment_name / ('non_cumulative_metric_%s.npy' % method),
                    non_cumulative_metric_dict[method])
            np.save(args.output / experiment_name / ('holdout_ece_%s_%s.npy' % (args.calibration_model, method)),
                    holdout_ece_dict[method])
    else:
        for method in ['non-active', 'ts']:
            avg_num_agreement_dict[method] = np.load(
                args.output / experiment_name / ('avg_num_agreement_%s.npy' % method))
            cumulative_metric_dict[method] = np.load(
                args.output / experiment_name / ('cumulative_metric_%s.npy' % method))
            non_cumulative_metric_dict[method] = np.load(
                args.output / experiment_name / ('non_cumulative_metric_%s.npy' % method))
            holdout_ece_dict[method] = np.load(
                args.output / experiment_name / ('holdout_ece_%s_%s.npy' % (args.calibration_model, method)))

    if PLOT:
        comparison_plot(args, experiment_name, avg_num_agreement_dict, cumulative_metric_dict,
                        non_cumulative_metric_dict, holdout_ece_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default='cifar100', help='input dataset')
    parser.add_argument('--output', type=pathlib.Path, default=OUTPUT_DIR, help='output prefix')
    parser.add_argument('-topk', type=int, default=10, help='number of optimal arms to identify')
    parser.add_argument('-metric', type=str, help='accuracy or calibration_error')
    parser.add_argument('-pseudocount', type=float, default=PRIOR_STRENGTH, help='strength of prior')
    parser.add_argument('-mode', type=str, help='min or max, identify topk with highest/lowest reward')
    parser.add_argument('--calibration_model', type=str, default=CALIBRATION_MODEL,
                        help='calibration models to apply on holdout data')
    parser.add_argument('--processes', type=int, default=4,
                        help='Number of sample processes. Increase to speed up sampling.')
    parser.add_argument('--debug', action='store_true', help='Enables debug statements')

    args, _ = parser.parse_known_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    if args.dataset not in DATASET_LIST:
        raise ValueError("%s is not in DATASET_LIST." % args.dataset)

    logger.info(f'Dataset: {args.dataset} :: Model: {args.mode}')
    if args.metric == 'accuracy':
        main_accuracy_topk(args, SAMPLE=True, EVAL=True, PLOT=True)
    elif args.metric == 'calibration_error':
        main_calibration_error_topk(args, SAMPLE=False, EVAL=True, PLOT=True)
