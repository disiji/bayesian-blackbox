import argparse
import ctypes
import logging
import pathlib
import random
import warnings
from collections import deque
from functools import reduce
from multiprocessing import Array, Process, JoinableQueue
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from active_utils import SAMPLE_CATEGORY, _get_confidence_k, get_ground_truth, eval_ece
from calibration import CALIBRATION_MODELS
from data_utils import datafile_dict, num_classes_dict, DATASET_LIST, prepare_data, train_holdout_split
from models import BetaBernoulli, ClasswiseEce
from tqdm import tqdm

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875
DPI = 300
FONT_SIZE = 8
OUTPUT_DIR = "../output_from_anvil/active_learning_topk"
RUNS = 100
LOG_FREQ = 100
CALIBRATION_FREQ = 100
PRIOR_STRENGTH = 5
CALIBRATION_MODEL = 'histogram_binning'


def get_samples_topk(args: argparse.Namespace,
                     categories: List[int],
                     observations: List[bool],
                     confidences: List[float],
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
        model = ClasswiseEce(num_classes, num_bins=10, weight=weight, prior=None)

    deques = [deque() for _ in range(num_classes)]
    for (category, score, observation) in zip(categories, confidences, observations):
        if args.metric == 'accuracy':
            deques[category].append(observation)
        elif args.metric == 'calibration_error':
            deques[category].append((observation, score))
    for _deque in deques:
        random.shuffle(_deque)

    sampled_categories = np.zeros((num_samples,), dtype=np.int)
    sampled_observations = np.zeros((num_samples,), dtype=np.int)
    sampled_scores = np.zeros((num_samples,), dtype=np.float)
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
                observation, score = deques[category].pop()
                model.update(category, observation, score)
                sampled_scores[idx] = score

            sampled_categories[idx] = category
            sampled_observations[idx] = observation

            idx += 1

    return sampled_categories, sampled_observations, sampled_scores


def eval(args: argparse.Namespace,
         categories: List[int],
         observations: List[bool],
         confidences: List[float],
         ground_truth: np.ndarray,
         num_classes: int,
         holdout_categories: List[int] = None,  # will be used if train classwise calibration model
         holdout_observations: List[bool] = None,
         holdout_confidences: List[float] = None,
         prior=None,
         weight=None) -> Tuple[np.ndarray, ...]:
    """

    :param args:
    :param categories:
    :param observations:
    :param confidences:
    :param ground_truth:
    :param num_classes:
    :param holdout_categories:
    :param holdout_observations:
    :param holdout_confidences:
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
        model = ClasswiseEce(num_classes, num_bins=10, weight=weight, prior=None)

    avg_num_agreement = np.zeros((num_samples // LOG_FREQ + 1,))
    cumulative_metric = np.zeros((num_samples // LOG_FREQ + 1,))
    non_cumulative_metric = np.zeros((num_samples // LOG_FREQ + 1,))

    if args.metric == 'calibration_error':
        holdout_calibrated_ece = np.zeros((num_samples // CALIBRATION_FREQ + 1,))

    topk_arms = np.zeros((num_classes,), dtype=np.bool_)
    holdout_X = np.array(holdout_confidences)
    holdout_X = np.array([1 - holdout_X, holdout_X]).T

    for idx, (category, observation, confidence) in enumerate(zip(categories, observations, confidences)):

        if args.metric == 'accuracy':
            model.update(category, observation)

        elif args.metric == 'calibration_error':
            model.update(category, observation, confidence)

        if idx % LOG_FREQ == 0:
            # select TOPK arms
            topk_arms[:] = 0
            metric_val = model.eval
            if args.mode == 'min':
                indices = metric_val.argsort()[:args.topk]
            elif args.mode == 'max':
                indices = metric_val.argsort()[-args.topk:]
            topk_arms[indices] = 1
            # evaluation
            avg_num_agreement[idx // LOG_FREQ] = topk_arms[ground_truth == 1].mean()
            # todo: each class is equally weighted by taking the mean. replace with frequency.(?)
            cumulative_metric[idx // LOG_FREQ] = model.frequentist_eval.mean()
            non_cumulative_metric[idx // LOG_FREQ] = metric_val[topk_arms].mean()

        if idx % CALIBRATION_FREQ == 0:
            # before calibration
            if idx == 0:
                holdout_calibrated_ece[idx] = eval_ece(holdout_confidences, holdout_observations, num_bins=10)
            else:
                calibration_model = CALIBRATION_MODELS[args.calibration_model](mode='equal_width')
                X = np.array(confidences[:idx])
                X = np.array([1 - X, X]).T
                y = np.array(observations[:idx]) * 1.0

                calibration_model.fit(X, y)
                calibrated_holdout_confidences = calibration_model.predict_proba(holdout_X)[:, 1].tolist()
                holdout_calibrated_ece[idx // CALIBRATION_FREQ] = eval_ece(calibrated_holdout_confidences,
                                                                           holdout_observations, num_bins=10)

    if args.metric == 'accuracy':
        return avg_num_agreement, cumulative_metric, non_cumulative_metric
    elif args.metric == 'calibration_error':
        return avg_num_agreement, cumulative_metric, non_cumulative_metric, holdout_calibrated_ece


def _comparison_plot(eval_result_dict: Dict[str, np.ndarray], eval_freq: int, figname: str, ylabel: str) -> None:
    # If labels are getting cut off make the figsize smaller
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO), dpi=300)

    for method_name, metric_eval in eval_result_dict.items():
        metric_eval = metric_eval[:-1]
        x = np.arange(len(metric_eval)) * eval_freq
        plt.plot(x, metric_eval, label=method_name)
    plt.xlabel('#Queries')
    plt.ylabel(ylabel)
    plt.legend()
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    # plt.ylim(0.0, 1.0)
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
        method_list = ['non-active', 'ts_uniform', 'ts_informed']
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
                         args.output / experiment_name / "holdout_ece.pdf", 'ECE')


def main_accuracy_topk(args: argparse.Namespace, SAMPLE=True, EVAL=True, PLOT=True) -> None:
    num_classes = num_classes_dict[args.dataset]

    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile_dict[args.dataset], False)

    num_samples = len(observations)

    UNIFORM_PRIOR = np.ones((num_classes, 2)) / 2 * PRIOR_STRENGTH
    confidence = _get_confidence_k(categories, confidences, num_classes)
    INFORMED_PRIOR = np.array([confidence, 1 - confidence]).T * PRIOR_STRENGTH

    experiment_name = '%s_%s_%s_top%d_runs%d' % (args.dataset, args.metric, args.mode, args.topk, RUNS)

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

    avg_num_agreement_dict = {
        'non-active': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }
    cumulative_metric_dict = {
        'non-active': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }
    non_cumulative_metric_dict = {
        'non-active': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }

    if SAMPLE:
        for r in tqdm(range(RUNS)):
            sampled_categories_dict['non-active'][r], sampled_observations_dict['non-active'][r], \
            sampled_scores_dict['non-active'][
                r] = get_samples_topk(args,
                                      categories,
                                      observations,
                                      confidences,
                                      num_classes,
                                      num_samples,
                                      sample_method='random',
                                      prior=UNIFORM_PRIOR,
                                      random_seed=r)

            sampled_categories_dict['ts_uniform'][r], sampled_observations_dict['ts_uniform'][r], \
            sampled_scores_dict['ts_uniform'][r] = get_samples_topk(args,
                                                                    categories,
                                                                    observations,
                                                                    confidences,
                                                                    num_classes,
                                                                    num_samples,
                                                                    sample_method='ts',
                                                                    prior=UNIFORM_PRIOR,
                                                                    random_seed=r)

            sampled_categories_dict['ts_informed'][r], sampled_observations_dict['ts_informed'][r], \
            sampled_scores_dict['ts_informed'][r] = get_samples_topk(args,
                                                                     categories,
                                                                     observations,
                                                                     confidences,
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
    else:
        # load sampled categories, scores and observations from file
        for method in ['non-active', 'ts_uniform', 'ts_informed']:
            sampled_categories_dict[method] = np.load(
                args.output / experiment_name / ('sampled_categories_%s.npy' % method))
            sampled_observations_dict[method] = np.load(
                args.output / experiment_name / ('sampled_observations_%s.npy' % method))
            sampled_scores_dict[method] = np.load(args.output / experiment_name / ('sampled_scores_%s.npy' % method))

    if EVAL:
        ground_truth = get_ground_truth(categories, observations, confidences, num_classes, args.metric, args.mode,
                                        topk=args.topk)
        for r in tqdm(range(RUNS)):
            avg_num_agreement_dict['non-active'][r], cumulative_metric_dict['non-active'][r], \
            non_cumulative_metric_dict['non-active'][r] = eval(args,
                                                               sampled_categories_dict['non-active'][r].tolist(),
                                                               sampled_observations_dict['non-active'][r].tolist(),
                                                               sampled_scores_dict['non-active'][r].tolist(),
                                                               ground_truth,
                                                               num_classes=num_classes,
                                                               prior=UNIFORM_PRIOR)

            avg_num_agreement_dict['ts_uniform'][r], cumulative_metric_dict['ts_uniform'][r], \
            non_cumulative_metric_dict['ts_uniform'][r] = eval(args,
                                                               sampled_categories_dict['ts_uniform'][r].tolist(),
                                                               sampled_observations_dict['ts_uniform'][r].tolist(),
                                                               sampled_scores_dict['ts_uniform'][r].tolist(),
                                                               ground_truth,
                                                               num_classes=num_classes,
                                                               prior=UNIFORM_PRIOR)

            avg_num_agreement_dict['ts_informed'][r], cumulative_metric_dict['ts_informed'][r], \
            non_cumulative_metric_dict['ts_informed'][r] = eval(args,
                                                                sampled_categories_dict['ts_informed'][r].tolist(),
                                                                sampled_observations_dict['ts_informed'][r].tolist(),
                                                                sampled_scores_dict['ts_informed'][r].tolist(),
                                                                ground_truth,
                                                                num_classes=num_classes,
                                                                prior=INFORMED_PRIOR)

        for method in ['non-active', 'ts_uniform', 'ts_informed']:
            np.save(args.output / experiment_name / ('avg_num_agreement_%s.npy' % method),
                    avg_num_agreement_dict[method])
            np.save(args.output / experiment_name / ('cumulative_metric_%s.npy' % method),
                    cumulative_metric_dict[method])
            np.save(args.output / experiment_name / ('non_cumulative_metric_%s.npy' % method),
                    non_cumulative_metric_dict[method])
    else:
        for method in ['non-active', 'ts_uniform', 'ts_informed']:
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
    categories, observations, confidences, idx2category, category2idx = prepare_data(datafile_dict[args.dataset], False)
    categories, observations, confidences, holdout_categories, holdout_observations, holdout_confidences = train_holdout_split(
        categories, observations, confidences,
        holdout_ratio=0.2)

    num_samples = len(observations)

    experiment_name = '%s_%s_%s_top%d_runs%d' % (args.dataset, args.metric, args.mode, args.topk, RUNS)

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

    avg_num_agreement_dict = {
        'non-active': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }
    cumulative_metric_dict = {
        'non-active': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }
    non_cumulative_metric_dict = {
        'non-active': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }

    holdout_ece_dict = {
        'non-active': np.zeros((RUNS, num_samples // CALIBRATION_FREQ + 1)),
        'ts': np.zeros((RUNS, num_samples // CALIBRATION_FREQ + 1)),
    }

    if SAMPLE:
        logging.info('Starting sampling')

        def sampler_worker(queue):
            # Continue to work until queue is empty
            while not queue.empty():
                # Get a job (e.g. run index)
                run_idx, sample_method = queue.get()
                print(run_idx)

                # @disiji - update if you change the name...
                if sample_method == 'non-active':
                    method = 'random'
                else:
                    method = sample_method

                sampled_categories, sampled_observations, sampled_scores = get_samples_topk(args,
                                                                                            categories,
                                                                                            observations,
                                                                                            confidences,
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

                queue.task_done()

        # Enqueue tasks
        logging.info('Enqueueing tasks')
        random_job_queue = JoinableQueue()
        ts_job_queue = JoinableQueue()
        for i in range(RUNS):
            random_job_queue.put((i, 'non-active'))
            ts_job_queue.put((i, 'ts'))

        # Start random workers
        logging.info('Running non-active sampling')
        for _ in range(args.sample_processes):
            process = Process(target=sampler_worker, args=(random_job_queue,))
            process.start()

        # Make sure all random workers are done before proceeding
        random_job_queue.join()

        # Start ts workers
        logging.info('Running Thompson sampling')
        for _ in range(args.sample_processes):
            process = Process(target=sampler_worker, args=(ts_job_queue,))
            process.start()

        # Make sure all ts workers are done before proceeding
        ts_job_queue.join()

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

        # Write to disk
        for method in ['non-active', 'ts']:
            np.save(args.output / experiment_name / ('sampled_categories_%s.npy' % method),
                    sampled_categories_dict[method])
            np.save(args.output / experiment_name / ('sampled_observations_%s.npy' % method),
                    sampled_observations_dict[method])
            np.save(args.output / experiment_name / ('sampled_scores_%s.npy' % method),
                    sampled_scores_dict[method])
    else:
        # load sampled categories, scores and observations from file
        for method in ['non-active', 'ts']:
            sampled_categories_dict[method] = np.load(
                args.output / experiment_name / ('sampled_categories_%s.npy' % method))
            sampled_observations_dict[method] = np.load(
                args.output / experiment_name / ('sampled_observations_%s.npy' % method))
            sampled_scores_dict[method] = np.load(args.output / experiment_name / ('sampled_scores_%s.npy' % method))

    if EVAL:
        logging.info('Starting evaluation')
        ground_truth = get_ground_truth(categories, observations, confidences, num_classes, args.metric, args.mode,
                                        topk=args.topk)
        for r in tqdm(range(RUNS)):
            avg_num_agreement_dict['non-active'][r], \
            cumulative_metric_dict['non-active'][r], \
            non_cumulative_metric_dict['non-active'][r], \
            holdout_ece_dict['non-active'][r] = eval(args,
                                                     sampled_categories_dict['non-active'][r].tolist(),
                                                     sampled_observations_dict['non-active'][r].tolist(),
                                                     sampled_scores_dict['non-active'][r].tolist(),
                                                     ground_truth,
                                                     num_classes=num_classes,
                                                     holdout_categories=holdout_categories,
                                                     holdout_observations=holdout_observations,
                                                     holdout_confidences=holdout_confidences,
                                                     prior=None)

            avg_num_agreement_dict['ts'][r], \
            cumulative_metric_dict['ts'][r], \
            non_cumulative_metric_dict['ts'][r], \
            holdout_ece_dict['ts'][r] = eval(args,
                                             sampled_categories_dict['ts'][r].tolist(),
                                             sampled_observations_dict['ts'][r].tolist(),
                                             sampled_scores_dict['ts'][r].tolist(),
                                             ground_truth,
                                             num_classes=num_classes,
                                             holdout_categories=holdout_categories,
                                             holdout_observations=holdout_observations,
                                             holdout_confidences=holdout_confidences,
                                             prior=None)

        for method in ['non-active', 'ts']:
            np.save(args.output / experiment_name / ('avg_num_agreement_%s.npy' % method),
                    avg_num_agreement_dict[method])
            np.save(args.output / experiment_name / ('cumulative_metric_%s.npy' % method),
                    cumulative_metric_dict[method])
            np.save(args.output / experiment_name / ('non_cumulative_metric_%s.npy' % method),
                    non_cumulative_metric_dict[method])
            np.save(args.output / experiment_name / ('holdout_ece_%s.npy' % method),
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
                args.output / experiment_name / ('holdout_ece_%s.npy' % method))

    if PLOT:
        comparison_plot(args, experiment_name, avg_num_agreement_dict, cumulative_metric_dict,
                        non_cumulative_metric_dict, holdout_ece_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default='cifar100', help='input dataset')
    parser.add_argument('--output', type=pathlib.Path, default=OUTPUT_DIR, help='output prefix')
    parser.add_argument('-topk', type=int, default=10, help='number of optimal arms to identify')
    parser.add_argument('-metric', type=str, help='accuracy or calibration_error')
    parser.add_argument('-mode', type=str, help='min or max, identify topk with highest/lowest reward')
    parser.add_argument('--calibration_model', type=str, default=CALIBRATION_MODEL,
                        help='calibration models to apply on holdout data')
    parser.add_argument('--sample_processes', type=int, default=4,
                        help='Number of sample processes. Increase to speed up sampling.')

    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    if args.dataset not in DATASET_LIST:
        raise ValueError("%s is not in DATASET_LIST." % args.dataset)

    print(args.dataset, args.mode, '...')
    if args.metric == 'accuracy':
        main_accuracy_topk(args, SAMPLE=True, EVAL=True, PLOT=True)
    elif args.metric == 'calibration_error':
        main_calibration_error_topk(args, SAMPLE=False, EVAL=True, PLOT=True)
