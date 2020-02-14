import argparse
import ctypes
import random
from collections import deque
from functools import reduce
from multiprocessing import Array

import matplotlib.pyplot as plt

from calibration import CALIBRATION_MODELS
from data_utils import *
from models import BetaBernoulli
from sampling import SAMPLE_CATEGORY

COLUMN_WIDTH = 3.25  # Inches
GOLDEN_RATIO = 1.61803398875
FONT_SIZE = 8

RUNS = 100
LOG_FREQ = 100
CALIBRATION_FREQ = 100
PRIOR_STRENGTH = 3
CALIBRATION_MODEL = 'classwise_histogram_binning'
HOLDOUT_RATIO = 0.1


#########################SAMPLE AND EVAL FOR ACTIVE TOPK##########################
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
                     random_seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        # if there are less than k available arms to play, switch to top 1.
        # If the sampling method has been switched to top1, then the return 'category_list' is an int

        # get a list of length topk
        categories_list = sample_fct(deques=deques,
                                     random_seed=random_seed,
                                     model=model,
                                     mode=args.mode,
                                     topk=topk,
                                     max_ttts_trial=50,
                                     ttts_beta=0.5,
                                     epsilon=0.1,
                                     ucb_c=1)

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


def evaluate(args: argparse.Namespace,
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
             weight=None,
             logits=None) -> Tuple[np.ndarray, ...]:
    """
    Evaluate topk ground truth agains predictions made by the model, which is trained on actively or
        non-actively selected samples.
    :return avg_num_agreement: (num_samples // LOG_FREQ, ) array.
            Average number of agreement between selected topk and ground truth topk at each step.
    :return holdout_calibrated_ece: (num_samples // CALIBRATION_FREQ , ) array.
            ECE evaluated on recalibrated holdout set.
    :return mrr: (num_samples // LOG_FREQ, ) array.
            MRR of ground truth topk at each step.
    """
    num_samples = len(categories)

    if args.metric == 'accuracy':
        model = BetaBernoulli(num_classes, prior)
    elif args.metric == 'calibration_error':
        model = ClasswiseEce(num_classes, num_bins=10, pseudocount=args.pseudocount, weight=weight)

    avg_num_agreement = np.zeros((num_samples // LOG_FREQ + 1,))
    mrr = np.zeros((num_samples // LOG_FREQ + 1,))

    if args.metric == 'calibration_error':

        holdout_calibrated_ece = np.zeros((num_samples // CALIBRATION_FREQ + 1,))

        if args.calibration_model in ['histogram_binning', 'isotonic_regression', 'bayesian_binning_quantiles',
                                      'classwise_histogram_binning', 'two_group_histogram_binning']:
            holdout_X = np.array(holdout_confidences)
            holdout_X = np.array([1 - holdout_X, holdout_X]).T

        elif args.calibration_model in ['platt_scaling', 'temperature_scaling']:
            holdout_indices_array = np.array(holdout_indices, dtype=np.int)
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

            # MRR
            mrr[idx // LOG_FREQ] = mean_reciprocal_rank(metric_val, ground_truth, args.mode)

        ########RECALIBRATION#############
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
        return avg_num_agreement, mrr
    elif args.metric == 'calibration_error':
        return avg_num_agreement, holdout_calibrated_ece, mrr


#########################PLOT##########################
def _comparison_plot(args: argparse.Namespace, eval_result_dict: Dict[str, np.ndarray], eval_freq: int, figname: str,
                     ylabel: str) -> None:
    # If labels are getting cut off make the figsize smaller
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO), dpi=300)

    if args.metric == 'calibration_error':
        total_samples = DATASIZE_DICT[args.dataset] * (1 - HOLDOUT_RATIO)
    elif args.metric == 'accuracy':
        total_samples = DATASIZE_DICT[args.dataset]

    for method_name in eval_result_dict:
        metric_eval = eval_result_dict[method_name]
        metric_eval = metric_eval[: int(len(metric_eval) / 2)]
        x = np.arange(len(metric_eval)) * eval_freq / total_samples
        plt.plot(x, metric_eval, label=method_name)
    plt.xlabel('#Percentage')
    plt.ylabel(ylabel)
    plt.legend(fontsize=FONT_SIZE - 2)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')


def comparison_plot(args: argparse.Namespace,
                    experiment_name: str,
                    avg_num_agreement_dict: Dict[str, np.ndarray],
                    holdout_ece_dict: Dict[str, np.ndarray] = None,
                    mrr_dict: Dict[str, np.ndarray] = None,
                    is_baseline: bool = False) -> None:
    """
    Simple metric comparison script.
    :param args: argparse.Namespace
    :param experiment_name: str
    :param avg_num_agreement_dict:
        Dict maps str to np.ndarray of shape (RUNS, num_samples // LOG_FREQ + 1)
    :param cumulative_metric_dict:
        Dict maps str to np.ndarray of shape (RUNS, num_samples // LOG_FREQ + 1)
    :param non_cumulative_metric_dict:
        Dict maps str to np.ndarray of shape (RUNS, num_samples // LOG_FREQ + 1)
    :return:
    """

    for method in mrr_dict:
        avg_num_agreement_dict[method] = avg_num_agreement_dict[method].mean(axis=0)
        if holdout_ece_dict:
            holdout_ece_dict[method] = holdout_ece_dict[method].mean(axis=0)
        if mrr_dict:
            mrr_dict[method] = mrr_dict[method].mean(axis=0)

    # put baseline e.g. epsilon_greedy in a separate plot since they are run separately in this experiment.
    # might merge them together later.
    if is_baseline:
        tmp = '_baseline'
    else:
        tmp = ''

    _comparison_plot(args, avg_num_agreement_dict, LOG_FREQ,
                     args.output / experiment_name / ("avg_num_agreement%s.pdf" % tmp),
                     'Avg number of agreement')
    _comparison_plot(args, mrr_dict, LOG_FREQ,
                     args.output / experiment_name / ("mrr%s.pdf" % tmp),
                     'Mean Reciprocal Rank %s' % args.metric)
    if holdout_ece_dict:
        _comparison_plot(args, holdout_ece_dict, CALIBRATION_FREQ,
                         args.output / experiment_name / ("holdout_ece_%s%s.pdf" % (args.calibration_model, tmp)),
                         'ECE')


#########################METRIC##########################
def mean_reciprocal_rank(metric_val: np.ndarray,
                         ground_truth: np.ndarray,
                         mode: str) -> float:
    """Computes mean reciprocal rank"""
    num_classes = metric_val.shape[0]
    k = np.sum(ground_truth)

    # Compute rank of each class
    argsort = metric_val.argsort()
    rank = np.empty_like(argsort)
    rank[argsort] = np.arange(num_classes) + 1
    if mode == 'max':  # Need to flip so that largest class has rank 1
        rank = num_classes - rank + 1

    # In top-k setting, we need to adjust so that other ground truth classes
    # are not considered in the ranking.
    raw_rank = rank[ground_truth]
    argsort = raw_rank.argsort()
    offset = np.empty_like(argsort)
    offset[argsort] = np.arange(k)
    adjusted_rank = raw_rank - offset

    return (1 / adjusted_rank).mean()


#########################MULTIPROCESSING##########################
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
