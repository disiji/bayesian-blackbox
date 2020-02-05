import argparse
import logging
import pathlib
from multiprocessing import Lock, Process, JoinableQueue

import numpy as np
from tqdm import tqdm

from active_learning_topk import get_samples_topk, eval, comparison_plot, MpSafeSharedArray
from active_utils import _get_confidence_k, get_ground_truth, get_bayesian_ground_truth
from data_utils import datafile_dict, num_classes_dict, logits_dict, prepare_data, train_holdout_split, \
    DATASET_LIST

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
CALIBRATION_MODEL = 'classwise_histogram_binning'
HOLDOUT_RATIO = 0.1

logger = logging.getLogger(__name__)
process_lock = Lock()


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
    bayesian_ucb
    sampled_categories_dict = {
        'epsilon_greedy': np.empty((RUNS, num_samples), dtype=int),
        'bayesian_ucb': np.empty((RUNS, num_samples), dtype=int),
    }
    sampled_observations_dict = {
        'epsilon_greedy': np.empty((RUNS, num_samples), dtype=bool),
        'bayesian_ucb': np.empty((RUNS, num_samples), dtype=bool),
    }
    sampled_scores_dict = {
        'epsilon_greedy': np.empty((RUNS, num_samples), dtype=float),
        'bayesian_ucb': np.empty((RUNS, num_samples), dtype=float),
    }
    sampled_labels_dict = {
        'epsilon_greedy': np.empty((RUNS, num_samples), dtype=int),
        'bayesian_ucb': np.empty((RUNS, num_samples), dtype=int),
    }
    sampled_indices_dict = {
        'epsilon_greedy': np.empty((RUNS, num_samples), dtype=int),
        'bayesian_ucb': np.empty((RUNS, num_samples), dtype=int),
    }

    avg_num_agreement_dict = {
        'epsilon_greedy_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'epsilon_greedy_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'epsilon_greedy_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }
    cumulative_metric_dict = {
        'epsilon_greedy_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'epsilon_greedy_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'epsilon_greedy_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }
    non_cumulative_metric_dict = {
        'epsilon_greedy_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'epsilon_greedy_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'epsilon_greedy_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }
    mrr_dict = {
        'epsilon_greedy_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'epsilon_greedy_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'epsilon_greedy_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'bayesian_ucb_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }

    if SAMPLE:
        for r in tqdm(range(RUNS)):
            sampled_categories_dict['epsilon_greedy'][r], sampled_observations_dict['epsilon_greedy'][r], \
            sampled_scores_dict['epsilon_greedy'][r], sampled_labels_dict['epsilon_greedy'][r], \
            sampled_indices_dict['epsilon_greedy'][r] = get_samples_topk(args,
                                                                         categories,
                                                                         observations,
                                                                         confidences,
                                                                         labels,
                                                                         indices,
                                                                         num_classes,
                                                                         num_samples,
                                                                         sample_method='epsilon_greedy',
                                                                         prior=UNIFORM_PRIOR * 1e-6,
                                                                         random_seed=r)
            sampled_categories_dict['bayesian_ucb'][r], sampled_observations_dict['bayesian_ucb'][r], \
            sampled_scores_dict['bayesian_ucb'][r], sampled_labels_dict['bayesian_ucb'][r], \
            sampled_indices_dict['bayesian_ucb'][r] = get_samples_topk(args,
                                                                       categories,
                                                                       observations,
                                                                       confidences,
                                                                       labels,
                                                                       indices,
                                                                       num_classes,
                                                                       num_samples,
                                                                       sample_method='bayesian_ucb',
                                                                       prior=UNIFORM_PRIOR * 1e-6,
                                                                       random_seed=r)
        # write samples to file
        for method in ['epsilon_greedy', 'bayesian_ucb']:
            np.save(args.output / experiment_name / ('sampled_categories_%s.npy' % method),
                    sampled_categories_dict[method])
            np.save(args.output / experiment_name / ('sampled_observations_%s.npy' % method),
                    sampled_observations_dict[method])
            np.save(args.output / experiment_name / ('sampled_scores_%s.npy' % method), sampled_scores_dict[method])
            np.save(args.output / experiment_name / ('sampled_labels_%s.npy' % method), sampled_labels_dict[method])
            np.save(args.output / experiment_name / ('sampled_indices_%s.npy' % method), sampled_indices_dict[method])
    else:
        # load sampled categories, scores and observations from file
        for method in ['epsilon_greedy', 'bayesian_ucb']:
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
            avg_num_agreement_dict['epsilon_greedy_no_prior'][r], cumulative_metric_dict['epsilon_greedy_no_prior'][r], \
            non_cumulative_metric_dict['epsilon_greedy_no_prior'][r], mrr_dict['epsilon_greedy_no_prior'][r] = eval(
                args,
                sampled_categories_dict['epsilon_greedy'][r].tolist(),
                sampled_observations_dict['epsilon_greedy'][r].tolist(),
                sampled_scores_dict['epsilon_greedy'][r].tolist(),
                sampled_labels_dict['epsilon_greedy'][r].tolist(),
                sampled_indices_dict['epsilon_greedy'][r].tolist(),
                ground_truth,
                num_classes=num_classes,
                prior=UNIFORM_PRIOR * 1e-6)
            avg_num_agreement_dict['epsilon_greedy_uniform'][r], cumulative_metric_dict['epsilon_greedy_uniform'][r], \
            non_cumulative_metric_dict['epsilon_greedy_uniform'][r], mrr_dict['epsilon_greedy_uniform'][r] = eval(
                args,
                sampled_categories_dict['epsilon_greedy'][r].tolist(),
                sampled_observations_dict['epsilon_greedy'][r].tolist(),
                sampled_scores_dict['epsilon_greedy'][r].tolist(),
                sampled_labels_dict['epsilon_greedy'][r].tolist(),
                sampled_indices_dict['epsilon_greedy'][r].tolist(),
                ground_truth,
                num_classes=num_classes,
                prior=UNIFORM_PRIOR)
            avg_num_agreement_dict['epsilon_greedy_informed'][r], cumulative_metric_dict['epsilon_greedy_informed'][r], \
            non_cumulative_metric_dict['epsilon_greedy_informed'][r], mrr_dict['epsilon_greedy_informed'][r] = eval(
                args,
                sampled_categories_dict['epsilon_greedy'][r].tolist(),
                sampled_observations_dict['epsilon_greedy'][r].tolist(),
                sampled_scores_dict['epsilon_greedy'][r].tolist(),
                sampled_labels_dict['epsilon_greedy'][r].tolist(),
                sampled_indices_dict['epsilon_greedy'][r].tolist(),
                ground_truth,
                num_classes=num_classes,
                prior=INFORMED_PRIOR)
            avg_num_agreement_dict['bayesian_ucb_no_prior'][r], cumulative_metric_dict['bayesian_ucb_no_prior'][r], \
            non_cumulative_metric_dict['bayesian_ucb_no_prior'][r], mrr_dict['bayesian_ucb_no_prior'][r] = eval(
                args,
                sampled_categories_dict['bayesian_ucb'][r].tolist(),
                sampled_observations_dict['bayesian_ucb'][r].tolist(),
                sampled_scores_dict['bayesian_ucb'][r].tolist(),
                sampled_labels_dict['bayesian_ucb'][r].tolist(),
                sampled_indices_dict['bayesian_ucb'][r].tolist(),
                ground_truth,
                num_classes=num_classes,
                prior=UNIFORM_PRIOR * 1e-6)
            avg_num_agreement_dict['bayesian_ucb_uniform'][r], cumulative_metric_dict['bayesian_ucb_uniform'][r], \
            non_cumulative_metric_dict['bayesian_ucb_uniform'][r], mrr_dict['bayesian_ucb_uniform'][r] = eval(
                args,
                sampled_categories_dict['bayesian_ucb'][r].tolist(),
                sampled_observations_dict['bayesian_ucb'][r].tolist(),
                sampled_scores_dict['bayesian_ucb'][r].tolist(),
                sampled_labels_dict['bayesian_ucb'][r].tolist(),
                sampled_indices_dict['bayesian_ucb'][r].tolist(),
                ground_truth,
                num_classes=num_classes,
                prior=UNIFORM_PRIOR)
            avg_num_agreement_dict['bayesian_ucb_informed'][r], cumulative_metric_dict['bayesian_ucb_informed'][r], \
            non_cumulative_metric_dict['bayesian_ucb_informed'][r], mrr_dict['bayesian_ucb_informed'][r] = eval(
                args,
                sampled_categories_dict['bayesian_ucb'][r].tolist(),
                sampled_observations_dict['bayesian_ucb'][r].tolist(),
                sampled_scores_dict['bayesian_ucb'][r].tolist(),
                sampled_labels_dict['bayesian_ucb'][r].tolist(),
                sampled_indices_dict['bayesian_ucb'][r].tolist(),
                ground_truth,
                num_classes=num_classes,
                prior=INFORMED_PRIOR)

        for method in ['epsilon_greedy_no_prior', 'epsilon_greedy_uniform', 'epsilon_greedy_informed',
                       'bayesian_ucb_no_prior', 'bayesian_ucb_uniform', 'bayesian_ucb_informed']:
            np.save(args.output / experiment_name / ('avg_num_agreement_%s.npy' % method),
                    avg_num_agreement_dict[method])
            np.save(args.output / experiment_name / ('cumulative_metric_%s.npy' % method),
                    cumulative_metric_dict[method])
            np.save(args.output / experiment_name / ('non_cumulative_metric_%s.npy' % method),
                    non_cumulative_metric_dict[method])
            np.save(args.output / experiment_name / ('mrr_%s.npy' % method), mrr_dict[method])
    else:
        for method in ['epsilon_greedy_no_prior', 'epsilon_greedy_uniform', 'epsilon_greedy_informed',
                       'bayesian_ucb_no_prior', 'bayesian_ucb_uniform', 'bayesian_ucb_informed']:
            avg_num_agreement_dict[method] = np.load(
                args.output / experiment_name / ('avg_num_agreement_%s.npy' % method))
            cumulative_metric_dict[method] = np.load(
                args.output / experiment_name / ('cumulative_metric_%s.npy' % method))
            non_cumulative_metric_dict[method] = np.load(
                args.output / experiment_name / ('non_cumulative_metric_%s.npy' % method))
            mrr_dict[method] = np.load(args.output / experiment_name / ('mrr_%s.npy' % method))

    if PLOT:
        comparison_plot(args, experiment_name, avg_num_agreement_dict, cumulative_metric_dict,
                        non_cumulative_metric_dict, mrr_dict=mrr_dict, is_baseline=True)


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
        'epsilon_greedy': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
        'bayesian_ucb': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
    }
    sampled_observations_dict = {
        'epsilon_greedy': MpSafeSharedArray((RUNS, num_samples), dtype=np.bool),
        'bayesian_ucb': MpSafeSharedArray((RUNS, num_samples), dtype=np.bool),
    }
    sampled_scores_dict = {
        'epsilon_greedy': MpSafeSharedArray((RUNS, num_samples), dtype=np.float),
        'bayesian_ucb': MpSafeSharedArray((RUNS, num_samples), dtype=np.float),
    }
    sampled_labels_dict = {
        'epsilon_greedy': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
        'bayesian_ucb': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
    }
    sampled_indices_dict = {
        'epsilon_greedy': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
        'bayesian_ucb': MpSafeSharedArray((RUNS, num_samples), dtype=np.int),
    }

    avg_num_agreement_dict = {
        'epsilon_greedy': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
        'bayesian_ucb': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
    }
    mrr_dict = {
        'epsilon_greedy': MpSafeSharedArray((RUNS, num_samples // CALIBRATION_FREQ + 1), dtype=np.float),
        'bayesian_ucb': MpSafeSharedArray((RUNS, num_samples // CALIBRATION_FREQ + 1), dtype=np.float),
    }
    cumulative_metric_dict = {
        'epsilon_greedy': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
        'bayesian_ucb': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
    }
    non_cumulative_metric_dict = {
        'epsilon_greedy': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
        'bayesian_ucb': MpSafeSharedArray((RUNS, num_samples // LOG_FREQ + 1), dtype=np.float),
    }
    holdout_ece_dict = {
        'epsilon_greedy': MpSafeSharedArray((RUNS, num_samples // CALIBRATION_FREQ + 1), dtype=np.float),
        'bayesian_ucb': MpSafeSharedArray((RUNS, num_samples // CALIBRATION_FREQ + 1), dtype=np.float),
    }

    if SAMPLE:
        logger.info('Starting sampling')

        def sampler_worker(queue):
            # Continue to work until queue is empty
            while not queue.empty():
                # Get a job (e.g. run index)
                run_idx, sample_method = queue.get()

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
            sampling_job_queue.put((i, 'epsilon_greedy'))
            sampling_job_queue.put((i, 'bayesian_ucb'))

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
        for method in ['epsilon_greedy', 'bayesian_ucb']:
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
        for method in ['epsilon_greedy', 'bayesian_ucb']:
            np.save(args.output / experiment_name / ('sampled_categories_%s.npy' % method),
                    sampled_categories_dict[method])
            np.save(args.output / experiment_name / ('sampled_observations_%s.npy' % method),
                    sampled_observations_dict[method])
            np.save(args.output / experiment_name / ('sampled_scores_%s.npy' % method), sampled_scores_dict[method])
            np.save(args.output / experiment_name / ('sampled_labels_%s.npy' % method), sampled_labels_dict[method])
            np.save(args.output / experiment_name / ('sampled_indices_%s.npy' % method), sampled_indices_dict[method])
    else:
        # load sampled categories, scores and observations from file
        for method in ['epsilon_greedy', 'bayesian_ucb']:
            sampled_categories_dict[method] = np.load(
                args.output / experiment_name / ('sampled_categories_%s.npy' % method))
            sampled_observations_dict[method] = np.load(
                args.output / experiment_name / ('sampled_observations_%s.npy' % method))
            sampled_scores_dict[method] = np.load(args.output / experiment_name / ('sampled_scores_%s.npy' % method))
            sampled_labels_dict[method] = np.load(args.output / experiment_name / ('sampled_labels_%s.npy' % method))
            sampled_indices_dict[method] = np.load(args.output / experiment_name / ('sampled_indices_%s.npy' % method))

    if EVAL:
        logger.info('Starting evaluation')
        ground_truth = get_bayesian_ground_truth(categories, observations, confidences, num_classes, args.metric,
                                                 args.mode, topk=args.topk, pseudocount=args.pseudocount)

        def eval_worker(queue):
            while not queue.empty():
                run_idx, method = queue.get()
                with process_lock:
                    logger.debug(f'Working on eval task :: Run: {run_idx} :: Method {method}')
                agreement, metric, noncum_metric, ece, mrr = eval(args,
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
                                                                  holdout_indices=holdout_indices)

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
                mrr_array = mrr_dict[method]
                with mrr_array.get_lock():
                    arr = mrr_array.get_array()
                    arr[run_idx] = mrr

                queue.task_done()

        # Enqueue tasks
        logger.debug('Enqueueing evaluation tasks')
        eval_job_queue = JoinableQueue()
        for i in range(RUNS):
            eval_job_queue.put((i, 'epsilon_greedy'))
            eval_job_queue.put((i, 'bayesian_ucb'))

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
        for method in ['epsilon_greedy', 'bayesian_ucb']:
            with avg_num_agreement_dict[method].get_lock():
                avg_num_agreement_dict[method] = avg_num_agreement_dict[method].get_array()
            with cumulative_metric_dict[method].get_lock():
                cumulative_metric_dict[method] = cumulative_metric_dict[method].get_array()
            with non_cumulative_metric_dict[method].get_lock():
                non_cumulative_metric_dict[method] = non_cumulative_metric_dict[method].get_array()
            with holdout_ece_dict[method].get_lock():
                holdout_ece_dict[method] = holdout_ece_dict[method].get_array()
            with mrr_dict[method].get_lock():
                mrr_dict[method] = mrr_dict[method].get_array()

        for method in ['epsilon_greedy', 'bayesian_ucb']:
            np.save(args.output / experiment_name / ('avg_num_agreement_%s.npy' % method),
                    avg_num_agreement_dict[method])
            np.save(args.output / experiment_name / ('mrr_%s.npy' % method), mrr_dict[method])
            np.save(args.output / experiment_name / ('cumulative_metric_%s.npy' % method),
                    cumulative_metric_dict[method])
            np.save(args.output / experiment_name / ('non_cumulative_metric_%s.npy' % method),
                    non_cumulative_metric_dict[method])
            np.save(args.output / experiment_name / ('holdout_ece_%s_%s.npy' % (args.calibration_model, method)),
                    holdout_ece_dict[method])

    else:
        for method in ['epsilon_greedy', 'bayesian_ucb']:
            avg_num_agreement_dict[method] = np.load(
                args.output / experiment_name / ('avg_num_agreement_%s.npy' % method))
            mrr_dict[method] = np.load(
                args.output / experiment_name / ('mrr_%s.npy' % method))
            cumulative_metric_dict[method] = np.load(
                args.output / experiment_name / ('cumulative_metric_%s.npy' % method))
            non_cumulative_metric_dict[method] = np.load(
                args.output / experiment_name / ('non_cumulative_metric_%s.npy' % method))
            holdout_ece_dict[method] = np.load(
                args.output / experiment_name / ('holdout_ece_%s_%s.npy' % (args.calibration_model, method)))

    if PLOT:
        comparison_plot(args, experiment_name, avg_num_agreement_dict, cumulative_metric_dict,
                        non_cumulative_metric_dict, holdout_ece_dict, mrr_dict=mrr_dict, is_baseline=True)


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
        main_calibration_error_topk(args, SAMPLE=True, EVAL=True, PLOT=True)
