import pathlib
from multiprocessing import Lock, Process, JoinableQueue

from tqdm import tqdm

from utils import *

OUTPUT_DIR = RESULTS_DIR + "active_learning_topk"

logger = logging.getLogger(__name__)
process_lock = Lock()


def main_accuracy_topk(args: argparse.Namespace, sample=True, eval=True, plot=True) -> None:
    num_classes = NUM_CLASSES_DICT[args.dataset]

    categories, observations, confidences, idx2category, category2idx, labels = prepare_data(
        DATAFILE_LIST[args.dataset], False)
    indices = np.arange(len(categories))

    num_samples = len(observations)

    uniform_prior = np.ones((num_classes, 2)) / 2 * args.pseudocount
    confidence = get_confidence_k(categories, confidences, num_classes)
    informed_prior = np.array([confidence, 1 - confidence]).T * args.pseudocount

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
    mrr_dict = {
        'non-active_no_prior': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'non-active_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'non-active_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_uniform': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
        'ts_informed': np.zeros((RUNS, num_samples // LOG_FREQ + 1)),
    }

    if sample:
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
                                                                     prior=uniform_prior * 1e-6,
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
                                                                     prior=uniform_prior,
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
                                                                      prior=informed_prior,
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

    if eval:
        ground_truth = get_ground_truth(categories, observations, confidences, num_classes, args.metric, args.mode,
                                        topk=args.topk)

        for r in tqdm(range(RUNS)):
            avg_num_agreement_dict['non-active_no_prior'][r], mrr_dict['non-active_no_prior'][r] = evaluate(
                args,
                sampled_categories_dict['non-active'][r].tolist(),
                sampled_observations_dict['non-active'][r].tolist(),
                sampled_scores_dict['non-active'][r].tolist(),
                sampled_labels_dict['non-active'][r].tolist(),
                sampled_indices_dict['non-active'][r].tolist(),
                ground_truth,
                num_classes,
                prior=uniform_prior * 1e-6
            )
            avg_num_agreement_dict['non-active_uniform'][r], mrr_dict['non-active_uniform'][r] = evaluate(
                args,
                sampled_categories_dict['non-active'][r].tolist(),
                sampled_observations_dict['non-active'][r].tolist(),
                sampled_scores_dict['non-active'][r].tolist(),
                sampled_labels_dict['non-active'][r].tolist(),
                sampled_indices_dict['non-active'][r].tolist(),
                ground_truth,
                num_classes,
                prior=uniform_prior)
            avg_num_agreement_dict['non-active_informed'][r], mrr_dict['non-active_informed'][r] = evaluate(
                args,
                sampled_categories_dict['non-active'][r].tolist(),
                sampled_observations_dict['non-active'][r].tolist(),
                sampled_scores_dict['non-active'][r].tolist(),
                sampled_labels_dict['non-active'][r].tolist(),
                sampled_indices_dict['non-active'][r].tolist(),
                ground_truth,
                num_classes,
                prior=informed_prior)

            avg_num_agreement_dict['ts_uniform'][r], mrr_dict['ts_uniform'][r] = evaluate(
                args,
                sampled_categories_dict['ts_uniform'][r].tolist(),
                sampled_observations_dict['ts_uniform'][r].tolist(),
                sampled_scores_dict['ts_uniform'][r].tolist(),
                sampled_labels_dict['ts_uniform'][r].tolist(),
                sampled_indices_dict['ts_uniform'][r].tolist(),
                ground_truth,
                num_classes,
                prior=uniform_prior)

            avg_num_agreement_dict['ts_informed'][r], mrr_dict['ts_informed'][r] = evaluate(
                args,
                sampled_categories_dict['ts_informed'][r].tolist(),
                sampled_observations_dict['ts_informed'][r].tolist(),
                sampled_scores_dict['ts_informed'][r].tolist(),
                sampled_labels_dict['ts_informed'][r].tolist(),
                sampled_indices_dict['ts_informed'][r].tolist(),
                ground_truth,
                num_classes,
                prior=informed_prior)

        for method in ['non-active_no_prior', 'non-active_uniform', 'non-active_informed', 'ts_uniform', 'ts_informed']:
            np.save(args.output / experiment_name / ('avg_num_agreement_%s.npy' % method),
                    avg_num_agreement_dict[method])
            np.save(args.output / experiment_name / ('mrr_%s.npy' % method), mrr_dict[method])
    else:
        for method in ['non-active_no_prior', 'non-active_uniform', 'non-active_informed', 'ts_uniform', 'ts_informed']:
            avg_num_agreement_dict[method] = np.load(
                args.output / experiment_name / ('avg_num_agreement_%s.npy' % method))
            mrr_dict[method] = np.load(args.output / experiment_name / ('mrr_%s.npy' % method))

    if plot:
        comparison_plot(args, experiment_name, avg_num_agreement_dict, mrr_dict=mrr_dict)


def main_calibration_error_topk(args: argparse.Namespace, sample=True, eval=True, plot=True) -> None:
    num_classes = NUM_CLASSES_DICT[args.dataset]

    global logits
    logits_path = LOGITSFILE_DICT.get(args.dataset,
                                      None)  # Since we haven't created all the logits yet, assign defaul value of None.
    if logits_path is not None:
        logits = np.genfromtxt(logits_path)[:, 1:]
    else:
        logits = None

    categories, observations, confidences, idx2category, category2idx, labels = prepare_data(
        DATAFILE_LIST[args.dataset], False)
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
    mrr_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples // CALIBRATION_FREQ + 1), dtype=np.float),
        'ts': MpSafeSharedArray((RUNS, num_samples // CALIBRATION_FREQ + 1), dtype=np.float),
    }
    holdout_ece_dict = {
        'non-active': MpSafeSharedArray((RUNS, num_samples // CALIBRATION_FREQ + 1), dtype=np.float),
        'ts': MpSafeSharedArray((RUNS, num_samples // CALIBRATION_FREQ + 1), dtype=np.float),
    }

    if sample:
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

    if eval:
        logger.info('Starting evaluation')
        ground_truth = get_bayesian_ground_truth(categories, observations, confidences, num_classes, args.metric,
                                                 args.mode, topk=args.topk, pseudocount=args.pseudocount)

        def eval_worker(queue):
            while not queue.empty():
                run_idx, method = queue.get()
                with process_lock:
                    logger.debug(f'Working on eval task :: Run: {run_idx} :: Method {method}')
                agreement, ece, mrr = evaluate(args,
                                               sampled_categories_dict[method][run_idx].tolist(),
                                               sampled_observations_dict[method][run_idx].tolist(),
                                               sampled_scores_dict[method][run_idx].tolist(),
                                               sampled_labels_dict[method][run_idx].tolist(),
                                               sampled_indices_dict[method][run_idx].tolist(),
                                               ground_truth,
                                               num_classes,
                                               holdout_categories=holdout_categories,
                                               holdout_observations=holdout_observations,
                                               holdout_confidences=holdout_confidences,
                                               holdout_labels=holdout_labels,
                                               holdout_indices=holdout_indices,
                                               logits=logits)

                # Write outputs
                avg_num_agreement_array = avg_num_agreement_dict[method]
                with avg_num_agreement_array.get_lock():
                    arr = avg_num_agreement_array.get_array()
                    arr[run_idx] = agreement
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
            with holdout_ece_dict[method].get_lock():
                holdout_ece_dict[method] = holdout_ece_dict[method].get_array()
            with mrr_dict[method].get_lock():
                mrr_dict[method] = mrr_dict[method].get_array()

        for method in ['non-active', 'ts']:
            np.save(args.output / experiment_name / ('avg_num_agreement_%s.npy' % method),
                    avg_num_agreement_dict[method])
            np.save(args.output / experiment_name / ('mrr_%s.npy' % method), mrr_dict[method])
            np.save(args.output / experiment_name / ('holdout_ece_%s_%s.npy' % (args.calibration_model, method)),
                    holdout_ece_dict[method])

    else:
        for method in ['non-active', 'ts']:
            avg_num_agreement_dict[method] = np.load(
                args.output / experiment_name / ('avg_num_agreement_%s.npy' % method))
            mrr_dict[method] = np.load(
                args.output / experiment_name / ('mrr_%s.npy' % method))
            holdout_ece_dict[method] = np.load(
                args.output / experiment_name / ('holdout_ece_%s_%s.npy' % (args.calibration_model, method)))

    if plot:
        comparison_plot(args, experiment_name, avg_num_agreement_dict, holdout_ece_dict, mrr_dict=mrr_dict)


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
        main_accuracy_topk(args, sample=True, eval=True, plot=True)
    elif args.metric == 'calibration_error':
        main_calibration_error_topk(args, sample=True, eval=True, plot=True)
