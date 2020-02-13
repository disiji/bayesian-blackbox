import argparse
import multiprocessing
import os
import random

import numpy as np

from data_utils import DATAFILE_LIST, DATASET_LIST, prepare_data, RESULTS_DIR
from models import SumOfBetaEce

random.seed(2020)
num_cores = multiprocessing.cpu_count()
NUM_BINS = 10
NUM_RUNS = 100
N_list = [100, 200, 500, 1000, 2000, 5000, 10000]

OUTPUT_DIR = RESULTS_DIR + "bayesian_reliability_comparison/"


def main(args) -> None:
    # load data
    categories, observations, confidences, idx2category, category2idx, labels = prepare_data(
        DATAFILE_LIST[args.dataset], False)
    # train a ground_truth ece model
    if args.ground_truth_type == 'bayesian':
        ground_truth_model = SumOfBetaEce(num_bins=args.num_bins, pseudocount=args.pseudocount)
    else:
        ground_truth_model = SumOfBetaEce(num_bins=args.num_bins, pseudocount=1e-3)
    ground_truth_model.update_batch(confidences, observations)

    results = np.zeros((args.num_runs, len(N_list), 5))

    for run_id in range(args.num_runs):

        tmp = list(zip(confidences, observations))
        random.shuffle(tmp)
        confidences, observations = zip(*tmp)

        model = SumOfBetaEce(num_bins=args.num_bins, pseudocount=args.pseudocount)

        for i in range(len(N_list)):
            tmp = 0 if i == 0 else N_list[i - 1]
            model.update_batch(confidences[tmp: N_list[i]], observations[tmp: N_list[i]])

            results[run_id, i, 0] = N_list[i]
            results[run_id, i, 1] = model.eval
            results[run_id, i, 2] = model.frequentist_eval
            results[run_id, i, 3] = model.calibration_estimation_error(ground_truth_model, args.weight_type)
            results[run_id, i, 4] = model.frequentist_calibration_estimation_error(ground_truth_model, args.weight_type)

    results_mean = np.mean(results, axis=0)
    results_variance = np.std(results, axis=0)

    if args.weight_type == 'online':
        OUTPUT_DIR += "online_weights/"
    try:
        os.stat(OUTPUT_DIR)
    except:
        os.mkdir(OUTPUT_DIR)

    if args.ground_truth_type == 'frequentist':
        filename_mean = OUTPUT_DIR + "frequentist_ground_truth_%s_pseudocount%d.csv" % (args.dataset, args.pseudocount)
        filename_std = OUTPUT_DIR + "frequentist_ground_truth_%s_pseudocount%d_std.csv" % (
            args.dataset, args.pseudocount)
    else:
        filename_mean = OUTPUT_DIR + "bayesian_ground_truth_%s_pseudocount%d.csv" % (args.dataset, args.pseudocount)
        filename_std = OUTPUT_DIR + "bayesian_ground_truth_%s_pseudocount%d_std.csv" % (
            args.dataset, args.pseudocount)

    header = 'N, bayesian_ece, frequentist_ece, bayesian_estimation_error, frequentist_estimation_error'
    np.savetxt(filename_mean, results_mean, delimiter=',', header=header)
    np.savetxt(filename_std, results_variance, delimiter=',', header=header)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default='cifar100', help='input dataset')
    parser.add_argument('-pseudocount', type=int, default=1, help='strength of prior')
    parser.add_argument('-ground_truth_type', type=str, default='bayesian',
                        help='compute ground truth in a Bayesian or frequentist way, bayesian or frequentist')
    parser.add_argument('-weight_type', type=str, default='pool',
                        help='weigh each bin with all data or only data seen so far, online or pool')
    parser.add_argument('--num_runs', type=int, default=NUM_RUNS, help='number of runs')
    parser.add_argument('--num_bins', type=int, default=NUM_BINS, help='number of bins in reliability diagram')

    args, _ = parser.parse_known_args()

    if args.dataset not in DATASET_LIST:
        raise ValueError("%s is not in DATASET_LIST." % args.dataset)

    main(args)
