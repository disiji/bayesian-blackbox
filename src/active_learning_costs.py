import argparse
import logging
import pathlib
from collections import deque, defaultdict
from typing import Callable, Deque, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from active_learning_topk import mean_reciprocal_rank
from data_utils import CIFAR100_SUPERCLASS_LOOKUP, DATAFILE_LIST, COST_MATRIX_FILE_DICT
from data_utils import RESULTS_DIR
from models import DirichletMultinomialCost, Model

OUTPUT_DIR = RESULTS_DIR + 'costs/cifar100'

logger = logging.getLogger(__name__)

LOG_FREQ = 10
N_SIMULATIONS = 100


class Dataset:
    def __init__(self,
                 labels: np.ndarray,
                 scores: np.ndarray) -> None:
        self.labels = labels
        self.scores = scores

    def __len__(self):
        return self.labels.shape[0]

    def enqueue(self) -> List[Deque[int]]:
        queues = [deque() for _ in range(self.num_classes)]
        for label, prediction in zip(self.labels, self.predictions):
            queues[prediction].append(label)
        return queues

    def shuffle(self) -> None:
        # To make sure the rows still align we shuffle an array of indices, and use these to
        # re-order the dataset's attributes.
        shuffle_ids = np.arange(self.labels.shape[0])
        np.random.shuffle(shuffle_ids)
        self.labels = self.labels[shuffle_ids]
        self.scores = self.scores[shuffle_ids]

    @classmethod
    def load_from_text(cls, fname: pathlib.Path) -> 'Dataset':
        """
        Load dataset from a text file. Assumed format is:

            correct_class score_0 ... score_k

        """
        array = np.genfromtxt(fname)
        labels = array[:, 0].astype(np.int)
        scores = array[:, 1:].astype(np.float)
        return cls(labels, scores)

    @property
    def num_classes(self) -> int:
        return self.scores.shape[-1]

    @property
    def confusion_probs(self) -> np.ndarray:
        arr = confusion_matrix(self.labels, self.predictions).transpose()
        return arr / arr.sum(axis=-1, keepdims=True)

    @property
    def confusion_prior(self) -> np.ndarray:
        arr = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            arr[i] = self.scores[self.predictions == i].sum(axis=0) / sum(self.predictions == i)
        return arr

    @property
    def predictions(self) -> np.ndarray:
        return np.argmax(self.scores, axis=-1)


class SuperclassDataset:
    def __init__(self,
                 labels: np.ndarray,
                 scores: np.ndarray,
                 superclass_lookup: Dict[int, int]) -> None:
        self.labels = labels
        self.scores = scores
        self.superclass_lookup = superclass_lookup
        self.reverse_lookup = defaultdict(list)
        for key, value in self.superclass_lookup.items():
            self.reverse_lookup[value].append(key)

    def __len__(self):
        return self.labels.shape[0]

    def shuffle(self) -> None:
        # To make sure the rows still align we shuffle an array of indices, and use these to
        # re-order the dataset's attributes.
        shuffle_ids = np.arange(self.labels.shape[0])
        np.random.shuffle(shuffle_ids)
        self.labels = self.labels[shuffle_ids]
        self.scores = self.scores[shuffle_ids]

    def generate(self) -> Iterable[Tuple[int, int]]:
        for label, prediction in zip(self.labels, self.predictions):
            if label == prediction:
                entry = 0
            elif self.superclass_lookup[label] == self.superclass_lookup[prediction]:
                entry = 1
            else:
                entry = 2
            yield prediction, entry

    def enqueue(self) -> List[Deque[int]]:
        queues = [deque() for _ in range(self.num_classes)]
        for prediction, entry in self.generate():
            queues[prediction].append(entry)
        return queues

    @classmethod
    def load_from_text(cls,
                       fname: pathlib.Path,
                       superclass_lookup: Dict[int, int]) -> 'Dataset':
        """
        Load dataset from a text file. Assumed format is:

            correct_class score_0 ... score_k

        """
        array = np.genfromtxt(fname)
        labels = array[:, 0].astype(np.int)
        scores = array[:, 1:].astype(np.float)
        return cls(labels, scores, superclass_lookup)

    @property
    def num_classes(self) -> int:
        return self.scores.shape[-1]

    @property
    def confusion_probs(self) -> np.ndarray:
        arr = np.zeros((self.num_classes, 3))
        for prediction, entry in self.generate():
            arr[prediction, entry] += 1
        return arr / arr.sum(axis=-1, keepdims=True)

    @property
    def confusion_prior(self) -> np.ndarray:
        arr = np.zeros((self.num_classes, 3))
        for class_idx in range(self.num_classes):
            mean_scores = self.scores[self.predictions == class_idx].mean(axis=0)
            # Correct prediction prob
            arr[class_idx, 0] = mean_scores[class_idx]
            # Within superclass confusion prob
            superclass_idx = self.superclass_lookup[class_idx]
            for other_class_idx in self.reverse_lookup[superclass_idx]:
                if other_class_idx == class_idx:
                    continue
                arr[class_idx, 1] += mean_scores[other_class_idx]
        # Law of total probability
        arr[:, 2] = 1 - arr[:, 0] - arr[:, 1]
        return arr

    @property
    def predictions(self) -> np.ndarray:
        return np.argmax(self.scores, axis=-1)


def random_choice_fn(sample: np.ndarray) -> np.ndarray:
    ids = np.arange(sample.shape[0])
    np.random.shuffle(ids)
    return ids


def max_choice_fn(sample: np.ndarray) -> np.ndarray:
    return np.argsort(sample)[::-1]


def select_and_label(dataset: Dataset,
                     model: Model,
                     topk: int,
                     choice_fn: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects data points from dataset according to criterion and updates the model.

    Parameters
    ==========
    dataset : Dataset
        Dataset of predictions and observations.
    model : Model
        Bayesian assessment model.
    choice_fn : Callable
        Function used to identify the next class to be labeled.
    """
    # Initialize outputs

    # Shuffle the dataset and enqueue queries
    dataset.shuffle()
    queues = dataset.enqueue()

    n_samples = len(dataset)

    mpe = np.zeros((n_samples // LOG_FREQ, dataset.num_classes))
    confusion_log = np.zeros((n_samples // LOG_FREQ, dataset.num_classes, dataset.num_classes))

    # Run experiment
    i = 0
    while i < n_samples:
        sample = model.sample()
        choices = choice_fn(sample)

        candidates = [choice for choice in choices if len(queues[choice]) > 0]
        if len(candidates) < topk:
            topk = 1

        for idx in range(topk):
            choice = candidates[idx]
            observation = queues[choice].pop()
            model.update(choice, observation)

            i += 1
            if not i % LOG_FREQ:
                index = i // LOG_FREQ - 1
                mpe[index] = model.mpe()
                confusion_log[index] = model.confusion_matrix()

    # In case we're one short
    mpe[-1] = model.mpe()

    return mpe, confusion_log


def pretty_print(arr):
    for row in arr:
        out = ' '.join('%0.4f' % x for x in row.tolist())
        print(out)


def eval(results: np.ndarray, ground_truth: list, topk: int) -> Dict[str, np.ndarray]:
    """

    :param results:(num_runs, num_samples // LOG_FREQ, num_classes)
    :param ground_truth: list of integers of length topk. Ground truth of topk classes.
    :param topk: int
    :return:
    """
    assert len(ground_truth) == topk
    num_runs, num_evals, num_classes = results.shape
    avg_num_agreement = [None] * num_evals
    mrr = [None] * num_evals

    ground_truth_array = np.zeros((num_classes,), dtype=np.bool_)
    ground_truth_array[np.array(ground_truth).astype(int)] = 1

    for idx in range(num_evals):
        current_result = results[:, idx, :]
        topk_arms = np.argsort(current_result, axis=-1)[:, -topk:]
        topk_list = topk_arms.flatten().tolist()
        avg_num_agreement[idx] = len([arm for arm in topk_list if arm in ground_truth]) * 1.0 / (
                topk * num_runs)
        mrr[idx] = sum([mean_reciprocal_rank(results[run_id, idx, :], ground_truth_array, 'max') for run_id in
                        range(num_runs)]) / num_runs
    return {
        'avg_num_agreement': avg_num_agreement,
        'mrr': mrr,
    }


# 01 loss
# Informative priors...avg predicted confidences by predicted class

def main(args: argparse.Namespace) -> None:
    # Set random seed to ensure reproducibility of experiments
    np.random.seed(args.seed)

    if not args.output.exists():
        args.output.mkdir()

    # Load the dataset and cost matrix
    if args.superclass:
        dataset = SuperclassDataset.load_from_text(DATAFILE_LIST[args.dataset], CIFAR100_SUPERCLASS_LOOKUP)
    else:
        dataset = Dataset.load_from_text(DATAFILE_LIST[args.dataset])

    cost_matrix = COST_MATRIX_FILE_DICT[args.type_cost]

    if cost_matrix is None:
        if args.superclass:
            costs = np.zeros((dataset.num_classes, 3))
            costs[:, 1] = 1
            costs[:, 2] = args.k
        else:
            # Randomly fill cost matrix with integers between 1 and 5 w/ zeros on diagonal.
            # costs = np.random.randint(1, 5, size=(dataset.num_classes, dataset.num_classes))
            costs = np.ones((dataset.num_classes, dataset.num_classes))
            costs[:, -1] = args.k * costs[:, -1]
            np.fill_diagonal(costs, 0)
    else:
        costs = np.load(cost_matrix)
    logging.info('Cost matrix:\n%s', costs)

    # Determine the highest cost predicted classes
    expected_costs = (dataset.confusion_probs * costs).sum(axis=-1)
    ground_truth = expected_costs.argsort()[-args.topk:][::-1].tolist()

    cost_string = '\n'.join('%i  %0.4f' % x for x in enumerate(expected_costs))
    logging.info('TopK highest expected cost predicted class: %i', *ground_truth)
    logging.info('Classwise expected costs:\n%s', cost_string)

    # Run experiments...
    # stores MPE of classwise cost after every LOG_FREQ steps for each run...
    random_no_prior_results = np.zeros((N_SIMULATIONS, len(dataset) // LOG_FREQ, dataset.num_classes))
    random_uniform_results = np.zeros((N_SIMULATIONS, len(dataset) // LOG_FREQ, dataset.num_classes))
    random_informed_results = np.zeros((N_SIMULATIONS, len(dataset) // LOG_FREQ, dataset.num_classes))
    active_uniform_results = np.zeros((N_SIMULATIONS, len(dataset) // LOG_FREQ, dataset.num_classes))
    active_informed_results = np.zeros((N_SIMULATIONS, len(dataset) // LOG_FREQ, dataset.num_classes))

    if args.superclass:
        # will note enter this branch for now...
        args.pseudocount = 3

    # Sampling...
    no_prior_alphas = np.ones((dataset.num_classes, dataset.num_classes)) * 1e-3
    uniform_prior_alphas = np.ones(
        (dataset.num_classes, dataset.num_classes)) * args.pseudocount / dataset.num_classes
    informed_prior_alphas = args.pseudocount * dataset.confusion_prior
    for i in tqdm(range(N_SIMULATIONS)):
        model = DirichletMultinomialCost(no_prior_alphas, costs)
        random_no_prior_results[i], random_no_prior_confusion_log = select_and_label(dataset=dataset,
                                                                                     model=model,
                                                                                     topk=args.topk,
                                                                                     choice_fn=random_choice_fn)

        model = DirichletMultinomialCost(uniform_prior_alphas, costs)
        random_uniform_results[i], random_uniform_confusion_log = select_and_label(dataset=dataset,
                                                                                   model=model,
                                                                                   topk=args.topk,
                                                                                   choice_fn=random_choice_fn)
        model = DirichletMultinomialCost(informed_prior_alphas, costs)
        random_informed_results[i], random_informed_confusion_log = select_and_label(dataset=dataset,
                                                                                     model=model,
                                                                                     topk=args.topk,
                                                                                     choice_fn=random_choice_fn)

        model = DirichletMultinomialCost(uniform_prior_alphas, costs)
        active_uniform_results[i], active_confusion_log = select_and_label(dataset=dataset,
                                                                           model=model,
                                                                           topk=args.topk,
                                                                           choice_fn=max_choice_fn)
        model = DirichletMultinomialCost(informed_prior_alphas, costs)
        active_informed_results[i], active_informed_confusion_log = select_and_label(dataset=dataset,
                                                                                     model=model,
                                                                                     topk=args.topk,
                                                                                     choice_fn=max_choice_fn)

    # Evaluation...
    random_no_prior_success = eval(random_no_prior_results, ground_truth, args.topk)['avg_num_agreement']
    random_uniform_success = eval(random_uniform_results, ground_truth, args.topk)['avg_num_agreement']
    random_informed_success = eval(random_informed_results, ground_truth, args.topk)['avg_num_agreement']
    active_success = eval(active_uniform_results, ground_truth, args.topk)['avg_num_agreement']
    active_informed_success = eval(active_informed_results, ground_truth, args.topk)['avg_num_agreement']

    random_no_prior_mrr = eval(random_no_prior_results, ground_truth, args.topk)['mrr']
    random_uniform_mrr = eval(random_uniform_results, ground_truth, args.topk)['mrr']
    random_informed_mrr = eval(random_informed_results, ground_truth, args.topk)['mrr']
    active_mrr = eval(active_uniform_results, ground_truth, args.topk)['mrr']
    active_informed_mrr = eval(active_informed_results, ground_truth, args.topk)['mrr']

    # Dump results...
    np.save(args.output / f'random_no_prior_success_top{args.topk}_pseudocount{args.pseudocount}.npy',
            random_no_prior_success)
    np.save(args.output / f'random_uniform_success_top{args.topk}_pseudocount{args.pseudocount}.npy',
            random_uniform_success)
    np.save(args.output / f'random_informed_success_top{args.topk}_pseudocount{args.pseudocount}.npy',
            random_informed_success)
    np.save(args.output / f'active_success_top{args.topk}_pseudocount{args.pseudocount}.npy', active_success)
    np.save(args.output / f'active_informed_success_top{args.topk}_pseudocount{args.pseudocount}.npy',
            active_informed_success)

    np.save(args.output / f'random_no_prior_mrr_top{args.topk}_pseudocount{args.pseudocount}.npy',
            random_no_prior_mrr)
    np.save(args.output / f'random_uniform_mrr_top{args.topk}_pseudocount{args.pseudocount}.npy',
            random_uniform_mrr)
    np.save(args.output / f'random_informed_mrr_top{args.topk}_pseudocount{args.pseudocount}.npy',
            random_informed_mrr)
    np.save(args.output / f'active_mrr_top{args.topk}_pseudocount{args.pseudocount}.npy', active_mrr)
    np.save(args.output / f'active_informed_mrr_top{args.topk}_pseudocount{args.pseudocount}.npy',
            active_informed_mrr)

    np.save(args.output / f'random_no_prior_confusion_log_top{args.topk}_pseudocount{args.pseudocount}.npy',
            random_no_prior_confusion_log)
    np.save(args.output / f'random_uniform_confusion_log_top{args.topk}_pseudocount{args.pseudocount}.npy',
            random_uniform_confusion_log)
    np.save(args.output / f'random_informed_confusion_log_top{args.topk}_pseudocount{args.pseudocount}.npy',
            random_informed_confusion_log)
    np.save(args.output / f'active_confusion_log_top{args.topk}_pseudocount{args.pseudocount}.npy',
            active_confusion_log)
    np.save(args.output / f'active_informed_confusion_log_top{args.topk}_pseudocount{args.pseudocount}.npy',
            active_informed_confusion_log)

    # Plot..
    fig, axes = plt.subplots(1, 1)
    x_axis = np.arange(len(random_no_prior_success)) * LOG_FREQ
    axes.plot(x_axis, random_no_prior_success, label='non-active(no prior)')
    axes.plot(x_axis, random_uniform_success, label='non-active(uniform prior)')
    axes.plot(x_axis, random_informed_success, label='non-active(informative prior)')
    axes.plot(x_axis, active_success, label='active (uniform prior)')
    axes.plot(x_axis, active_informed_success, label='active (informative prior)')
    axes.legend()
    plt.savefig(args.output / f'success_curve_top{args.topk}_pseudocount{args.pseudocount}.png')

    fig, axes = plt.subplots(1, 1)
    x_axis = np.arange(len(random_no_prior_mrr)) * LOG_FREQ
    axes.plot(x_axis, random_no_prior_mrr, label='non-active(no prior)')
    axes.plot(x_axis, random_uniform_mrr, label='non-active(uniform prior)')
    axes.plot(x_axis, random_informed_mrr, label='non-active(informative prior)')
    axes.plot(x_axis, active_mrr, label='active (uniform prior)')
    axes.plot(x_axis, active_informed_mrr, label='active (informative prior)')
    axes.legend()
    plt.savefig(args.output / f'mrr_curve_top{args.topk}_pseudocount{args.pseudocount}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cifar100', help='input dataset')
    parser.add_argument('-output', type=pathlib.Path, default=OUTPUT_DIR, help='output prefix')
    parser.add_argument('-topk', type=int, default=1, help='number of optimal arms to identify')
    parser.add_argument('-s', '--seed', type=int, default=1337, help='random seed')
    parser.add_argument('-type_cost', type=str, default=None, help='human or superclass')
    parser.add_argument('-pseudocount', type=float, default=1, help='pseudocount per row for confusion matrix.')
    parser.add_argument('-k', type=float, default=2, help='relative cost')
    parser.add_argument('--superclass', action='store_true')

    args, _ = parser.parse_known_args()
    args.output = args.output / args.type_cost

    logging.basicConfig(level=logging.INFO)

    main(args)
