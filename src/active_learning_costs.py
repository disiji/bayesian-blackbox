import argparse
from collections import deque, defaultdict
import logging
import pathlib
from typing import Callable, Deque, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from cifar100meta import superclass_lookup
from models import DirichletMultinomialCost, Model


logger = logging.getLogger(__name__)

LOG_FREQ = 1
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
        arr[:,2] = 1 - arr[:,0] - arr[:,1]
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
                     choice_fn: Callable) -> None:  # IDK what return type should be...
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

    # Run experiment
    for i in range(n_samples):
        sample = model.sample()
        choices = choice_fn(sample)
        for choice in choices:
            if queues[choice]:
                observation = queues[choice].pop()
                break
            else:
                observation = None
        model.update(choice, observation)

        if not i % LOG_FREQ:
            index = i // LOG_FREQ - 1
            mpe[index] = model.mpe()

    # In case we're one short
    mpe[-1] = model.mpe()

    return mpe


def pretty_print(arr):
    for row in arr:
        out = ' '.join('%0.4f' % x for x in row.tolist())
        print(out)


# 01 loss
# Informative priors...avg predicted confidences by predicted class
def main(args: argparse.Namespace) -> None:
    # Set random seed to ensure reproducibility of experiments
    np.random.seed(args.seed)

    if not args.output.exists():
        args.output.mkdir()

    # Load the dataset and cost matrix
    if args.superclass:
        dataset = SuperclassDataset.load_from_text(args.input, superclass_lookup)
    else:
        dataset = Dataset.load_from_text(args.input)

    if args.cost_matrix is None:
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
        costs = np.load(args.cost_matrix)
    logging.info('Cost matrix:\n%s', costs)

    # Determine the highest cost predicted class
    expected_costs = (dataset.confusion_probs * costs).sum(axis=-1)
    highest_cost_class = np.argmax(expected_costs)
    cost_string = '\n'.join('%i  %0.4f' % x for x in enumerate(expected_costs))
    logging.info('Highest expected cost predicted class: %i', highest_cost_class)
    logging.info('Classwise expected costs:\n%s', cost_string)

    pretty_print(dataset.confusion_prior)
    print()
    pretty_print(dataset.confusion_probs)

    # Run experiments...
    random_results = np.zeros((N_SIMULATIONS, len(dataset) // LOG_FREQ, dataset.num_classes))
    active_results = np.zeros((N_SIMULATIONS, len(dataset) // LOG_FREQ, dataset.num_classes))
    active_informed_results = np.zeros((N_SIMULATIONS, len(dataset) // LOG_FREQ, dataset.num_classes))

    if args.superclass:
        pseudocount = 3
    else:
        pseudocount = dataset.num_classes

    for i in tqdm(range(N_SIMULATIONS)):
        alphas = np.ones((dataset.num_classes, pseudocount))
        model = DirichletMultinomialCost(alphas, costs)
        random_results[i] = select_and_label(dataset=dataset,
                                             model=model,
                                             choice_fn=random_choice_fn)
        model.mpe()

        alphas = np.ones((dataset.num_classes, pseudocount))
        model = DirichletMultinomialCost(alphas, costs)
        active_results[i] = select_and_label(dataset=dataset,
                                             model=model,
                                             choice_fn=max_choice_fn)

        model = DirichletMultinomialCost(pseudocount * dataset.confusion_prior, costs)
        active_informed_results[i] = select_and_label(dataset=dataset,
                                                      model=model,
                                                      choice_fn=max_choice_fn)

    random_success = (np.argmax(random_results, axis=-1) == highest_cost_class).mean(axis=0)
    active_success = (np.argmax(active_results, axis=-1) == highest_cost_class).mean(axis=0)
    active_informed_success = (np.argmax(active_informed_results, axis=-1) == highest_cost_class).mean(axis=0)

    fig, axes = plt.subplots(1,1)
    x_axis = np.arange(len(random_success)) * LOG_FREQ
    axes.plot(x_axis, random_success, label='random')
    axes.plot(x_axis, active_success, label='active (uniform prior)')
    axes.plot(x_axis, active_informed_success, label='active (informative prior)')
    axes.legend()
    plt.savefig(args.output / 'success_curve.png')

    fig, axes = plt.subplots(1, 1)
    n_samples=1000
    posterior_samples = model.sample(n_samples)
    max_expected_costs = posterior_samples.argmax(axis=1)
    hist = np.zeros((dataset.num_classes,))
    for i in range(dataset.num_classes):
        hist[i] = (max_expected_costs == i).sum() / n_samples
    axes.bar(np.arange(dataset.num_classes), hist)
    fig.savefig(args.output / 'posterior_costs.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path, help='input dataset')
    parser.add_argument('output', type=pathlib.Path, help='output prefix')
    parser.add_argument('-c', '--cost_matrix', type=pathlib.Path, default=None,
                        help='path to a serialized numpy array containng the cost matrix')
    parser.add_argument('-s', '--seed', type=int, default=1337, help='random seed')
    parser.add_argument('-k', type=float, default=2, help='relative cost')
    parser.add_argument('--superclass', action='store_true')
    parser.add_argument('--human', action='store_true')
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
