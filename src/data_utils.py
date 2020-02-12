import logging
import warnings
from typing import List

import numpy as np
import pandas as pd

from models import BetaBernoulli, ClasswiseEce

logger = logging.getLogger(__name__)

np.random.seed(0)

DATA_DIR = '/Users/disiji/Dropbox/current/bayesian-blackbox/data/'

datafile_dict = {
    'cifar100': DATA_DIR + 'cifar100/cifar100_predictions_dropout.txt',
    'imagenet': DATA_DIR + 'imagenet/resnet152_imagenet_outputs.txt',
    'imagenet2_topimages': DATA_DIR + 'imagenet/resnet152_imagenetv2_topimages_outputs.txt',
    '20newsgroup': DATA_DIR + '20newsgroup/bert_20_newsgroups_outputs.txt',
    'svhn': DATA_DIR + 'svhn/svhn_predictions.txt',
    'dbpedia': DATA_DIR + 'dbpedia/bert_dbpedia_outputs.txt',
}

logits_dict = {
    'cifar100': DATA_DIR + 'cifar100/resnet110_cifar100_logits.txt',
    'imagenet': DATA_DIR + 'imagenet/resnet152_imagenet_logits.txt',
}

datasize_dict = {
    'cifar100': 10000,
    'imagenet': 50000,
    'imagenet2_topimages': 10000,
    '20newsgroup': 7532,
    'svhn': 26032,
    'dbpedia': 70000,
}

num_classes_dict = {
    'cifar100': 100,
    'imagenet': 1000,
    'imagenet2_topimages': 1000,
    '20newsgroup': 20,
    'svhn': 10,
    'dbpedia': 14,
}

output_str_dict = {
    'weighted_pool_bayesian_estimation_error': 'weighted_pool_error_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'weighted_pool_frequentist_estimation_error': 'weighted_pool_error_%s_PseudoCount%.1f_runs%d_frequentist.csv',
    'weighted_online_bayesian_estimation_error': 'weighted_online_error_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'weighted_online_frequentist_estimation_error': 'weighted_online_error_%s_PseudoCount%.1f_runs%d_frequentist.csv',
    'unweighted_bayesian_estimation_error': 'unweighted_error_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'unweighted_frequentist_estimation_error': 'unweighted_error_%s_PseudoCount%.1f_runs%d_frequentist.csv',
    'pool_bayesian_ece': 'pool_ece_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'pool_frequentist_ece': 'pool_ece_%s_PseudoCount%.1f_runs%d_frequentist.csv',
    'online_bayesian_ece': 'online_ece_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'online_frequentist_ece': 'online_ece_%s_PseudoCount%.1f_runs%d_frequentist.csv',
    'bayesian_mce': 'mce_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'frequentist_mce': 'mce_%s_PseudoCount%.1f_runs%d_frequentist.csv'
}
cost_matrix_dir_dict = {
    'human': DATA_DIR + 'cost/cifar100_people_full/costs.npy',
    'superclass': DATA_DIR + 'cost/cifar100_superclass_full/costs.npy'
}
DATASET_LIST = ['imagenet', 'dbpedia', 'cifar100', '20newsgroup', 'svhn', 'imagenet2_topimages']


# this function is different from bayesian_reliabiitly.prepare_data

def prepare_data(filename, four_column=False):
    """

    :param filename: str
    :param four_column: indicates whether the dataformat is "index, correct class, predicted class, confidence"
                        or true label followed by a vector of scores for each class
    :return:
            categories: List[int], predicted class
            observations: List[bool], whether predicted class is the same as truth class
            confidence: List[float]
            idx2category: Dict[int, str] or None
            category2idx: Dict[str, int] or None

    """
    if four_column:
        # when file is in 4 column format: index, correct class, predicted class, confidence
        with open(filename, 'r') as f:
            category2idx = dict()
            idx2category = []
            categories = []
            observations = []
            confidences = []
            labels = []
            next(f)
            for line in f:
                _, correct, predicted, confidence = line.split()
                if predicted not in category2idx:
                    category2idx[predicted] = len(category2idx)
                    idx2category.append(predicted)
                idx = category2idx[predicted]
                categories.append(idx)
                observations.append(correct == predicted)
                confidences.append(float(confidence))
                labels.append(correct)

    else:
        data = np.genfromtxt(filename)
        categories = np.argmax(data[:, 1:], axis=1).astype(int)
        confidences = list(np.max(data[:, 1:], axis=1).astype(float))
        observations = list((categories == data[:, 0]))
        categories = list(categories)
        labels = list(data[:, 0])
        idx2category = None
        category2idx = None
        logger.debug("Dataset Accuracy: %.3f" % (len([_ for _ in observations if _ == True]) * 1.0 / len(observations)))

    return categories, observations, confidences, idx2category, category2idx, labels


def train_holdout_split(categories: List[int],
                        observations: List[bool],
                        confidences: List[float],
                        labels: List[int],
                        indices: List[int],
                        holdout_ratio: float = 0.2):
    """
    Split categories, observations and confidences into train and holdout with hold_ratio.
    """
    num_samples = len(categories)

    permutation = np.random.permutation(num_samples)
    mask = np.zeros(num_samples)
    mask[permutation[:int(len(categories) * holdout_ratio)]] = 1

    train_categories = [categories[idx] for idx in range(num_samples) if mask[idx] == 0]
    train_observations = [observations[idx] for idx in range(num_samples) if mask[idx] == 0]
    train_confidences = [confidences[idx] for idx in range(num_samples) if mask[idx] == 0]
    train_labels = [labels[idx] for idx in range(num_samples) if mask[idx] == 0]
    train_indices = [indices[idx] for idx in range(num_samples) if mask[idx] == 0]

    holdout_categories = [categories[idx] for idx in range(num_samples) if mask[idx] == 1]
    holdout_observations = [observations[idx] for idx in range(num_samples) if mask[idx] == 1]
    holdout_confidences = [confidences[idx] for idx in range(num_samples) if mask[idx] == 1]
    holdout_labels = [labels[idx] for idx in range(num_samples) if mask[idx] == 1]
    holdout_indices = [indices[idx] for idx in range(num_samples) if mask[idx] == 1]

    return train_categories, train_observations, train_confidences, train_labels, train_indices, holdout_categories, holdout_observations, holdout_confidences, holdout_labels, holdout_indices


def eval_ece(confidences: List[float], observations: List[bool], num_bins=10):
    """

    :param confidences:
    :param observations:
    :param num_bins:
    :return:
    """
    confidences = np.array(confidences)
    observations = np.array(observations) * 1.0
    bins = np.linspace(0, 1, num_bins + 1)
    digitized = np.digitize(confidences, bins[1:-1])

    w = np.array([(digitized == i).sum() for i in range(num_bins)])
    w = w / sum(w)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        confidence_bins = np.array([confidences[digitized == i].mean() for i in range(num_bins)])
        accuracy_bins = np.array([observations[digitized == i].mean() for i in range(num_bins)])
    confidence_bins[np.isnan(confidence_bins)] = 0
    accuracy_bins[np.isnan(accuracy_bins)] = 0
    diff = np.absolute(confidence_bins - accuracy_bins)
    ece = np.inner(diff, w)
    return ece


def get_confidence_k(categories: List[int], confidences: List[float], num_classes: int) -> np.ndarray:
    """

    :param categories:
    :param confidences:
    :param num_classes:
    :return: confidence_k: (num_classes, )
    """
    df = pd.DataFrame(list(zip(categories, confidences)), columns=['Predicted', 'Confidence'])
    confidence_k = np.array([df[(df['Predicted'] == id)]['Confidence'].mean()
                             for id in range(num_classes)])
    return confidence_k


def get_accuracy_k(categories: List[int], observations: List[bool], num_classes: int) -> np.ndarray:
    observations = np.array(observations) * 1.0
    df = pd.DataFrame(list(zip(categories, observations)), columns=['Predicted', 'Observations'])
    accuracy_k = np.array([df[(df['Predicted'] == class_idx)]['Observations'].mean()
                           for class_idx in range(num_classes)])
    return accuracy_k


def get_ece_k(categories: List[int], observations: List[bool], confidences: List[float], num_classes: int,
              num_bins=10) -> np.ndarray:
    """

    :param categories:
    :param observations:
    :param confidences:
    :param num_classes:
    :param num_bins:
    :return:
    """
    ece_k = np.zeros((num_classes,))

    for class_idx in range(num_classes):
        mask_idx = [i for i in range(len(observations)) if categories[i] == class_idx]
        observations_sublist = [observations[i] for i in mask_idx]
        confidences_sublist = [confidences[i] for i in mask_idx]
        ece_k[class_idx] = eval_ece(confidences_sublist, observations_sublist, num_bins)

    return ece_k


def get_ground_truth(categories: List[int], observations: List[bool], confidences: List[float], num_classes: int,
                     metric: str, mode: str, topk: int = 1) -> np.ndarray:
    """
    Compute ground truth given metric and mode with all data points.
    :param categories:
    :param observations:
    :param confidences:
    :param metric:
    :param mode:
    :return: binary np.ndarray of shape (num_classes, ) indicating each class in top k or not.
    """
    if metric == 'accuracy':
        metric_val = get_accuracy_k(categories, observations, num_classes)
    elif metric == 'calibration_error':
        metric_val = get_ece_k(categories, observations, confidences, num_classes, num_bins=10)

    output = np.zeros((num_classes,), dtype=np.bool_)

    if mode == 'max':
        indices = metric_val.argsort()[-topk:]
    else:
        indices = metric_val.argsort()[:topk]

    output[indices] = 1

    return output


def get_bayesian_ground_truth(categories: List[int], observations: List[bool], confidences: List[float],
                              num_classes: int,
                              metric: str, mode: str, topk: int = 1, pseudocount: int = 1, prior=None) -> np.ndarray:
    """
    Compute ground truth given metric and mode with all data points.
    :param categories:
    :param observations:
    :param confidences:
    :param metric:
    :param mode:
    :return: binary np.ndarray of shape (num_classes, ) indicating each class in top k or not.
    """

    if metric == 'accuracy':
        model = BetaBernoulli(num_classes, prior=prior)
        model.update_batch(scores, observations)
    elif metric == 'calibration_error':
        model = ClasswiseEce(num_classes, num_bins=10, pseudocount=pseudocount)
        model.update_batch(categories, observations, confidences)

    metric_val = model.eval

    output = np.zeros((num_classes,), dtype=np.bool_)

    if mode == 'max':
        indices = metric_val.argsort()[-topk:]
    else:
        indices = metric_val.argsort()[:topk]

    output[indices] = 1

    return output




############################################################################
#CIFAR100 meta data needed to map classes to superclasses and vice versa.
############################################################################

cifar100_classes = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle",
    "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle",
    "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
    "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road",
    "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm"
]

cifar100_superclasses = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor_things", "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores", "medium_mammals", "non-insect_invertebrates", "people",
    "reptiles", "small_mammals", "trees", "vehicles_1", "vehicles_2"
]

cifar100_reverse_superclass_lookup = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
}

cifar100_superclass_lookup = {class_: superclass for superclass, class_list in
                              cifar100_reverse_superclass_lookup.items() for class_ in
                              class_list}
