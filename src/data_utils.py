import random
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

np.random.seed(0)

datafile_dict = {
    'cifar100': '../data/cifar100/cifar100_predictions_dropout.txt',
    'imagenet': '../data/imagenet/resnet152_imagenet_outputs.txt',
    'imagenet2_topimages': '../data/imagenet/resnet152_imagenetv2_topimages_outputs.txt',
    '20newsgroup': '../data/20newsgroup/bert_20_newsgroups_outputs.txt',
    'svhn': '../data/svhn/svhn_predictions.txt',
    'dbpedia': '../data/dbpedia/bert_dbpedia_outputs.txt',
}


logits_dict = {
    'cifar100': '../data/cifar100/resnet110_cifar100_logits.txt',
    'imagenet': '../data/imagenet/resnet152_imagenet_logits.txt',
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
    'human': '../output/cost_result_matrices/cifar100_people_full/costs.npy',
    'superclass': '../output/cost_result_matrices/cifar100_superclass_full/costs.npy'
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
