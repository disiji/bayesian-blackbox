import random
from collections import deque
from copy import deepcopy

import numpy as np
from scipy.stats import beta
from scipy.stats import entropy


# def prepare_deques(categories, observations, num_classes):
#     """
#     INPUT:
#         categories: a list of length num_samples;
#         observations: a list of length num_samples; each element takes value from True or False
#     OUTPUT:
#         returns a list of num_classes queues; partitioning the data points in observations with predicted label in categories.
#     """
#     deques = [deque() for _ in range(num_classes)]
#     for category, observation in zip(categories, observations):
#         deques[category].append(observation)
#     for _deque in deques:
#         random.shuffle(_deque)
#     return deques


################### active overall accuracy estimation
### choices: np.array (n, 2), [category, observation] for each row
### theta_hat: np.array(num_classes, ), empirical estimation of acccuracy
### theta_prior: np.array(num_classes, ), prior belief of per category accuracy
### n_pool: np.array(num_classes, ), number of samples from each category of all pooled data
### n_opt: np.array(num_classes, ), optimal number of samples from each category
### n_current: np.array(num_classes, )

def get_theta_hat(choices, num_classes, theta_prior=None):
    if theta_prior == None:
        theta_prior = np.ones(num_classes) * 0.5
    theta_hat = theta_prior
    for k in range(num_classes):
        idx = (choices[:, 0] == k)
        observations_k = choices[idx, 1] * 1.0
        if observations_k.shape[0] > 0:
            theta_hat[k] = sum(observations_k) / observations_k.shape[0]
    return theta_hat


def get_n_opt(n_pool, choices, sample_idx, num_classes, theta_prior=None):
    theta_hat = get_theta_hat(choices, num_classes, theta_prior)
    p_opt = np.sqrt(theta_hat * (1 - theta_hat)) * n_pool
    p_opt /= p_opt.sum()
    n_opt = p_opt * sample_idx
    return n_opt


def get_category_opt(n_pool, n_current, n_opt, mode="KL"):
    # make sure the returned cateogry has data points left to be sampled
    KL_divergence = np.zeros(n_pool.shape[0])
    if mode == "KL":
        for k in range(n_pool.shape[0]):
            n_next = deepcopy(n_current)
            n_next[k] += 1
            KL_divergence[k] = entropy(n_opt, n_next)
    for k in np.argsort(KL_divergence):
        if n_pool[k] > n_current[k]:
            return k


def get_overall_accuracy(choices, n_pool, theta_prior=None):
    p_pool = n_pool / n_pool.sum()
    theta_hat = get_theta_hat(choices, n_pool.shape[0], theta_prior)
    return (theta_hat * p_pool).sum()


##################### active learning BetaBernoulli

class GenerativeModel:
    def __init__(self, theta: np.ndarray):
        self._num_categories = theta.shape[0]
        self._theta = theta

    def sample(self, n: int):
        """Generates a dataset of n data points from the model"""
        categories = np.random.randint(self._num_categories, size=n)
        p_success = self._theta[categories]
        rng = np.random.rand(n)
        observations = rng < p_success
        return categories, observations


class BetaBernoulli:
    def __init__(self, k: int, prior=None):
        self._k = k
        if prior is None:
            self._params = np.ones((k, 2)) * 0.5
        else:
            self._params = np.copy(prior)

    def update(self, category: int, observation: bool):
        """Updates the posterior of the Beta-Bernoulli model."""
        if observation:
            self._params[category, 0] += 1
        else:
            self._params[category, 1] += 1

    @property
    def theta(self):
        return self._params[:, 0] / (self._params[:, 0] + self._params[:, 1])

    def sample(self):
        """Draw sample thetas from the posterior."""
        theta = np.random.beta(self._params[:, 0], self._params[:, 1])
        return np.array(theta)

    def get_params(self):
        return self._params

    def get_variance(self):
        return beta.var(self._params[:, 0], self._params[:, 1])

    def get_overall_acc(self, weight):
        return np.dot(beta.mean(self._params[:, 0], self._params[:, 1]), weight)
