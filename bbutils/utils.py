from collections import deque
import numpy as np
import random
from copy import deepcopy
from scipy.stats import entropy
from math import *

def prepare_deques(categories, observations, num_classes):
    """
    INPUT:
        categories: a list of length num_samples; 
        observations: a list of length num_samples; each element takes value from True or False
    OUTPUT:
        returns a list of num_classes queues; partitioning the data points in observations with predicted label in categories.
    """
    deques = [deque() for _ in range(num_classes)]
    for category, observation in zip(categories, observations):
        deques[category].append(observation)
    for _deque in deques:
        random.shuffle(_deque)
    return deques

################### active overall accuracy estimation
### choices: np.array (n, 2), [category, observation] for each row
### theta_hat: np.array(num_classes, ), empirical estimation of acccuracy
### theta_prior: np.array(num_classes, ), prior belief of per category accuracy
### n_pool: np.array(num_classes, ), number of samples from each category of all pooled data
### n_opt: np.array(num_classes, ), optimal number of samples from each category
### n_current: np.array(num_classes, )

def get_theta_hat(choices, num_classes, theta_prior=None):
    if theta_prior  == None:
        theta_prior = np.ones(num_classes) * 0.5
    theta_hat = theta_prior
    for k in range(num_classes):
        idx = (choices[:,0] == k)
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
    
def get_overall_accuracy(choices, n_pool, theta_prior = None):
    p_pool = n_pool / n_pool.sum()
    theta_hat = get_theta_hat(choices, n_pool.shape[0], theta_prior)
    return (theta_hat * p_pool).sum()