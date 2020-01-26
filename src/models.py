"""
Bayesian blackbox assesment models.
"""
import copy
import math
from typing import List

import numpy as np
from scipy.stats import beta


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


class Model:
    """
    Abstract base class to be inhereted by all models.

    Derived classes must implement an update and sample method.
    """

    def update(self, predicted_class: int, true_class: int) -> None:
        """
        Update the model given a new observation.

        Parameters
        ==========
        predicted_class : int
            The class predicted by the blackbox classifier.
        true_class : int
            The true class revealed by the oracle.
        """
        raise NotImplementedError

    def sample(self) -> np.ndarray:
        """
        Sample a parameter vector from the model posterior.

        Returns
        =======
        params : np.ndarray
            The sampled parameter vector.
        """
        raise NotImplementedError


class BetaBernoulli(Model):
    def __init__(self, k: int, prior=None):
        self._k = k
        self._prior = prior
        if prior is None:
            self._prior = np.ones((k, 2)) * 0.5

        self._params = copy.deepcopy(self._prior)

    @property
    def eval(self):
        return self._params[:, 0] / (self._params[:, 0] + self._params[:, 1])

    @property
    def variance(self):
        return beta.var(self._params[:, 0], self._params[:, 1])

    @property
    def frequentist_eval(self):
        counts = self._params - self._prior + 0.0001
        return counts[:, 0] / (counts[:, 0] + counts[:, 1])

    def update(self, category: int, observation: bool):
        """Updates the posterior of the Beta-Bernoulli model."""
        if observation:
            self._params[category, 0] += 1
        else:
            self._params[category, 1] += 1

    def update_batch(self, categories: List[int], observations: List[bool]):
        """Updates the posterior of the Beta-Bernoulli model for a batch of observations."""
        for category, observation in zip(categories, observations):
            if observation:
                self._params[category, 0] += 1
            else:
                self._params[category, 1] += 1

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """Draw sample thetas from the posterior.

        Parameters
        ==========
        num_samples : int
            Number of times to sample from posterior. Default: 1.

        Returns
        =======
        An (k, num_samples) array of samples of theta. If num_samples == 1 then last dimension is squeezed.
        """
        theta = np.random.beta(self._params[:, 0], self._params[:, 1], size=(num_samples, self._k))
        return np.array(theta).T.squeeze()

    def get_params(self):
        return self._params

    def get_overall_acc(self, weight):
        return np.dot(beta.mean(self._params[:, 0], self._params[:, 1]), weight)


class SumOfBetaEce(Model):
    """Model ECE as weighted sum of absolute shifted Beta distributions, with each Beta distribution capturing the
    accuracy per bin.

    Parameters
    ==========
    num_bins: number of bins for calibration
    weight: np.ndarray (num_bins, ), weight of each bin.
    prior_alpha: np.ndarray (num_bins, ), alpha parameter of the Beta distribution for each bin
    prior_beta: np.ndarray (num_bins, ), beta parameter of the Beta distribution for each bin
    """

    def __init__(self, num_bins: int, weight: np.ndarray = None, pseudocount: int = 3, prior_alpha: np.ndarray = None,
                 prior_beta: np.ndarray = None):
        """
        Init model parameters self._alpha and self._beta, either with pseudocount (put mean of beta on diagonal
        with prior strength pseudocount) or with given prior_alpha and prior_beta.

        :param num_bins:
        :param weight:
        :param prior_alpha:
        :param prior_beta:
        """
        # constants
        self._num_bins = num_bins
        self._weight = weight
        self._diagonal = np.array([(i + 0.5) / num_bins for i in range(0, num_bins)])

        # parameters to update:
        self._counts = np.ones((num_bins, 2)) * 0.0001
        self._confidence = np.array([(i + 0.5) / num_bins for i in range(0, num_bins)])

        # initialize the mode of each Beta distribution on diagonal
        if prior_alpha is None:
            # self._alpha = np.array([(i + 0.5) * (pseudocount - 2) / num_bins + 1 for i in range(self._num_bins)])
            self._alpha = (np.arange(num_bins) + 0.5) * pseudocount / num_bins
        else:
            self._alpha = np.copy(prior_alpha)

        if prior_beta is None:
            # self._beta = np.array(
            #     [(self._alpha[i] - 1) * self._num_bins / (i + 0.5) - (self._alpha[i] - 2) for i in
            #      range(self._num_bins)])
            self._beta = pseudocount - self._alpha
        else:
            self._beta = np.copy(prior_beta)

    @property
    def eval(self) -> float:
        """
        Eval MPE of ECE by taking the weighted absolute difference between MPE of bin-wise theta and confidence.
        :return: float
        """
        theta = self._alpha / (self._alpha + self._beta)
        if self._weight is not None:  # pool weights
            weight = self._weight
        else:  # online weights
            tmp = np.sum(self._counts, axis=1)
            weight = tmp / sum(tmp)
        return np.dot(np.abs(theta - self._confidence), weight)

    @property
    def variance(self) -> float:
        """
        Eval variance of posterior of ECE.
        :return:
        """
        variance_bin = beta.var(self._alpha, self._beta)
        if self._weight is not None:  # pool weights
            weight = self._weight
        else:  # online weights
            tmp = np.sum(self._counts, axis=1)
            weight = tmp / sum(tmp)
        return np.dot(weight * weight, variance_bin)

    @property
    def frequentist_eval(self) -> None:
        """
        Eval ECE in a frequentist's way

        :return:
        """
        tmp = np.sum(self._counts, axis=1)
        accuracy = self._counts[:, 0] / tmp
        weight = tmp / sum(tmp)
        return np.dot(np.abs(accuracy - self._confidence), weight)

    def get_params(self):
        return self._alpha, self._beta

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """Draw sample ECEs from posterior.

        :param num_samples : int
            Number of times to sample from posterior. Default: 1.

        :return: An (num_samples, ) array of ECE. If n_samples == 1 then last dimension is squeezed.
        """

        # draw samples from each Beta distribution
        theta = np.random.beta(self._alpha, self._beta,
                               size=(num_samples, self._num_bins))
        # compute ECE with samples
        if self._weight is not None:  # pool weights
            weight = self._weight
        else:  # online weights
            tmp = np.sum(self._counts, axis=1)
            weight = tmp / sum(tmp)
        return np.dot(np.abs(theta - self._confidence), weight)

    def update(self, score: float, observation: bool) -> None:
        """
        Update the model parameters with one labeled sample (score, observation).

        :param score: float, confidence of the prediction
        :param observation: bool, whether predicted label is the same as true label
        """
        bin_idx = math.floor(score * self._num_bins)
        if score == 1:
            bin_idx -= 1
        if observation:
            self._alpha[bin_idx] += 1
            self._counts[bin_idx][0] += 1
        else:
            self._beta[bin_idx] += 1
            self._counts[bin_idx][1] += 1
        self._confidence[bin_idx] = (self._confidence[bin_idx] * (self._counts[bin_idx].sum() - 1) + score) / (
            self._counts[bin_idx].sum())

    def update_batch(self, scores: List[float], observations: List[bool]):
        """
        Update the model parameters with a batch of labeled sample.
        :param scores:
        :param observations:
        :return:
        """
        for score, observation in zip(scores, observations):
            self.update(score, observation)


class ClasswiseEce(Model):
    """

    """

    def __init__(self, k: int, num_bins: int, pseudocount: float, weight: None, prior=None):
        """
        Parameters
        ==========
        k: number of classes
        num_bins: number of bins for evaluating ECE
        pseudocount:
        weight: a list of (num_bins, ) arrays of length k
        prior: an (number of classes, k, 2) array for alpha and beta parameters in the prior Beta distributions.


        Returns
        =======
        """
        self._k = k

        if weight is None:
            weight = [None] * self._k

        if prior is None:
            self._classwise_ece_models = [
                copy.deepcopy(
                    SumOfBetaEce(num_bins, weight=weight[class_idx], pseudocount=pseudocount, prior_alpha=None,
                                 prior_beta=None))
                for class_idx in range(k)]
        else:
            self._classwise_ece_models = [
                copy.deepcopy(
                    SumOfBetaEce(num_bins, weight=weight[class_idx], prior_alpha=prior[class_idx, :, 0].squeeze(),
                                 prior_beta=prior[class_idx, :, 1].squeeze()))
                for class_idx in range(k)]

    def update(self, category: int, observation: bool, score: float):
        """

        :param category:
        :param observation:
        :param score:
        :return:
        """
        self._classwise_ece_models[category].update(score, observation)

    def update_batch(self, categories: List[int], observations: List[bool], scores: List[float]):
        """

        :param categories:
        :param observations:
        :param scores:
        :return:
        """
        for (category, observation, score) in zip(categories, observations, scores):
            self.update(category, observation, score)

    @property
    def eval(self) -> np.ndarray:
        """
        Evaluate ECE for each class

        Returns
        =======
        An (k,) array of ECE evaluate for each class.
        """
        classwise_ece = np.array([self._classwise_ece_models[class_idx].eval for class_idx in range(self._k)])
        return classwise_ece

    @property
    def frequentist_eval(self) -> np.ndarray:
        """
        Evaluate ECE for each class

        Returns
        =======
        An (k,) array of ECE evaluate for each class.
        """
        classwise_ece = np.array(
            [self._classwise_ece_models[class_idx].frequentist_eval for class_idx in range(self._k)])
        return classwise_ece

    @property
    def variance(self):
        """
        Variance of posterior ECE for each class

        Returns
        =======
        An (k,) array of variance evaluate for each class.
        """
        classwise_ece_variance = np.array(
            [self._classwise_ece_models[class_idx].variance for class_idx in range(self._k)])
        return classwise_ece_variance

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """Draw sample eces from the posterior.

        Parameters
        ==========
        num_samples : int
            Number of times to sample from posterior. Default: 1.

        Returns
        =======
        An (k, num_samples) array of samples of theta. If num_samples == 1 then last dimension is squeezed.
        """
        samples = np.array(
            [self._classwise_ece_models[class_idx].sample(num_samples) for class_idx in range(self._k)]).squeeze()
        return samples


class DirichletMultinomialCost(Model):
    """
    Multinomial w/ Dirichlet prior for predicted class cost estimation.

    WARNING: Arrays passed to constructor are copied!

    Parameters
    ==========
    alphas : np.ndarray
        An array of shape (n_classes, n_classes) where each row parameterizes a single Dirichlet distribution.
    costs : np.ndarray
        An array of shape (n_classes, n_classes). The cost matrix.
    """

    def __init__(self, alphas: np.ndarray, costs: np.ndarray) -> None:
        assert alphas.shape == costs.shape
        self._alphas = np.copy(alphas)
        self._costs = np.copy(costs)

    def update(self, predicted_class: int, true_class: int) -> None:
        """Update the posterior of the model."""
        self._alphas[predicted_class, true_class] += 1

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Draw sample expected costs from the posterior.

        Parameters
        ==========
        n_samples : int
            Number of times to sample from posterior. Default: 1.

        Returns
        =======
        An (n, n_samples) array of expected costs. If n_samples == 1 then last dimension is squeezed.
        """
        # Draw multinomial probabilities (e.g. the confusion probabilities) from posterior
        if n_samples == 1:
            posterior_draw = np.zeros_like(self._alphas)
            for i, alpha in enumerate(self._alphas):
                posterior_draw[i] = np.random.dirichlet(alpha)
        else:
            posterior_draw = np.zeros((n_samples, *self._alphas.shape))
            for i, alpha in enumerate(self._alphas):
                posterior_draw[:, i, :] = np.random.dirichlet(alpha, size=(n_samples,))

        # Compute expected costs of each predicted class
        expected_costs = (np.expand_dims(self._costs, 0) * posterior_draw).sum(axis=-1)
        return expected_costs.squeeze()

    def mpe(self) -> np.ndarray:
        """Mean posterior estimate of expected costs"""
        z = self._alphas.sum(axis=-1, keepdims=True)
        expected_probs = self._alphas / z
        expected_costs = (self._costs * expected_probs).sum(axis=-1)
        return expected_costs

    def confusion_matrix(self) -> np.ndarray:
        z = self._alphas.sum(axis=-1, keepdims=True)
        return self._alphas / z


class SpikeAndBetaSlab(Model):
    """
    Spike(on diagonal) and slab (beta distribution) for modeling calibration of the model.

    WARNING: Arrays passed to constructor are copied!

    Parameters
    ==========
    k: int, number of bins
    prior_mu: np.ndarray (k, ), weight on the spike component for each bin
    prior_alpha: np.ndarray (k, ), alpha parameter of the Beta distribution for each bin
    prior_beta: np.ndarray (k, ), beta parameter of the Beta distribution for each bin
    """

    def __init__(self, k: int, prior_mu: None, prior_alpha: None, prior_beta: None):

        # constants
        self._k = k
        self._theta_0 = np.array([(i + 0.5) / k for i in range(k)])  # spike on diagonal

        # parameters to update:
        self._mu = None  # weight on the spike component, reflects how calibrated the model is
        self._alpha = None  #
        self._beta = None

        if prior_mu is None:
            self._mu = np.ones((k,)) * 0.5
        else:
            self._mu = np.copy(prior_mu)

        if prior_alpha is None:
            self._alpha = np.ones((k,)) * 0.5
        else:
            self._alpha = np.copy(prior_alpha)

        if prior_beta is None:
            self._beta = np.ones((k,)) * 0.5
        else:
            self._beta = np.copy(prior_beta)

    def update(self, category: int, observation: bool):
        """Updates the posterior: of the SpikeAndBetaSlab-Bernoulli model with one observation."""
        if observation:
            self._mu[category] = (self._mu[category] * self._theta_0[category]) / (
                    self._mu[category] * self._theta_0[category] + 1 - self._mu[category])
            self._alpha[category] += 1
        else:
            self._mu[category] = (self._mu[category] * (1 - self._theta_0[category])) / (
                    self._mu[category] * (1 - self._theta_0[category]) + 1 - self._mu[category])
            self._beta[category] += 1

    def _binom(n: int, k: int):
        return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

    def update_batch(self, category: int, n: int, k: int):
        """Updates the posterior of the SpikeAndBetaSlab-Binomial model with k out of n positive observations in category"""
        self._mu[category] = 1 - (1 - self._mu[category]) \
                             / (self._mu[category] * _binom(n, k) * self._theta_0[category] ** k * (
                1 - self._theta_0[category]) ** (
                                        n - k) + 1 - self._mu[category])
        self._alpha[category] += k
        self._beta[category] += n - k

    @property
    def theta(self):
        """Return the mean of each bin"""
        return self._mu * self._theta_0 + (1 - self._mu) * self._alpha / (self._alpha + self._beta)
