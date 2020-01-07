"""
Bayesian blackbox assesment models.
"""
import math

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
        if prior is None:
            self._params = np.ones((k, 2)) * 0.5
        else:
            self._params = prior

    @property
    def theta(self):
        return self._params[:, 0] / (self._params[:, 0] + self._params[:, 1])

    def update(self, category: int, observation: bool):
        """Updates the posterior of the Beta-Bernoulli model."""
        if observation:
            self._params[category, 0] += 1
        else:
            self._params[category, 1] += 1

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
        """Mean posterior estimate of expected costs, computed using Monte Carlo sampling"""
        z = self._alphas.sum(axis=-1, keepdims=True)
        expected_probs = self._alphas / z
        expected_costs = (self._costs * expected_probs).sum(axis=-1)
        return expected_costs


class SumOfBetaEce(Model):
    """Model ECE as weighted sum of absolute shifted Beta distributions, with each Beta distribution capturing the accuracy per bin.

    Parameters
    ==========
    weight: np.ndarray (k, ), weight of each bin.
    prior_alpha: np.ndarray (k, ), alpha parameter of the Beta distribution for each bin
    prior_beta: np.ndarray (k, ), beta parameter of the Beta distribution for each bin
    """

    def __init__(self, k: int, weight: None, prior_alpha: None, prior_beta: None):
        # constants
        self._k = k
        self._weight = weight
        self._diagonal = np.array([(i + 0.5) / k for i in range(0, k)])

        # parameters to update:
        self._alpha = None  #
        self._beta = None
        self._counts = np.zeros((k,))

        # initialize the mode of each Beta distribution on diagonal
        peusdo_count = 10
        if prior_alpha is None:
            self._alpha = np.array([(i + 0.5) * (peusdo_count - 2) / k + 1 for i in range(self._k)])
        else:
            self._alpha = np.copy(prior_alpha)

        if prior_beta is None:
            self._beta = np.array(
                [(self._alpha[i] - 1) * self._k / (i + 0.5) - (self._alpha[i] - 2) for i in range(self._k)])
        else:
            self._beta = np.copy(prior_beta)

    @property
    def ece(self):
        theta = self._alpha / (self._alpha + self._beta)
        if self._weight:  # pool weights
            weight = self._weight
        else:  # online weights
            weight = self._counts / sum(self._counts)

        return weight * np.abs(theta - self._diagonal)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Draw samples from the posterior distribution of ECE.

        Parameters
        ==========
        :param n_samples:
        :return:
        """

        # draw samples from each Beta distribution
        theta = np.random.beta(self._alpha, self._beta, size=(n_samples, self._k))  # theta: (n_samples, k)
        # compute ECE with samples

        if self._weight:  # pool weights
            weight = self._weight
        else:  # online weights
            weight = self._counts / sum(self._counts)

        return weight * np.abs(theta - self._diagonal)

    def update(self, predicted_class: int, true_class: int):
        """ update self._alpha, self._beta, self._counts

        Parameters
        ==========
        :param predicted_class:
        :param true_class:
        :return:
        """
        if true_class == predicted_class:
            self._alpha[predicted_class] += 1
        else:
            self._beta[predicted_class] += 1
        self._counts[predicted_class] += 1


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
