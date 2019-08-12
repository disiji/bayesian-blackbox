"""
Bayesian blackbox assesment models.
"""
import numpy as np


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

    def sample(self) -> np.ndarray:
        """Draw sample expected costs from the posterior."""
        # Draw multinomial probabilities (e.g. the confusion probabilities) from posterior
        posterior_draw = np.zeros_like(self._alphas)
        for i, alpha in enumerate(self._alphas):
            posterior_draw[i] = np.random.dirichlet(alpha)

        # Compute expected costs of each predicted class
        expected_costs = (self._costs * posterior_draw).sum(axis=-1)
        return expected_costs

    def mpe(self, n_samples: int = 1000) -> np.ndarray:
        """Mean posterior estimate of expected costs, computed using Monte Carlo sampling"""
        num_classes = self._alphas.shape[0]
        mpe = np.zeros((num_classes,))
        for i, alpha in enumerate(self._alphas):
            samples = np.random.dirichlet(alpha, size=(n_samples,))
            costs = self._costs[i].reshape(1, -1)
            expected_cost = (samples * self._costs[i]).sum(-1).mean()
            mpe[i] = expected_cost
        return mpe

