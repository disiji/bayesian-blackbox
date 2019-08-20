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


class DirichletMultinomialCost:
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
        expected_costs = (self._costs * posterior_draw).sum(axis=-1)  # TODO: Double check...

        return expected_costs


class BetaBernoulli:
    def __init__(self, k: int):
        self._k = k
        self._params = np.ones((k, 2))

    def update(self, category: int, observation: bool):
        """Updates the posterior of the Beta-Bernoulli model."""
        if observation:
            self._params[category, 0] += 1
        else:
            self._params[category, 1] += 1

    def sample(self):
        """Draw sample thetas from the posterior."""
        theta = np.random.beta(self._params[:,0], self._params[:,1])
        return np.array(theta)