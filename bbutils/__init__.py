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


class BetaBernoulli:
    def __init__(self, k: int, prior=None):
        self._k = k
        if prior.shape[0] == 0:
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
        theta = np.random.beta(self._params[:,0], self._params[:,1])
        return np.array(theta)
    
    def get_params(self):
        return self._params
    
    def get_variance(self):
        return beta.var(self._params[:,0], self._params[:,1])
    
    def get_overall_acc(self, weight, theta_prior):
        return np.dot(beta.mean(self._params[:,0]-theta_prior[:,0] + 0.0002, self._params[:,1]-theta_prior[:,1] + 0.0008), weight)
        
