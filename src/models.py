"""
Bayesian blackbox assesment models.
"""
import copy
import math
from typing import List, Tuple

import numpy as np
from scipy.stats import beta


class Model:
    """
    Abstract base class to be inhereted by all models.
    Derived classes must implement an update and sample method.
    """

    def update(self, predicted_class: int, true_class: int) -> None:
        """
        Update the model given a new observation.
        :param predicted_class: int
            The class predicted by the blackbox classifier.
        :param true_class: int
            The true class revealed by the oracle.
        """
        raise NotImplementedError

    def sample(self) -> np.ndarray:
        """
        Sample a parameter vector from the model posterior.
        :return: np.ndarray
            The sampled parameter vector.
        """
        raise NotImplementedError


class BetaBernoulli(Model):
    """
    Model classwise accuracy with a Beta Bernoulli distribution for each predicted class.
    """

    def __init__(self, k: int, prior=None):
        """
        :param k: int
            The number of classes.
        :param prior: np.ndarray (k, 2) or None
            alpha and beta parameters of prior Beta distributions. Default: None.
        """
        self._k = k
        self._prior = prior
        if prior is None:
            self._prior = np.ones((k, 2)) * 0.5

        self._params = copy.deepcopy(self._prior)

    @property
    def eval(self) -> np.ndarray:
        """
        MPE of posterior classwise accuracy.
        :return: An (k, ) array of MPE of posteriors of classwise accuracies.
        """
        return self._params[:, 0] / (self._params[:, 0] + self._params[:, 1])

    @property
    def frequentist_eval(self) -> np.ndarray:
        """
        MLE of classwise accuracy.
        :return: An (k, ) array of MLE of classwise accuracies.
        """
        counts = self._params - self._prior + 0.0001
        return counts[:, 0] / (counts[:, 0] + counts[:, 1])

    @property
    def variance(self) -> np.ndarray:
        """
        Variance of posterior classwise accuracy.
        :return: An (k, ) array of variance of posteriors of classwise accuracies.
        """
        return beta.var(self._params[:, 0], self._params[:, 1])

    def get_params(self) -> np.ndarray:
        """
        Returns alpha and beta parameters of the Beta posterior distribution of classwise accuracies.
        :return: An (k, 2) array of alpha and beta parameters of posterior Beta distributions.
        """
        return self._params

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Draw sample thetas from the posterior.
        :param num_samples: int
            Number of times to sample from posterior. Default: 1.
        :return: An (k, num_samples) array of samples of theta. If num_samples == 1 then last dimension is squeezed.
        """
        theta = np.random.beta(self._params[:, 0], self._params[:, 1], size=(num_samples, self._k))
        return np.array(theta).T.squeeze()

    def update(self, category: int, observation: bool) -> None:
        """
        Updates the posterior of the Beta-Bernoulli model.
        :param category: int
            The index of the predicted class.
        :param observation: bool
            Indicator for whether the predicted class agrees with the true class label.
        """
        if observation:
            self._params[category, 0] += 1
        else:
            self._params[category, 1] += 1

    def update_batch(self, categories: List[int], observations: List[bool]) -> None:
        """
        Updates the posterior of the Beta-Bernoulli model for a batch of observations.
        :param categories: List[int]
            A list of predicted classes of samples.
        :param observations: List[bool]
            A list of boolean observations, each observation represents whether the predicted class agrees with the
                true class label.
        """
        for category, observation in zip(categories, observations):
            if observation:
                self._params[category, 0] += 1
            else:
                self._params[category, 1] += 1


class SumOfBetaEce(Model):
    """Model ECE as weighted sum of absolute shifted Beta distributions, with each Beta distribution capturing the
    accuracy per bin.
    """

    def __init__(self, num_bins: int, weight: np.ndarray = None, pseudocount: int = 3, prior_alpha: np.ndarray = None,
                 prior_beta: np.ndarray = None):
        """
        Init model parameters self._alpha and self._beta, either with pseudocount (put mean of beta on diagonal
        with prior strength pseudocount) or with given prior_alpha and prior_beta.

        :param num_bins: number of bins for calibration
        :param weight: np.ndarray (num_bins, ), weight of each bin.
        :param prior_alpha: np.ndarray (num_bins, ), alpha parameter of the Beta distribution for each bin
        :param prior_beta: np.ndarray (num_bins, ), beta parameter of the Beta distribution for each bin
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
            self._alpha = (np.arange(num_bins) + 0.5) * pseudocount / num_bins
        else:
            self._alpha = np.copy(prior_alpha)

        if prior_beta is None:
            self._beta = pseudocount - self._alpha
        else:
            self._beta = np.copy(prior_beta)

    @property
    def beta_params_mpe(self) -> np.ndarray:
        """
        MPE of accuracy per bin.
        :return: (self._num_bins, )
        """
        return self._alpha / (self._alpha + self._beta)

    @property
    def counts_per_bin(self) -> np.ndarray:
        """
        Returns the number of samples that the model has seen so far in each bin.
        :return: An (num_bins, ) array of counts of samples.
        """
        return np.sum(self._counts, axis=1)

    # todo: @disiji update this function with sampling: estiamte MPE of  bin-wise absolute difference via sampling.
    @property
    def eval(self) -> float:
        """
        Eval MPE of ECE by taking the weighted absolute difference between MPE of bin-wise theta and confidence.
        :return: float
            MPE of ECE posterior.
        """
        theta = self._alpha / (self._alpha + self._beta)
        if self._weight is not None:  # pool weights
            weight = self._weight
        else:  # online weights
            tmp = np.sum(self._counts, axis=1)
            weight = tmp / sum(tmp)
        return np.dot(np.abs(theta - self._confidence), weight)

    @property
    def frequentist_eval(self) -> float:
        """
        Eval ECE in a frequentist's way: compute accuracy per bin with the frequentist's way and compute its weighted
            difference to bin-wise confidence score.
        :return: float
        """
        tmp = np.sum(self._counts, axis=1)
        accuracy = self._counts[:, 0] / tmp
        weight = tmp / sum(tmp)
        return np.dot(np.abs(accuracy - self._confidence), weight)

    @property
    def variance(self) -> float:
        """
        Variance of posterior ECE estimated with sampling. Variance of a model is used in Bayesian active learning
            methods like Bayesian UCB.
        :return: float
            Variance of posterior ECE estimated with Monte Carlo samples.
        """
        num_samples = 100
        samples = self.sample(num_samples)
        return np.var(samples)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get alpha and beta parameters of Beta distributions for bin-wise accuracy.
        :return: Tuple(np.ndarray, np.ndarray)
            Each np.ndarrays of shape (num_bins, ), representing parameteters of the Beta posterior for each bin.
        """
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

    def update_batch(self, scores: List[float], observations: List[bool]) -> None:
        """
        Update the model parameters with a batch of labeled sample.
        :param scores: List[float]
            A list of scores of samples.
        :param observations: List[bool]
            A list of boolean observations, whether predicted labels are the same as true labels.
        """
        for score, observation in zip(scores, observations):
            self.update(score, observation)

    def calibration_estimation_error(self, ground_truth_model, weight_type='online') -> float:
        """
        Computes the difference between reliability diagram curve generated with the model and the ground truth.
        The difference is computed by taking the weighted average of absolute difference per bin,
            where weight is estimated empirically from ground_truth_model.
        Accuracy per bin is computed as MPE of Beta per bin.
        :param ground_truth_model: SumOfBetaEce
                Ece model trained with all available data.
        :return: float
        """
        theta = self._alpha / (self._alpha + self._beta)
        ground_truth_alpha, ground_truth_beta = ground_truth_model.get_params()
        ground_truth_theta = ground_truth_alpha / (ground_truth_alpha + ground_truth_beta)

        if weight_type == 'online':
            if self._weight is not None:  # pool weights
                weight = self._weight
            else:  # online weights
                tmp = np.sum(self._counts, axis=1)
                weight = tmp / sum(tmp)
        else:
            if ground_truth_model._weight is not None:  # pool weights
                weight = ground_truth_model._weight
            else:  # online weights
                tmp = np.sum(ground_truth_model._counts, axis=1)
                weight = tmp / sum(tmp)

        return np.dot(np.abs(theta - ground_truth_theta), weight)

    def frequentist_calibration_estimation_error(self, ground_truth_model, weight_type='online') -> float:
        """
        Computes the difference between reliability diagram curve generated with the model and the ground truth.
        The difference is computed by taking the weighted average of absolute difference per bin,
            where weight is estimated empirically from ground_truth_model.
        Accuracy per bin is computed in a frequentist's way.
        :param ground_truth_model: SumOfBetaEce
                Ece model trained with all available data.
        :return: float
        """
        accuracy = self._counts[:, 0] / np.sum(self._counts, axis=1)

        ground_truth_alpha, ground_truth_beta = ground_truth_model.get_params()
        ground_truth_theta = ground_truth_alpha / (ground_truth_alpha + ground_truth_beta)

        if weight_type == 'online':
            if self._weight is not None:  # pool weights
                weight = self._weight
            else:  # online weights
                tmp = np.sum(self._counts, axis=1)
                weight = tmp / sum(tmp)
        else:
            if ground_truth_model._weight is not None:  # pool weights
                weight = ground_truth_model._weight
            else:  # online weights
                tmp = np.sum(ground_truth_model._counts, axis=1)
                weight = tmp / sum(tmp)

        return np.dot(np.abs(accuracy - ground_truth_theta), weight)


class ClasswiseEce(Model):
    """
    Model classwise ECE with a SumOfBetaECE for each predicted class.
    """

    def __init__(self, k: int, num_bins: int, pseudocount: float, weight=None, prior=None) -> None:
        """
        :param k: int
            The number of classes
        :param num_bins: int
            The number of bins for evaluating ECE
        :param pseudocount: float
            The strength of priors for accuracy of each bin.
        :param weight: a list of (num_bins, ) arrays of length k
            Weight of each bin. Default: None.
        :param prior: an (number of classes, k, 2) array
            Alpha and beta parameters in the prior Beta distributions.
        """
        self._k = k
        self._num_bins = num_bins

        if weight is None:
            weight = [None] * self._k

        if prior is None:
            self._classwise_ece_models = [SumOfBetaEce(num_bins,
                                                       weight=weight[class_idx],
                                                       pseudocount=pseudocount,
                                                       prior_alpha=None, prior_beta=None)
                                          for class_idx in range(k)]
        else:
            self._classwise_ece_models = [SumOfBetaEce(num_bins,
                                                       weight=weight[class_idx],
                                                       prior_alpha=prior[class_idx, :, 0].squeeze(),
                                                       prior_beta=prior[class_idx, :, 1].squeeze())
                                          for class_idx in range(k)]

    @property
    def eval(self) -> np.ndarray:
        """
        Evaluate ECE for each class.
        :return: An (k,) array of ECE evaluate for each class.
        """
        classwise_ece = np.array([self._classwise_ece_models[class_idx].eval for class_idx in range(self._k)])
        return classwise_ece

    @property
    def frequentist_eval(self) -> np.ndarray:
        """
        Evaluate ECE for each class, accuracy per bin estiamted with the frequentist's method.
        :return: An (k,) array of ECE evaluate for each class.
        """
        classwise_ece = np.array(
            [self._classwise_ece_models[class_idx].frequentist_eval for class_idx in range(self._k)])
        return classwise_ece

        return classwise_ece_variance

    @property
    def variance(self) -> np.ndarray:
        """
        Variance of posterior ECE for each class.
        :param num_samples: int
            The number of samples for Monte Carlo estimation of variance, for each class.
        :return: An (k,) array of variance evaluate for each class.
        """
        classwise_ece_variance = np.array(
            [self._classwise_ece_models[class_idx].variance for class_idx in range(self._k)])
        return classwise_ece_variance

    @property
    def beta_params_mpe(self) -> np.ndarray:
        """
        Computes MPE of accuracy per predicted class per bin. This estimation is used for recalibration.
        :return: (self._k, self._num_bins)
        """
        return np.array([self._classwise_ece_models[class_idx].beta_params_mpe for class_idx in range(self._k)])

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Draw sample eces from the posterior.
        :param num_samples: int
            Number of times to sample from posterior. Default: 1.
        :return: An (k, num_samples) array of samples of theta. If num_samples == 1 then last dimension is squeezed.
        """
        samples = np.array(
            [self._classwise_ece_models[class_idx].sample(num_samples) for class_idx in range(self._k)]).squeeze()
        return samples

    def update(self, category: int, observation: bool, score: float) -> None:
        """
        Update the model parameters with one labeled sample (category, score, observation).
        :param category: int
            The predicted class of the sample.
        :param observation: bool
            Whether predicted label is the same as true label.
        :param score: float
            The confidence of the prediction.
        """
        self._classwise_ece_models[category].update(score, observation)

    def update_batch(self, categories: List[int], observations: List[bool], scores: List[float]) -> None:
        """
        Update the model parameters with a list of  labeled samples.
        :param categories: List[int]
            A list of predicted classes of samples.
        :param observations: List[bool]
            A list of boolean observations.
        :param scores: List[float]
            A list of confidences of predictions.
        """
        for (category, observation, score) in zip(categories, observations, scores):
            self.update(category, observation, score)


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
        """
        Draw sample expected costs from the posterior.
        :param n_samples: int
            Number of times to sample from posterior. Default: 1.
        :return: An (n, n_samples) array of expected costs. If n_samples == 1 then last dimension is squeezed.
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
