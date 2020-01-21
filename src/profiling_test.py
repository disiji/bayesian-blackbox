import numpy as np
from models import SumOfBetaEce

num_bins = 10
weight = np.ones((num_bins,))
alpha = np.ones((num_bins,))
beta = np.ones((num_bins,))

model = SumOfBetaEce(num_bins, weight, alpha, beta)
model.sample(100)