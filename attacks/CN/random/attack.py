
import random
from collections import Counter
import numpy as np

def generate_random_perturbed_dataset(dataset, perturbations):
	batch_size, num_dims = dataset.shape

	total_perturbations_possible = perturbations * batch_size
	perturbed_batch_idx = random.choices(range(0, batch_size), k=total_perturbations_possible)
	perturbed_batch_idx_dict = Counter(perturbed_batch_idx)
	perturbed_batch_idx_dict = {k: v for k, v in perturbed_batch_idx_dict.items() if v <= min(5, perturbations + 1)}

	perturbed_dataset = np.copy(dataset)
	for batch_idx, perturbs in perturbed_batch_idx_dict.items():
		perturbed_dim_idx = random.sample(range(0, num_dims), perturbs)
		perturbed_dataset[batch_idx, perturbed_dim_idx] = 1 - dataset[batch_idx, perturbed_dim_idx]

	return perturbed_dataset
