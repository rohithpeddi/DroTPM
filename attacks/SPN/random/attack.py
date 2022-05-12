import torch
import random
from collections import Counter


def generate_random_perturbed_dataset(dataset, perturbations, device):
	batch_size, num_dims = dataset.shape

	total_perturbations_possible = perturbations * batch_size
	perturbed_batch_idx = random.choices(range(0, batch_size), k=total_perturbations_possible)
	perturbed_batch_idx_dict = Counter(perturbed_batch_idx)
	perturbed_batch_idx_dict = {k: v for k, v in perturbed_batch_idx_dict.items() if v <= min(5, perturbations + 1)}

	perturbed_dataset = dataset.clone().detach()
	for batch_idx, perturbs in perturbed_batch_idx_dict.items():
		perturbed_dim_idx = random.sample(range(0, num_dims), perturbs)
		perturbed_dataset[batch_idx, perturbed_dim_idx] = 1 - dataset[batch_idx, perturbed_dim_idx]

	# random_perturbations = torch.rand(batch_size, num_dims, device=device)
	# perturbations_quant = torch.topk(random_perturbations, perturbations, largest=False)[0][:, -1:]
	# boolean_mask = random_perturbations <= perturbations_quant
	# final_perturbations = torch.where(boolean_mask, torch.tensor(1, device=device), torch.tensor(0, device=device))
	#
	# perturbed_dataset = dataset + final_perturbations - 2 * torch.mul(dataset, final_perturbations)
	return perturbed_dataset
