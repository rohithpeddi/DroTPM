import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from itertools import combinations

from constants import *

############################################################################


device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################


def fetch_perturbed_idx(perturbations, num_dims):
	indices = list(range(0, num_dims))
	return list(combinations(indices, perturbations))


def fetch_perturbed_idx_appended(perturbations, num_dims, included_idx):
	indices = set(range(0, num_dims)) - set(included_idx)
	perturbed_idx = []
	for idx in indices:
		perturbed = included_idx.copy()
		perturbed.append(idx)
		perturbed_idx.append(perturbed)
	return perturbed_idx


def generate_perturbed_inputs(inputs, perturbed_idx):
	perturbed_inputs = inputs.detach().clone()
	perturbed_inputs[:, perturbed_idx] = 1 - perturbed_inputs[:, perturbed_idx]
	return perturbed_inputs


def generate_log_likelihood_score(dataset_name, einet, inputs, perturbed_idx, batch_size):
	perturbed_inputs = generate_perturbed_inputs(inputs, perturbed_idx)
	dataset = TensorDataset(perturbed_inputs)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	data_loader = tqdm(
		data_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Generating ll scores for {}'.format(dataset_name), unit='batch'
	)

	ll_scores = []
	with torch.no_grad():
		for batched_inputs in data_loader:
			ll_scores.append(einet(batched_inputs[0]))

	del dataset, data_loader, perturbed_inputs
	return torch.sum(torch.cat(ll_scores))


def generate_adv_dataset(einet, dataset_name, inputs, labels, perturbations, combine=False, batched=False,
						 train_data=None):
	adv_inputs = inputs.detach().clone()
	original_N, num_dims = inputs.shape

	if dataset_name in SMALL_VARIABLE_DATASETS:
		batch_size = max(1, int(12000 / num_dims)) if batched else 1
	else:
		batch_size = max(1, int(1800 / num_dims)) if batched else 1

	min_perturbed_idx = []
	perturbed_idx_list = fetch_perturbed_idx_appended(perturbations, num_dims, min_perturbed_idx)
	for i in range(perturbations):
		ll_scores = []
		for perturbed_idx in perturbed_idx_list:
			ll_score = generate_log_likelihood_score(dataset_name, einet, inputs, perturbed_idx, batch_size)
			ll_scores.append(ll_score)
		min_idx = torch.argmin(torch.tensor(ll_scores))
		min_perturbed_idx = perturbed_idx_list[min_idx]
		perturbed_idx_list = fetch_perturbed_idx_appended(perturbations, num_dims, min_perturbed_idx)


	# # 1. Fetch a list of perturbed indices - Exhaustive search over the dimension space
	# perturbed_idx_list = fetch_perturbed_idx(perturbations, num_dims)
	#
	# # 2. Loop through each set of perturbed indices and pick the one that has least log-likelihood score
	# ll_scores = []
	# for perturbed_idx in perturbed_idx_list:
	# 	ll_score = generate_log_likelihood_score(dataset_name, einet, inputs, perturbed_idx, batch_size)
	# 	ll_scores.append(ll_score)
	# min_idx = torch.argmin(torch.tensor(ll_scores))

	perturbed_inputs = generate_perturbed_inputs(inputs, min_perturbed_idx)

	if combine:
		return torch.cat((adv_inputs, perturbed_inputs))
	else:
		return perturbed_inputs
