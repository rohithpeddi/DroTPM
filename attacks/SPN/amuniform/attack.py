import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from constants import *


def fetch_perturbed_idx_appended(num_dims, included_idx):
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


def generate_adv_dataset(einet, dataset_name, inputs, labels, perturbations, device, combine=False, batched=False,
						 train_data=None):
	adv_inputs = inputs.detach().clone()
	original_N, num_dims = inputs.shape

	if dataset_name in SMALL_VARIABLE_DATASETS:
		batch_size = max(1, int(12000 / num_dims)) if batched else 1
	else:
		batch_size = max(1, int(10000 / num_dims)) if batched else 1

	min_perturbed_idx = []
	perturbed_idx_list = fetch_perturbed_idx_appended(num_dims, min_perturbed_idx)
	for i in range(perturbations):
		ll_scores = []
		counter = 1
		for perturbed_idx in perturbed_idx_list:
			if counter % 100 == 0:
				print("Running {}/{}".format(counter, len(perturbed_idx_list)))
			ll_score = generate_log_likelihood_score(dataset_name, einet, inputs, perturbed_idx, batch_size)
			ll_scores.append(ll_score)
			counter = counter + 1
		min_idx = torch.argmin(torch.tensor(ll_scores))
		min_perturbed_idx = perturbed_idx_list[min_idx]
		perturbed_idx_list = fetch_perturbed_idx_appended(num_dims, min_perturbed_idx)

	perturbed_inputs = generate_perturbed_inputs(inputs, min_perturbed_idx)

	if combine:
		return torch.cat((adv_inputs, perturbed_inputs))
	else:
		return perturbed_inputs
