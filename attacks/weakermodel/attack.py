import torch
import random
import numpy as np
from tqdm import tqdm
import deeprob.spn.structure as spn
from torch.utils.data import TensorDataset, DataLoader

from constants import *


def generate_adversarial_sample_batched(clts_bag, inputs, perturbations):
	batch_size, num_dims = inputs.shape
	iteration_inputs = inputs.clone().detach().cpu().numpy()

	identity = np.concatenate((np.zeros(num_dims).reshape((1, -1)), np.identity(num_dims)))
	identity = np.tile(identity, (batch_size, 1))

	for iteration in range(perturbations):
		perturbed_set = np.repeat(iteration_inputs, repeats=(num_dims + 1), axis=0)
		perturbed_set = identity + perturbed_set - 2 * np.multiply(identity, perturbed_set)

		lls = []
		for clt in clts_bag:
			outputs = clt.log_likelihood(perturbed_set)
			lls.append(outputs)
		lls = np.array(lls)
		lls_mean = (lls.mean(axis=0)).reshape(-1)

		arg_min_idx = []
		for batch_idx in range(batch_size):
			batch_input_min_idx = np.argmin(
				lls_mean[batch_idx * (num_dims + 1):min((batch_idx + 1) * (num_dims + 1), lls_mean.shape[0])])
			arg_min_idx.append(batch_idx * (num_dims + 1) + batch_input_min_idx)
		iteration_inputs = perturbed_set[arg_min_idx, :]

	adv_sample_batched = iteration_inputs
	return adv_sample_batched


def fetch_bags_of_clts(train_x):
	clt_bag = []
	n_samples, n_features = train_x.shape

	# Initialize the scope and domains
	scope = list(range(n_features))
	domains = [[0, 1]] * n_features

	# Instantiate the random state
	random_state = np.random.RandomState(42)

	for bag_id in range(10):
		sample = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
		train_sample = train_x[np.unique(sample), :]

		# Instantiate and fit a Binary Chow-Liu Tree (CLT)
		clt = spn.BinaryCLT(scope)
		clt.fit(train_sample, domains, alpha=0.01, random_state=random_state)

		clt_bag.append(clt)

	return clt_bag


def generate_adv_dataset(einet, dataset_name, test_data, test_labels, perturbations, device, combine=False,
						 batched=False, train_data=None):
	adv_inputs = test_data.detach().clone()

	batch_size = int(10000 / adv_inputs.shape[1]) if batched else 1

	clts_bag = fetch_bags_of_clts(train_data.detach().clone().cpu().numpy())

	dataset = TensorDataset(adv_inputs)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	data_loader = tqdm(
		data_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Generating adv samples for {}'.format(dataset_name), unit='batch'
	)
	perturbed_inputs = []
	for inputs in data_loader:
		adv_sample = generate_adversarial_sample_batched(clts_bag, inputs[0], perturbations)
		if len(perturbed_inputs) == 0:
			perturbed_inputs = adv_sample
		else:
			perturbed_inputs = np.concatenate((perturbed_inputs, adv_sample), axis=0)
	perturbed_inputs = torch.tensor(perturbed_inputs, device=torch.device(device))
	if combine:
		return torch.cat((adv_inputs, perturbed_inputs))
	else:
		return perturbed_inputs
