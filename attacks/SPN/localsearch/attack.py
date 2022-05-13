import torch
import random
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from constants import *


def generate_adversarial_sample_batched(einet, inputs, perturbations, device):
	batch_size, num_dims = inputs.shape
	iteration_inputs = inputs.clone().detach()

	identity = torch.cat((torch.zeros(num_dims, device=torch.device(device)).reshape((1, -1)),
						  torch.eye(num_dims, device=torch.device(device))))
	identity = identity.repeat((batch_size, 1))

	for iteration in range(perturbations):

		perturbed_set = torch.repeat_interleave(iteration_inputs,
												(num_dims + 1) * (
													torch.ones(batch_size, device=torch.device(device)).int()),
												dim=0)
		perturbed_set = identity + perturbed_set - 2 * torch.mul(identity, perturbed_set)

		if num_dims > 500:
			outputs = []
			perturbed_dataset = TensorDataset(perturbed_set)
			perturbed_dataloader = DataLoader(perturbed_dataset, shuffle=False, batch_size=100)
			for perturbed_inputs in perturbed_dataloader:
				outputs.append(einet(perturbed_inputs[0]))
			outputs = (torch.cat(outputs)).clone().detach()
		else:
			outputs = (einet(perturbed_set)).clone().detach()

		arg_min_idx = []
		for batch_idx in range(batch_size):
			batch_input_min_idx = torch.argmin(
				outputs[batch_idx * (num_dims + 1):min((batch_idx + 1) * (num_dims + 1), outputs.shape[0])])
			arg_min_idx.append(batch_idx * (num_dims + 1) + batch_input_min_idx)
		iteration_inputs = perturbed_set[arg_min_idx, :]

	adv_sample_batched = iteration_inputs
	return adv_sample_batched


def generate_adversarial_sample(einet, inputs, perturbations):
	iteration_inputs = inputs.clone().detach()
	for iteration in range(BINARY_DEBD_HAMMING_THRESHOLD):
		num_dims = iteration_inputs.shape[1]
		perturbed_set = []
		for dimension in range(num_dims):
			perturbed_sample = iteration_inputs.clone().detach()
			perturbed_sample[:, dimension] = 1 - perturbed_sample[:, dimension]
			perturbed_set.append(perturbed_sample)
		perturbed_set = torch.cat(perturbed_set)

		outputs = (einet(perturbed_set)).clone().detach()
		min_idx = torch.argmin(outputs)
		iteration_inputs = perturbed_set[min_idx, :]
	adv_sample = iteration_inputs
	return adv_sample


def generate_adv_dataset(einet, dataset_name, inputs, labels, perturbations, device, combine=False, batched=False,
						 train_data=None):
	adv_inputs = inputs.detach().clone()
	original_N, num_dims = inputs.shape

	if dataset_name in SMALL_VARIABLE_DATASETS:
		batch_size = max(1, int(5000 / num_dims)) if batched else 1
	else:
		batch_size = max(1, int(2000 / num_dims)) if batched else 1

	dataset = TensorDataset(adv_inputs)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	data_loader = tqdm(
		data_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Generating adv samples for {}'.format(dataset_name), unit='batch'
	)
	perturbed_inputs = []
	for inputs in data_loader:
		if batched:
			adv_sample = generate_adversarial_sample_batched(einet, inputs[0], perturbations, device)
		else:
			adv_sample = generate_adversarial_sample(einet, inputs[0], perturbations)
		perturbed_inputs.append(adv_sample)
	perturbed_inputs = torch.cat(perturbed_inputs)
	if combine:
		return torch.cat((adv_inputs, perturbed_inputs))
	else:
		return perturbed_inputs
