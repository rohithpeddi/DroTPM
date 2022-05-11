import torch
import random
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from constants import *


# def generate_adversarial_sample_batched(einet, inputs, perturbations, k=10):
# 	batch_size, num_dims = inputs.shape
# 	iteration_inputs = inputs.clone().detach()
#
# 	for iteration in range(perturbations):
# 		# Always retain the current element for comparison
# 		dim_idx = random.sample(range(1, num_dims+1), k)
# 		dim_idx.append(0)
#
# 		identity = torch.cat((torch.zeros(num_dims, device=torch.device(device)).reshape((1, -1)),
# 							  torch.eye(num_dims, device=torch.device(device))))
#
# 		# Pick (k+1) nearest neighbours and use them for search
# 		identity = identity[dim_idx, :]
# 		identity = identity.repeat((batch_size, 1))
#
# 		perturbed_set = torch.repeat_interleave(iteration_inputs,
# 												(k+1) * (torch.ones(batch_size, device=torch.device(device)).int()),
# 												dim=0)
# 		perturbed_set = identity + perturbed_set - 2 * torch.mul(identity, perturbed_set)
#
# 		if num_dims > 500:
# 			outputs = []
# 			perturbed_dataset = TensorDataset(perturbed_set)
# 			perturbed_dataloader = DataLoader(perturbed_dataset, shuffle=False, batch_size=100)
# 			for perturbed_inputs in perturbed_dataloader:
# 				outputs.append(einet(perturbed_inputs[0]))
# 			outputs = (torch.cat(outputs)).clone().detach()
# 		else:
# 			outputs = (einet(perturbed_set)).clone().detach()
#
# 		arg_min_idx = []
# 		for batch_idx in range(batch_size):
# 			batch_input_min_idx = torch.argmin(
# 				outputs[batch_idx * (k+1):min((batch_idx + 1) * (k+1), outputs.shape[0])])
# 			arg_min_idx.append(batch_idx * (k+1) + batch_input_min_idx)
# 		iteration_inputs = perturbed_set[arg_min_idx, :]
#
# 	adv_sample_batched = iteration_inputs
# 	return adv_sample_batched


def generate_adversarial_sample_batched(einet, inputs, perturbations, device, k=10):
	batch_size, num_dims = inputs.shape
	iteration_inputs = inputs.clone().detach()

	for iteration in range(perturbations):
		# Always retain the current element for comparison
		dim_idx = random.sample(range(1, num_dims + 1), k)
		dim_idx.append(0)

		identity = torch.cat((torch.zeros(num_dims, device=torch.device(device)).reshape((1, -1)),
							  torch.eye(num_dims, device=torch.device(device))))

		# Pick (k+1) nearest neighbours and use them for search
		identity = identity[dim_idx, :]
		identity = identity.repeat((batch_size, 1))

		perturbed_set = torch.repeat_interleave(iteration_inputs,
												(k + 1) * (torch.ones(batch_size, device=torch.device(device)).int()),
												dim=0)
		perturbed_set = identity + perturbed_set - 2 * torch.mul(identity, perturbed_set)

		arg_min_idx = []
		if iteration == perturbations - 1:
			print("Finding minimum among perturbations")
			if num_dims > 500:
				outputs = []
				perturbed_dataset = TensorDataset(perturbed_set)
				perturbed_dataloader = DataLoader(perturbed_dataset, shuffle=False, batch_size=100)
				for perturbed_inputs in perturbed_dataloader:
					outputs.append(einet(perturbed_inputs[0]))
				outputs = (torch.cat(outputs)).clone().detach()
			else:
				outputs = (einet(perturbed_set)).clone().detach()

			for batch_idx in range(batch_size):
				batch_input_min_idx = torch.argmin(
					outputs[batch_idx * (k + 1):min((batch_idx + 1) * (k + 1), outputs.shape[0])])
				arg_min_idx.append(batch_idx * (k + 1) + batch_input_min_idx)
		else:
			for batch_idx in range(batch_size):
				batch_input_min_idx = random.randint(0, k)
				arg_min_idx.append(batch_idx * (k + 1) + batch_input_min_idx)

		iteration_inputs = perturbed_set[arg_min_idx, :]

	adv_sample_batched = iteration_inputs
	return adv_sample_batched


def generate_adv_dataset(einet, dataset_name, inputs, labels, perturbations, device, combine=False, batched=False,
						 train_data=None):
	adv_inputs = inputs.detach().clone()
	original_N, num_dims = inputs.shape

	k = min(max(10, int(0.3 * num_dims)), 50)
	if dataset_name in SMALL_VARIABLE_DATASETS:
		batch_size = max(1, int(5000 / k)) if batched else 1
	else:
		batch_size = max(1, int(1000 / k)) if batched else 1

	dataset = TensorDataset(adv_inputs)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	data_loader = tqdm(
		data_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Generating adv samples for {}'.format(dataset_name), unit='batch'
	)
	perturbed_inputs = []
	for inputs in data_loader:
		if batched:
			adv_sample = generate_adversarial_sample_batched(einet, inputs[0], perturbations, device, k=k)
		else:
			AssertionError("Not implemented error")
		perturbed_inputs.append(adv_sample)
	perturbed_inputs = torch.cat(perturbed_inputs)
	if combine:
		return torch.cat((adv_inputs, perturbed_inputs))
	else:
		return perturbed_inputs
