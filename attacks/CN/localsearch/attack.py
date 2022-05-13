import numpy as np
import asyncio


def generate_adversarial_sample(sample, cnet, perturbations):
	num_dims = sample.shape[0]
	# 1. Use the current sample + one bit flipped
	identity = np.concatenate((np.zeros((1, num_dims)), np.eye(num_dims)), axis=0)
	iteration_input = sample
	for iteration in range(perturbations):
		perturbed_set = np.repeat(iteration_input, [num_dims + 1], axis=0)
		perturbed_set = identity + perturbed_set - 2 * np.multiply(identity, perturbed_set)
		perturbed_input_probabilities = cnet.computeLL_each_datapoint(perturbed_set)

		min_probability_idx = np.argmin(perturbed_input_probabilities, axis=0)
		iteration_input = perturbed_set[min_probability_idx]
	adv_sample = iteration_input
	return adv_sample


def generate_adv_dataset(cnet, dataset_name, inputs, perturbations, combine=False, train_data=None):
	adv_inputs = np.copy(inputs)
	original_N, num_dims = inputs.shape

	loop = asyncio.get_event_loop()
	looper = asyncio.gather(
		*[generate_adversarial_sample(adv_inputs[i], cnet, perturbations) for i in range(original_N)])
	perturbed_inputs = loop.run_until_complete(looper)

	if combine:
		return np.concatenate((inputs, perturbed_inputs), axis=0)
	else:
		return perturbed_inputs
