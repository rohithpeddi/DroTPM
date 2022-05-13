import numpy as np
import asyncio


def background(f):
	def wrapped(*args, **kwargs):
		return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

	return wrapped


@background
def generate_adversarial_sample(sample, cnet, perturbations):
	sample = np.asarray(sample)
	num_dims = sample.shape[0]
	# 1. Use the current sample + one bit flipped
	identity = np.concatenate((np.zeros((1, num_dims)), np.eye(num_dims)), axis=0)
	iteration_input = sample
	for iteration in range(perturbations):
		perturbed_set = np.tile(iteration_input, (num_dims + 1, 1))
		perturbed_set = (identity + perturbed_set - 2 * np.multiply(identity, perturbed_set)).astype(int)
		perturbed_input_probabilities = cnet.computeLL_each_datapoint(perturbed_set)

		min_probability_idx = np.argmin(perturbed_input_probabilities, axis=0)
		iteration_input = perturbed_set[min_probability_idx, :]
	adv_sample = iteration_input
	return tuple(adv_sample)


def generate_adv_dataset(cnet, dataset_name, inputs, perturbations, combine=False, train_data=None):
	adv_inputs = np.copy(inputs)
	original_N, num_dims = inputs.shape

	loop = asyncio.get_event_loop()
	looper = asyncio.gather(
		*[generate_adversarial_sample(tuple(adv_inputs[i, :]), cnet, perturbations) for i in range(original_N)])
	perturbed_inputs = np.asarray(loop.run_until_complete(looper))

	if combine:
		return np.concatenate((inputs, perturbed_inputs), axis=0)
	else:
		return perturbed_inputs
