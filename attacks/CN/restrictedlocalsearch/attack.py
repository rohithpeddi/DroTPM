import numpy as np
import asyncio


def generate_adversarial_sample(sample, cnet, perturbations, k):
	num_dims = sample.shape[0]

	repeated_sample = np.repeat(sample, [k + 1], axis=0)
	perturbed_set = np.zeros((1, num_dims))
	for row in range(k):
		perturbation = np.array([0] * (num_dims - perturbations) + [1] * perturbations)
		np.random.shuffle(perturbation)
		perturbation = perturbation.reshape(1, -1)

		perturbed_set = np.concatenate((perturbed_set, perturbation), axis=0)

	perturbed_set = repeated_sample + perturbed_set - 2 * np.multiply(repeated_sample, perturbed_set)
	perturbed_input_probabilities = cnet.computeLL_each_datapoint(perturbed_set)

	min_probability_idx = np.argmin(perturbed_input_probabilities, axis=0)
	adv_sample = perturbed_set[min_probability_idx]

	return adv_sample


def generate_adv_dataset(cnet, dataset_name, inputs, perturbations, combine=False, train_data=None):
	adv_inputs = np.copy(inputs)
	original_N, num_dims = inputs.shape

	k = min(max(10, int(0.3 * num_dims)), 100)

	loop = asyncio.get_event_loop()
	looper = asyncio.gather(
		*[generate_adversarial_sample(adv_inputs[i], cnet, perturbations, k) for i in range(original_N)])
	perturbed_inputs = loop.run_until_complete(looper)

	if combine:
		return np.concatenate((inputs, perturbed_inputs), axis=0)
	else:
		return perturbed_inputs
