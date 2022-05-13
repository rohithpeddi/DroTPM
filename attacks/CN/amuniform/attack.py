import numpy as np
import asyncio


def fetch_perturbed_idx_appended(num_dims, included_idx):
	indices = set(range(0, num_dims)) - set(included_idx)
	perturbed_idx = []
	for idx in indices:
		perturbed = included_idx.copy()
		perturbed.append(idx)
		perturbed_idx.append(perturbed)
	return perturbed_idx


def generate_perturbed_inputs(inputs, perturbed_idx):
	perturbed_inputs = np.copy(inputs)
	perturbed_inputs[:, perturbed_idx] = 1 - perturbed_inputs[:, perturbed_idx]
	return perturbed_inputs


def background(f):
	def wrapped(*args, **kwargs):
		return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

	return wrapped


@background
def generate_log_likelihood_score(cnet, inputs, perturbed_idx):
	perturbed_inputs = generate_perturbed_inputs(inputs, perturbed_idx)
	ll_score = cnet.computeLL(perturbed_inputs)
	return ll_score


def generate_adv_dataset(cnet, dataset_name, inputs, perturbations, combine=False, train_data=None):
	adv_inputs = np.copy(inputs)
	original_N, num_dims = inputs.shape

	min_perturbed_idx = []
	perturbed_idx_list = fetch_perturbed_idx_appended(num_dims, min_perturbed_idx)
	for i in range(perturbations):
		loop = asyncio.get_event_loop()
		looper = asyncio.gather(
			*[generate_log_likelihood_score(cnet, adv_inputs, perturbed_idx) for perturbed_idx in perturbed_idx_list])
		ll_scores = np.asarray(loop.run_until_complete(looper))

		min_log_score_idx = np.argmin(ll_scores)
		min_perturbed_idx = perturbed_idx_list[min_log_score_idx]
		perturbed_idx_list = fetch_perturbed_idx_appended(num_dims, min_perturbed_idx)

	perturbed_inputs = generate_perturbed_inputs(inputs, min_perturbed_idx)

	if combine:
		return np.concatenate((inputs, perturbed_inputs), axis=0)
	else:
		return perturbed_inputs
