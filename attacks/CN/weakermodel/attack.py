import numpy as np

import deeprob.spn.structure as spn

import asyncio


def background(f):
	def wrapped(*args, **kwargs):
		return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

	return wrapped


@background
def generate_adversarial_sample(inputs, clts_bag, perturbations):
	batch_size, num_dims = ((np.asarray(inputs)).reshape(1, -1)).shape
	iteration_inputs = np.copy(inputs)

	identity = np.concatenate((np.zeros(num_dims).reshape((1, -1)), np.identity(num_dims)))
	identity = np.tile(identity, (batch_size, 1))

	for iteration in range(perturbations):
		perturbed_set = np.tile(iteration_inputs, (num_dims+1, 1))
		perturbed_set = (identity + perturbed_set - 2 * np.multiply(identity, perturbed_set)).astype(int)

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

	adv_sample = iteration_inputs
	return tuple(adv_sample.reshape(-1))


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


def generate_adv_dataset(cnet, dataset_name, inputs, perturbations, combine=False, train_data=None):
	adv_inputs = np.copy(inputs)
	original_N, num_dims = adv_inputs.shape

	clts_bag = fetch_bags_of_clts(np.copy(train_data))

	loop = asyncio.get_event_loop()
	looper = asyncio.gather(
		*[generate_adversarial_sample(tuple(adv_inputs[i, :]), clts_bag, perturbations) for i in range(original_N)])
	perturbed_inputs = np.asarray(loop.run_until_complete(looper))

	if combine:
		return np.concatenate((inputs, perturbed_inputs), axis=0)
	else:
		return perturbed_inputs

