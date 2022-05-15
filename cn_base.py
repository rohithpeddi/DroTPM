import copy
import os
import random

import numpy as np

import datasets
from constants import *
from utils import mkdir_p

from attacks.CN.localsearch import attack as local_search
from attacks.CN.restrictedlocalsearch import attack as restricted_local_search
from attacks.CN.amuniform import attack as am_uniform
from attacks.CN.random import attack as wasserstein_random_samples
from attacks.CN.weakermodel import attack as weaker_model

from CN import CNET, utilM
from CN.Util import Util


# 1. Load dataset
def load_dataset(dataset_name):
	train_x, train_labels, test_x, test_labels, valid_x, valid_labels = None, None, None, None, None, None
	if dataset_name in DEBD_DATASETS:
		train_x, test_x, valid_x = datasets.load_debd(dataset_name)
		train_labels, valid_labels, test_labels = None, None, None
	return train_x, valid_x, test_x, train_labels, valid_labels, test_labels


# 2. Load pretrained structure of the dataset if present
def load_pretrained_cnet(run_id, dataset_name, attack_type, perturbations, specific_filename=None):
	cnet = None
	if attack_type is None or attack_type == CLEAN:
		RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), CLEAN_CNET_MODEL_DIRECTORY)
		mkdir_p(RUN_MODEL_DIRECTORY)
		file_name = os.path.join(RUN_MODEL_DIRECTORY, "{}.npz".format(dataset_name))
	else:
		if attack_type == AMBIGUITY_SET_UNIFORM:
			RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id),
											   AMU_CNET_MODEL_DIRECTORY + "/{}".format(perturbations))
		elif attack_type == WASSERSTEIN_RANDOM_SAMPLES:
			RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id),
											   WASSERSTEIN_SAMPLES_CNET_MODEL_DIRECTORY + "/{}".format(perturbations))

		if attack_type == WASSERSTEIN_RANDOM_SAMPLES:
			file_name = os.path.join(RUN_MODEL_DIRECTORY, "{}_adv.npz".format(specific_filename))
		else:
			file_name = os.path.join(RUN_MODEL_DIRECTORY, "{}_adv.npz".format(dataset_name))

	return None


def get_filename(run_id, dataset_name, perturbations, attack_type):
	RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), CLEAN_CNET_MODEL_DIRECTORY)
	mkdir_p(RUN_MODEL_DIRECTORY)
	file_name = os.path.join(RUN_MODEL_DIRECTORY, "{}_{}_{}.npz".format(dataset_name, perturbations, attack_type))
	return file_name


# 3. Save Model
def save_cnet(run_id, trained_cnet, train_x, dataset_name, perturbations, attack_type):
	main_dict = {}
	utilM.save_cutset(main_dict, trained_cnet.tree, np.arange(train_x.shape[1]), ccpt_flag=True)
	filename = get_filename(run_id, dataset_name, perturbations, attack_type)
	np.savez_compressed(filename, module=main_dict)


# def fetch_learning_rate(dataset_name, train_x):
# 	xycounts = Util.compute_xycounts(train_x) + 1  # laplace correction
# 	xcounts = Util.compute_xcounts(train_x) + 2  # laplace correction
# 	xyprob = Util.normalize2d(xycounts)
# 	xprob = Util.normalize1d(xcounts)
#
# 	learning_rate = 1/max(np.max(xycounts/(1-xyprob)), np.max(xcounts/(1-xprob)))
# 	learning_rate = 0.1
# 	print("Found optimal learning rate for dataset {} as {}, standard slow one {}".format(dataset_name, learning_rate, min(DEFAULT_CNET_LEARNING_RATE, 1 / (train_x.shape[0] * 100))))
# 	return learning_rate


# 4. Train cutset network
def train_cnet(run_id, trained_cnet, dataset_name, train_x, valid_x, test_x, perturbations, attack_type=CLEAN,
			   is_adv=False, learning_rate=0.1):
	# 1. Learn the structure and parameters using the original training set
	if trained_cnet is None:
		trained_clean_cnet = CNET.learn_best_cutset(train_x, valid_x, test_x, max_depth=10)
		trained_cnet = copy.deepcopy(trained_clean_cnet)
	# 2. If adversarial training then
	if is_adv:
		# 3. Randomize training parameters
		print("Keeping the structure and randomizing the parameters for the dataset {} and learning with lr {}".format(dataset_name, learning_rate))
		trained_cnet.randomize_params()

		# 4. Start gradient ascent on the concave maximization problem using sub gradient method
		previous_train_ll, current_train_ll = 0, 0
		for epoch in range(MAX_NUM_EPOCHS):
			if epoch > 1 and abs(current_train_ll - previous_train_ll) < 1e-3:
				break
			adv_train_x = fetch_adv_data(trained_cnet, dataset_name, train_x, train_x, perturbations, attack_type,
										 combine=False)
			trained_cnet.sgd_update_params(adv_train_x, eta=learning_rate)
			train_ll, valid_ll, test_ll = evaluate_lls(trained_cnet, train_x, valid_x, test_x, epoch)
			previous_train_ll = current_train_ll
			current_train_ll = valid_ll

	save_cnet(run_id, trained_cnet, train_x, dataset_name, perturbations, attack_type)
	return trained_cnet


def evaluate_lls(cnet, train_x, valid_x, test_x, epoch):
	train_ll = np.sum(cnet.getWeights(train_x)) / train_x.shape[0]
	valid_ll = np.sum(cnet.getWeights(valid_x)) / valid_x.shape[0]
	test_ll = np.sum(cnet.getWeights(test_x)) / test_x.shape[0]
	print("[{}] train LL {} valid LL {} test LL {}".format(epoch, train_ll, valid_ll, test_ll))
	return train_ll, valid_ll, test_ll


def fetch_attack_method(attack_type):
	if attack_type == RESTRICTED_LOCAL_SEARCH:
		return restricted_local_search
	elif attack_type == LOCAL_SEARCH:
		return local_search
	elif attack_type == WEAKER_MODEL:
		return weaker_model
	elif attack_type == AMBIGUITY_SET_UNIFORM:
		return am_uniform
	elif attack_type == WASSERSTEIN_RANDOM_SAMPLES:
		return wasserstein_random_samples


def fetch_adv_data(cnet, dataset_name, train_data, test_data, perturbations, attack_type, combine=False):
	attack = fetch_attack_method(attack_type)
	adv_data = attack.generate_adv_dataset(cnet=cnet, dataset_name=dataset_name, inputs=test_data,
										   perturbations=perturbations, combine=combine, train_data=train_data)
	print("Fetched adversarial examples : {}/{}".format(adv_data.shape[0], test_data.shape[0]))
	return adv_data


def get_stats(likelihoods):
	mean_ll = np.mean(likelihoods)
	stddev_ll = 2.0 * np.std(likelihoods) / np.sqrt(len(likelihoods))
	return mean_ll, stddev_ll


def test_cnet(dataset_name, trained_cnet, data_cnet, train_data, test_data, perturbations, attack_type, is_adv):
	if is_adv:
		test_data = fetch_adv_data(data_cnet, dataset_name, train_data, test_data, perturbations, attack_type,
								   combine=False)

	test_lls = trained_cnet.getWeights(test_data)
	mean_ll, stddev_ll = get_stats(test_lls)
	return mean_ll, stddev_ll, test_data


def fetch_average_likelihoods_for_data(dataset_name, trained_cnet, test_x,
									   average_repeat_size=DEFAULT_AVERAGE_REPEAT_SIZE):
	test_data_size, num_dims = test_x.shape
	likelihoods = dict()

	for idx in range(test_data_size):
		test_sample = np.copy(test_x[idx])
		iteration_inputs = test_sample
		num_repetitions = min(num_dims, average_repeat_size)

		for perturbation in range(1, 6):
			dim_idx = random.sample(range(1, num_dims + 1), num_repetitions)
			dim_idx.append(0)

			identity = np.concatenate((np.zeros((1, num_dims)), np.eye(num_dims)), axis=0)
			identity = identity[dim_idx, :]
			if perturbation == 1:
				perturbed_set = np.tile(iteration_inputs, (num_repetitions + 1, 1))

			perturbed_set = identity + perturbed_set - 2 * np.multiply(identity, perturbed_set)
			iteration_inputs = perturbed_set.astype(int)

			if perturbation in PERTURBATIONS:
				ll_sample = trained_cnet.getWeights(iteration_inputs)
				if perturbation not in likelihoods:
					likelihoods[perturbation] = []
				(likelihoods[perturbation]).append(ll_sample.mean())

	av_mean_dict = dict()
	av_std_dict = dict()
	for perturbation in [1, 3, 5]:
		lls = np.array(likelihoods[perturbation])
		mean_ll, stddev_ll = get_stats(lls)
		av_mean_dict[perturbation] = mean_ll
		av_std_dict[perturbation] = stddev_ll
	return av_mean_dict, av_std_dict


def test_conditional_cnet(dataset_name, trained_cnet, test_attack_type, perturbations, evidence_percentage, test_x):
	pass


def fetch_average_conditional_likelihoods_for_data(dataset_name, trained_cnet, test_x):
	pass
