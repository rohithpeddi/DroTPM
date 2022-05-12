import os

import datasets
from constants import *
from utils import mkdir_p


# 1. Load dataset
def load_dataset(dataset_name):
	train_x, train_labels, test_x, test_labels, valid_x, valid_labels = None, None, None, None, None, None
	if dataset_name in DEBD_DATASETS:
		train_x, test_x, valid_x = datasets.load_debd(dataset_name)
		train_labels, valid_labels, test_labels = None, None, None
	return train_x, train_labels, test_x, test_labels, valid_x, valid_labels


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

		if attack_type == WASSERSTEIN_RANDOM_SAMPLES:
			file_name = os.path.join(RUN_MODEL_DIRECTORY,
									 "{}_adv.npz".format(specific_filename))
		else:
			file_name = os.path.join(RUN_MODEL_DIRECTORY,
									 "{}_adv.npz".format(dataset_name))


# 3. Save Model
def save_cnet():
	pass


# 4. Train cutset network
def train_cnet(run_id, dataset_name, train_x, valid_x, test_x, perturbations, attack_type=CLEAN, is_adv=False):
	pass


def fetch_attack_method(attack_type):
	pass


def fetch_adv_data():
	pass


def test_cnet(dataset_name, trained_cnet, trained_clean_cnet, train_x, test_x, perturbations, attack_type, is_adv):
	pass


def test_conditional_cnet(dataset_name, trained_cnet, test_attack_type, perturbations, evidence_percentage,
						  test_x):
	pass


def fetch_average_likelihoods_for_data(dataset_name, trained_cnet, test_x):
	pass


def fetch_average_conditional_likelihoods_for_data(dataset_name, trained_cnet, test_x):
	pass
