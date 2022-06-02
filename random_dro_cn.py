import numpy as np

import wandb
import cn_base as CN
from CN import CNET, utilM
from constants import *
import argparse
import copy

from attacks.CN.random import attack as random_attack
from deeprob.torch.callbacks import EarlyStopping

# wandb_run = wandb.init(project="DROCN", entity="utd-ml-pgm")

columns = ["attack_type", "perturbations", "standard_mean_ll", "standard_std_ll",
		   "ls1_mean_ll", "ls1_std_ll", "ls3_mean_ll", "ls3_std_ll", "ls5_mean_ll", "ls5_std_ll",
		   "rls1_mean_ll", "rls1_std_ll", "rls3_mean_ll", "rls3_std_ll", "rls5_mean_ll", "rls5_std_ll",
		   "av1_mean_ll", "av1_std_ll", "av3_mean_ll", "av3_std_ll", "av5_mean_ll", "av5_std_ll",
		   "w1_mean_ll", "w1_std_ll", "w3_mean_ll", "w3_std_ll", "w5_mean_ll", "w5_std_ll"]

wandb_tables = dict()


def fetch_wandb_table(dataset_name):
	if dataset_name not in wandb_tables:
		dataset_wandb_tables = dict()
		ll_table = wandb.Table(columns=columns)
		dataset_wandb_tables[LOGLIKELIHOOD_TABLE] = ll_table
		wandb_tables[dataset_name] = dataset_wandb_tables
	return wandb_tables[dataset_name]


def evaluation_message(message):
	print("\n")
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


# ------------------------------------------------------------------------------------------
# ------  TESTING AREA ------
def test_trained_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet, train_x, test_x, perturbations, ll_table, train_attack_type):
	av_mean_ll_dict, av_std_ll_dict = CN.fetch_average_likelihoods_for_data(dataset_name, trained_adv_cnet, test_x)

	def attack_test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet, train_x, test_x,
						 perturbations, attack_type, is_adv):
		mean_ll, std_ll, attack_test_x = CN.test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet,
													  train_x, test_x, perturbations=perturbations,
													  attack_type=attack_type, is_adv=is_adv)
		evaluation_message("{}-{} Mean LL : {}, Std LL : {}".format(attack_type, perturbations, mean_ll, std_ll))
		return mean_ll, std_ll, attack_test_x

	# 1. Original Test Set
	standard_mean_ll, standard_std_ll, standard_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
																		  trained_clean_cnet, train_x, test_x,
																		  perturbations=0,
																		  attack_type=CLEAN, is_adv=False)

	# ------  LOCAL SEARCH AREA ------

	# 2. Local Search - 1 Test Set
	ls1_mean_ll, ls1_std_ll, ls1_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
														   trained_clean_cnet, train_x, test_x,
														   perturbations=1, attack_type=LOCAL_SEARCH,
														   is_adv=True)

	# 3. Local Search - 3 Test Set
	ls3_mean_ll, ls3_std_ll, ls3_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
														   trained_clean_cnet, train_x, test_x,
														   perturbations=3, attack_type=LOCAL_SEARCH,
														   is_adv=True)

	# 4. Local Search - 5 Test Set
	ls5_mean_ll, ls5_std_ll, ls5_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
														   trained_clean_cnet, train_x, test_x,
														   perturbations=5, attack_type=LOCAL_SEARCH,
														   is_adv=True)

	# ------  RESTRICTED LOCAL SEARCH AREA ------

	# 5. Restricted Local Search - 1 Test Set
	rls1_mean_ll, rls1_std_ll, rls1_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
															  trained_clean_cnet, train_x, test_x,
															  perturbations=1,
															  attack_type=RESTRICTED_LOCAL_SEARCH, is_adv=True)

	# 6. Restricted Local Search - 3 Test Set
	rls3_mean_ll, rls3_std_ll, rls3_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
															  trained_clean_cnet, train_x, test_x,
															  perturbations=3,
															  attack_type=RESTRICTED_LOCAL_SEARCH, is_adv=True)

	# 7. Restricted Local Search - 5 Test Set
	rls5_mean_ll, rls5_std_ll, rls5_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
															  trained_clean_cnet, train_x, test_x,
															  perturbations=5,
															  attack_type=RESTRICTED_LOCAL_SEARCH, is_adv=True)

	# ------------- WEAKER MODEL ATTACK --------

	# 11. Weaker model attack - 1 Test Set
	w1_mean_ll, w1_std_ll, w1_test_x = attack_test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet,
														train_x, test_x, perturbations=1,
														attack_type=WEAKER_MODEL, is_adv=True)

	# 12. Weaker model attack - 3 Test Set
	w3_mean_ll, w3_std_ll, w3_test_x = attack_test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet,
														train_x, test_x, perturbations=3,
														attack_type=WEAKER_MODEL, is_adv=True)

	# 13. Weaker model attack - 5 Test Set
	w5_mean_ll, w5_std_ll, w5_test_x = attack_test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet,
														train_x, test_x, perturbations=5,
														attack_type=WEAKER_MODEL, is_adv=True)

	# -------------------------------- LOG LIKELIHOOD TO WANDB TABLES ------------------------------------

	ll_table.add_data(train_attack_type, perturbations, standard_mean_ll, standard_std_ll,
					  ls1_mean_ll, ls1_std_ll, ls3_mean_ll, ls3_std_ll, ls5_mean_ll, ls5_std_ll,
					  rls1_mean_ll, rls1_std_ll, rls3_mean_ll, rls3_std_ll, rls5_mean_ll, rls5_std_ll,
					  av_mean_ll_dict[1], av_std_ll_dict[1], av_mean_ll_dict[3], av_std_ll_dict[3],
					  av_mean_ll_dict[5], av_std_ll_dict[5],
					  w1_mean_ll, w1_std_ll, w3_mean_ll, w3_std_ll, w5_mean_ll, w5_std_ll)


def train_random_dro_from_perturbed_datasets(run_id, trained_cnet, dataset_name, train_x, valid_x, test_x,
											 perturbations, perturbed_training_datasets, attack_type, is_adv,
											 learning_rate):
	# 1. Learn the structure and parameters using the original training set
	if trained_cnet is None:
		trained_clean_cnet = CNET.learn_best_cutset(train_x, valid_x, test_x, max_depth=10)
		trained_cnet = copy.deepcopy(trained_clean_cnet)
	# 2. If adversarial training then
	if is_adv:
		# 3. Randomize training parameters
		print("Keeping the structure and randomizing the parameters for the dataset {} and learning with lr {}".format(
			dataset_name, learning_rate))
		trained_cnet.randomize_params()

		# 4. Start gradient ascent on the concave maximization problem using sub gradient method
		previous_train_ll, current_train_ll = 0, 0
		for epoch in range(MAX_NUM_EPOCHS):
			if epoch > 1 and current_train_ll - previous_train_ll < 1e-3:
				break

			# Finding the dataset with minimum perturbations
			perturbed_likelihoods = []
			for i in range(len(perturbed_training_datasets)):
				perturbed_dataset = perturbed_training_datasets[i]
				perturbed_data_likelihood = trained_cnet.computeLL(perturbed_dataset)
				perturbed_likelihoods.append(perturbed_data_likelihood)

			min_likelihood_dataset_idx = np.argmin(np.array(perturbed_likelihoods))
			print("Minimum Likelihood is observed at idx : {}".format(min_likelihood_dataset_idx))
			adv_train_x = perturbed_training_datasets[min_likelihood_dataset_idx]

			# Using the worst case dataset to update parameters of the cnet
			trained_cnet.sgd_update_params(adv_train_x, eta=learning_rate)
			train_ll, valid_ll, test_ll = CN.evaluate_lls(trained_cnet, train_x, valid_x, test_x, epoch)
			previous_train_ll = current_train_ll
			current_train_ll = valid_ll

	return trained_cnet


def random_dro_cn(run_id, specific_datasets, is_adv, train_attack_type, perturbations, learning_rate):
	if specific_datasets is None:
		specific_datasets = DISCRETE_DATASETS
	else:
		specific_datasets = [specific_datasets] if type(specific_datasets) is not list else specific_datasets

	for dataset_name in specific_datasets:

		dataset_wandb_tables = fetch_wandb_table(dataset_name)
		ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]

		evaluation_message("Dataset : {}".format(dataset_name))

		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = CN.load_dataset(dataset_name)

		evaluation_message("Loading Clean CNet")
		trained_clean_cnet = CN.load_pretrained_cnet(run_id, dataset_name, CLEAN, perturbations)

		# Train a clean CNET on original training data
		if trained_clean_cnet is None:
			evaluation_message("Training clean einet")
			trained_clean_cnet = CN.train_cnet(run_id, trained_clean_cnet, dataset_name, train_x, valid_x, test_x,
											   perturbations, attack_type=CLEAN, is_adv=False,
											   learning_rate=learning_rate)
		if perturbations == 1:
			test_trained_cnet(dataset_name, trained_clean_cnet, trained_clean_cnet, train_x, test_x, perturbations,
							  ll_table, train_attack_type=CLEAN)

		for samples in [10, 30, 50]:
			perturbed_training_datasets = []
			evaluation_message("Training using samples {}".format(samples))
			for sample in range(samples):
				# 1. Generate randomly perturbed dataset
				perturbed_train_x = random_attack.generate_random_perturbed_dataset(train_x, perturbations)
				perturbed_training_datasets.append(perturbed_train_x)

			# Approach-1
			# Use the raw perturbed training samples in inner minimization problem
			specific_filename = "{}_{}_{}".format(dataset_name, perturbations, samples)

			print(specific_filename)
			trained_random_dro_cnet = CN.load_pretrained_cnet(run_id, dataset_name,
															  attack_type=WASSERSTEIN_RANDOM_SAMPLES,
															  perturbations=perturbations,
															  specific_filename=specific_filename)

			if trained_random_dro_cnet is None:
				print("Could not find trained models")
				trained_random_dro_cnet = train_random_dro_from_perturbed_datasets(run_id, trained_clean_cnet,
																				   dataset_name, train_x, valid_x,
																				   test_x,
																				   perturbations,
																				   perturbed_training_datasets,
																				   attack_type=WASSERSTEIN_RANDOM_SAMPLES,
																				   is_adv=True,
																				   learning_rate=learning_rate)

			test_trained_cnet(dataset_name, trained_random_dro_cnet, trained_clean_cnet, train_x, test_x, perturbations,
							  ll_table, train_attack_type=specific_filename)

			CN.save_cnet(run_id, trained_random_dro_cnet, train_x, dataset_name, perturbations,
						 attack_type=WASSERSTEIN_RANDOM_SAMPLES)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dataset', type=str, required=True, help="dataset name")
	parser.add_argument('--runid', type=str, required=True, help="run id")
	parser.add_argument('--lr', type=str, required=True, help="learning rate")
	ARGS = parser.parse_args()
	print(ARGS)

	dataset_name = ARGS.dataset
	run_id = ARGS.runid
	lr = float(ARGS.lr)

	for perturbation in DRO_PERTURBATIONS:
		evaluation_message(
			"Logging values for {}, perturbation {}, train attack type {}".format(dataset_name, perturbation,
																				  WASSERSTEIN_RANDOM_SAMPLES))
		random_dro_cn(run_id=1451, specific_datasets=dataset_name, is_adv=True,
					  train_attack_type=WASSERSTEIN_RANDOM_SAMPLES, perturbations=perturbation, learning_rate=lr)

	dataset_wandb_tables = fetch_wandb_table(dataset_name)
	ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]

# wandb_run.log({"{}-{}-RANDOM-DRO-LL".format(dataset_name, lr): ll_table})
