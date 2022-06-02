import numpy as np
import argparse
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import wandb
import einet_sgd_base_spn as SPN
from EinsumNetworkSGD import EinsumNetwork
from EinsumNetworkSGD.ExponentialFamilyArray import NormalArray, CategoricalArray, BinomialArray
from constants import *
from deeprob.torch.callbacks import EarlyStopping
from attacks.SPN.random import attack as random_attack

wandb_run = wandb.init(project="DRO-Random-SGD-SPN", entity="utd-ml-pgm")

columns = ["attack_type", "perturbations", "standard_mean_ll", "ls1_mean_ll", "ls3_mean_ll", "ls5_mean_ll",
		   "rls1_mean_ll", "rls3_mean_ll", "rls5_mean_ll", "av1_mean_ll", "av3_mean_ll", "av5_mean_ll",
		   "w1_mean_ll", "w3_mean_ll", "w5_mean_ll"]

wandb_tables = dict()


def fetch_wandb_table(dataset_name):
	if dataset_name not in wandb_tables:
		dataset_wandb_tables = dict()

		ll_table = wandb.Table(columns=columns)
		dataset_wandb_tables[LOGLIKELIHOOD_TABLE] = ll_table

		cll_wandb_tables = dict()
		for evidence_percentage in EVIDENCE_PERCENTAGES:
			cll_table = wandb.Table(columns=columns)
			cll_wandb_tables[evidence_percentage] = cll_table
		dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES] = cll_wandb_tables

		wandb_tables[dataset_name] = dataset_wandb_tables

	return wandb_tables[dataset_name]


def evaluation_message(message):
	print("\n")
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


def fetch_einet_args_discrete(dataset_name, num_var, exponential_family, exponential_family_args):
	einet_args = dict()
	einet_args[NUM_VAR] = num_var
	num_distributions, batch_size = None, DEFAULT_TRAIN_BATCH_SIZE
	learning_rate = 0.1
	weight_decay = 1e-4
	batch_size = 700

	if dataset_name in ['plants', 'tretail']:
		num_distributions = 20
	if dataset_name in ['accidents']:
		num_distributions = 20
		batch_size = 600
	elif dataset_name in ['pumsb_star', 'kosarek', 'msweb']:
		num_distributions = 10
	elif dataset_name in ['nltcs']:
		num_distributions = 10
		batch_size = 700
	elif dataset_name in ['msnbc']:
		num_distributions = 10
		batch_size = 500
	elif dataset_name in ['kdd']:
		num_distributions = 10
		batch_size = 700
	elif dataset_name in ['baudio', 'jester', 'book']:
		num_distributions = 10
	elif dataset_name in ['bnetflix']:
		num_distributions = 10
		batch_size = 600
	elif dataset_name in ['tmovie']:
		num_distributions = 20
	elif dataset_name in ['dna']:
		num_distributions = 20
		batch_size = 700
	elif dataset_name in ['cwebkb', 'cr52']:
		num_distributions = 10
	elif dataset_name in ['bbc']:
		batch_size = 700
		num_distributions = 10
	elif dataset_name in ['c20ng']:
		num_distributions = 10
	elif dataset_name in ['ad']:
		num_distributions = 10
	elif dataset_name in [BINARY_MNIST, BINARY_FASHION_MNIST]:
		num_distributions = 10
		batch_size = 100
	einet_args[NUM_SUMS] = num_distributions
	einet_args[NUM_CLASSES] = GENERATIVE_NUM_CLASSES
	einet_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
	einet_args[EXPONENTIAL_FAMILY] = exponential_family
	einet_args[EXPONENTIAL_FAMILY_ARGS] = exponential_family_args
	einet_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS
	einet_args[BATCH_SIZE] = batch_size
	einet_args[LEARNING_RATE] = learning_rate
	einet_args[WEIGHT_DECAY] = weight_decay
	return einet_args


def train_einet(dataset_name, einet, train_x, valid_x, test_x, batch_size, learning_rate, weight_decay):
	early_stopping = EarlyStopping(einet, patience=DEFAULT_EINET_PATIENCE, filepath="{}_{}".format(dataset_name, EARLY_STOPPING_FILE),
								   delta=EARLY_STOPPING_DELTA)
	optimizer = optim.Adam(list(einet.parameters()), lr=learning_rate, weight_decay=weight_decay)
	train_dataset = TensorDataset(train_x)
	for epoch_count in range(MAX_NUM_EPOCHS):
		train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
		train_dataloader = tqdm(
			train_dataloader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
			desc='Training epoch : {}, for dataset : {}'.format(epoch_count, dataset_name),
			unit='batch'
		)
		einet.train()
		for inputs in train_dataloader:
			optimizer.zero_grad()
			outputs = einet.forward(inputs[0])
			log_likelihood = outputs.sum()
			objective = -log_likelihood
			objective.backward()
			optimizer.step()
		train_ll, valid_ll, test_ll = SPN.evaluate_lls(einet, train_x, valid_x, test_x,
													   epoch_count=epoch_count)
		if epoch_count > 1:
			early_stopping(-valid_ll, epoch_count)
			if early_stopping.should_stop:
				print("Early Stopping... {}".format(early_stopping))
				break
	return einet


def train_random_dro_from_perturbed_datasets(dataset_name, perturbed_training_datasets, einet, train_x, valid_x, test_x,
											 batch_size, learning_rate, weight_decay):
	early_stopping = EarlyStopping(einet, patience=DEFAULT_EINET_PATIENCE, filepath="{}_{}".format(dataset_name, EARLY_STOPPING_FILE),
								   delta=EARLY_STOPPING_DELTA)
	optimizer = optim.Adam(list(einet.parameters()), lr=learning_rate, weight_decay=weight_decay)

	train_dataset = TensorDataset(train_x)
	for epoch_count in range(1, MAX_NUM_EPOCHS):
		train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
		einet.train()

		train_dataloader = tqdm(
			train_dataloader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
			desc='Training epoch : {}, for dataset : {}'.format(epoch_count, dataset_name),
			unit='batch'
		)

		batch_counter = 0
		for inputs in train_dataloader:
			einet.train()
			optimizer.zero_grad()

			batch_input = inputs[0]

			outputs = einet.forward(batch_input)
			log_likelihood = outputs.sum()
			objective = -log_likelihood
			objective.backward()

			optimizer.step()
			batch_counter = batch_counter + 1

		train_ll, valid_ll, test_ll = SPN.evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=epoch_count)

		if epoch_count > 1:
			early_stopping(-valid_ll, epoch_count)
			if early_stopping.should_stop and epoch_count > 5:
				print("Early Stopping... {}".format(early_stopping))
				break

		einet.eval()
		perturbed_likelihoods = []
		for i in range(len(perturbed_training_datasets)):
			perturbed_dataset = perturbed_training_datasets[i]
			perturbed_data_likelihood = EinsumNetwork.eval_loglikelihood_batched(einet, perturbed_dataset,
																				 batch_size=EVAL_BATCH_SIZE) / \
										train_x.shape[0]
			perturbed_likelihoods.append(perturbed_data_likelihood)
		min_likelihood_dataset_idx = torch.argmin(torch.tensor(perturbed_likelihoods))
		print("Minimum Likelihood is observed at idx : {}".format(min_likelihood_dataset_idx))
		train_dataset = TensorDataset(perturbed_training_datasets[min_likelihood_dataset_idx])

	einet.load_state_dict(early_stopping.get_best_state())
	train_ll, valid_ll, test_ll = SPN.evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=epoch_count)

	return einet


def test_trained_einet(dataset_name, trained_adv_einet, trained_clean_einet, train_x, test_x, test_labels, ll_table,
					   cll_tables, train_attack_type, perturbations):
	# ------  AVERAGE ATTACK AREA ------
	av_mean_ll_dict, av_std_ll_dict = SPN.fetch_average_likelihoods_for_data(dataset_name, trained_adv_einet, device,
																			 test_x)
	evaluation_message("{} Mean LL : {}, Std LL : {}".format(AVERAGE, av_mean_ll_dict[1], av_std_ll_dict[1]))
	evaluation_message("{} Mean LL : {}, Std LL : {}".format(AVERAGE, av_mean_ll_dict[3], av_std_ll_dict[3]))
	evaluation_message("{} Mean LL : {}, Std LL : {}".format(AVERAGE, av_mean_ll_dict[5], av_std_ll_dict[5]))

	def attack_test_einet(dataset_name, trained_adv_einet, trained_clean_einet, train_x, test_x, test_labels,
						  perturbations, attack_type, batch_size, is_adv):
		mean_ll, std_ll, attack_test_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, train_x,
														test_x, test_labels, perturbations=perturbations, device=device,
														attack_type=attack_type, batch_size=batch_size, is_adv=is_adv)
		evaluation_message("{}-{} Mean LL : {}, Std LL : {}".format(attack_type, perturbations, mean_ll, std_ll))

		return mean_ll, std_ll, attack_test_x

	# 1. Original Test Set
	standard_mean_ll, standard_std_ll, standard_test_x = attack_test_einet(dataset_name, trained_adv_einet,
																		   trained_clean_einet, train_x, test_x,
																		   test_labels, perturbations=0,
																		   attack_type=CLEAN,
																		   batch_size=DEFAULT_EVAL_BATCH_SIZE,
																		   is_adv=False)

	# ------  LOCAL SEARCH AREA ------

	# 2. Local Search - 1 Test Set
	ls1_mean_ll, ls1_std_ll, ls1_test_x = attack_test_einet(dataset_name, trained_adv_einet,
															trained_clean_einet, train_x, test_x, test_labels,
															perturbations=1, attack_type=LOCAL_SEARCH,
															batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)

	# 3. Local Search - 3 Test Set
	ls3_mean_ll, ls3_std_ll, ls3_test_x = attack_test_einet(dataset_name, trained_adv_einet,
															trained_clean_einet, train_x, test_x, test_labels,
															perturbations=3, attack_type=LOCAL_SEARCH,
															batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)

	# 4. Local Search - 5 Test Set
	ls5_mean_ll, ls5_std_ll, ls5_test_x = attack_test_einet(dataset_name, trained_adv_einet,
															trained_clean_einet, train_x, test_x, test_labels,
															perturbations=5, attack_type=LOCAL_SEARCH,
															batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)

	# ------  RESTRICTED LOCAL SEARCH AREA ------

	# 5. Restricted Local Search - 1 Test Set
	rls1_mean_ll, rls1_std_ll, rls1_test_x = attack_test_einet(dataset_name, trained_adv_einet,
															   trained_clean_einet, train_x, test_x,
															   test_labels, perturbations=1,
															   attack_type=RESTRICTED_LOCAL_SEARCH,
															   batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)

	# 6. Restricted Local Search - 3 Test Set
	rls3_mean_ll, rls3_std_ll, rls3_test_x = attack_test_einet(dataset_name, trained_adv_einet,
															   trained_clean_einet, train_x, test_x,
															   test_labels, perturbations=3,
															   attack_type=RESTRICTED_LOCAL_SEARCH,
															   batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)

	# 7. Restricted Local Search - 5 Test Set
	rls5_mean_ll, rls5_std_ll, rls5_test_x = attack_test_einet(dataset_name, trained_adv_einet,
															   trained_clean_einet, train_x, test_x,
															   test_labels, perturbations=5,
															   attack_type=RESTRICTED_LOCAL_SEARCH,
															   batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)

	# ------------- WEAKER MODEL ATTACK --------

	# 11. Weaker model attack - 1 Test Set
	w1_mean_ll, w1_std_ll, w1_test_x = attack_test_einet(dataset_name, trained_adv_einet, trained_clean_einet,
														 train_x, test_x, test_labels, perturbations=1,
														 attack_type=WEAKER_MODEL,
														 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)

	# 12. Weaker model attack - 3 Test Set
	w3_mean_ll, w3_std_ll, w3_test_x = attack_test_einet(dataset_name, trained_adv_einet, trained_clean_einet,
														 train_x, test_x, test_labels, perturbations=3,
														 attack_type=WEAKER_MODEL,
														 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)

	# 13. Weaker model attack - 5 Test Set
	w5_mean_ll, w5_std_ll, w5_test_x = attack_test_einet(dataset_name, trained_adv_einet, trained_clean_einet,
														 train_x, test_x, test_labels, perturbations=5,
														 attack_type=WEAKER_MODEL,
														 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)

	ll_table.add_data(train_attack_type, perturbations, standard_mean_ll, ls1_mean_ll, ls3_mean_ll, ls5_mean_ll,
					  rls1_mean_ll, rls3_mean_ll, rls5_mean_ll,
					  av_mean_ll_dict[1], av_mean_ll_dict[3], av_mean_ll_dict[5],
					  w1_mean_ll, w3_mean_ll, w5_mean_ll)

	# ------------------------------------------------------------------------------------------------
	# ----------------------------- CONDITIONAL LIKELIHOOD AREA --------------------------------------
	# ------------------------------------------------------------------------------------------------

	def attack_test_conditional_einet(test_attack_type, perturbations, dataset_name, trained_adv_einet,
									  evidence_percentage, test_x, batch_size):
		mean_ll, std_ll = SPN.test_conditional_einet(test_attack_type, perturbations, dataset_name,
													 trained_adv_einet,
													 evidence_percentage, test_x, batch_size=batch_size)
		evaluation_message(
			"{}-{},  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(test_attack_type,
																				  perturbations,
																				  evidence_percentage,
																				  mean_ll, std_ll))
		dataset_distribution_evidence_results["{}-{} Mean LL".format(test_attack_type, perturbations)] = mean_ll
		dataset_distribution_evidence_results["{}-{} Std LL".format(test_attack_type, perturbations)] = std_ll

		return mean_ll, std_ll

	# ---------- AVERAGE ATTACK AREA ------

	# 8. Average attack dictionary
	av_mean_cll_dict, av_std_cll_dict = SPN.fetch_average_conditional_likelihoods_for_data(dataset_name,
																						   trained_adv_einet,
																						   device, test_x)

	for evidence_percentage in EVIDENCE_PERCENTAGES:
		cll_table = cll_tables[evidence_percentage]

		dataset_distribution_evidence_results = dict()

		# 1. Original Test Set
		standard_mean_cll, standard_std_cll = attack_test_conditional_einet(CLEAN, 0, dataset_name,
																			trained_adv_einet,
																			evidence_percentage,
																			standard_test_x,
																			batch_size=DEFAULT_EVAL_BATCH_SIZE)

		# ---------- LOCAL SEARCH AREA ------

		# 2. Local search - 1
		ls1_mean_cll, ls1_std_cll = attack_test_conditional_einet(LOCAL_SEARCH, 1, dataset_name,
																  trained_adv_einet,
																  evidence_percentage,
																  ls1_test_x,
																  batch_size=DEFAULT_EVAL_BATCH_SIZE)

		# 3. Local search - 3
		ls3_mean_cll, ls3_std_cll = attack_test_conditional_einet(LOCAL_SEARCH, 3, dataset_name,
																  trained_adv_einet,
																  evidence_percentage,
																  ls3_test_x,
																  batch_size=DEFAULT_EVAL_BATCH_SIZE)

		# 4. Local search - 5
		ls5_mean_cll, ls5_std_cll = attack_test_conditional_einet(LOCAL_SEARCH, 5, dataset_name,
																  trained_adv_einet,
																  evidence_percentage,
																  ls5_test_x,
																  batch_size=DEFAULT_EVAL_BATCH_SIZE)

		# ---------- RESTRICTED LOCAL SEARCH AREA ------

		# 5. Restricted Local search - 1
		rls1_mean_cll, rls1_std_cll = attack_test_conditional_einet(RESTRICTED_LOCAL_SEARCH, 1, dataset_name,
																	trained_adv_einet, evidence_percentage,
																	rls1_test_x,
																	batch_size=DEFAULT_EVAL_BATCH_SIZE)

		# 6. Restricted Local search - 3
		rls3_mean_cll, rls3_std_cll = attack_test_conditional_einet(RESTRICTED_LOCAL_SEARCH, 3, dataset_name,
																	trained_adv_einet, evidence_percentage,
																	rls3_test_x,
																	batch_size=DEFAULT_EVAL_BATCH_SIZE)

		# 7. Restricted Local search - 5
		rls5_mean_cll, rls5_std_cll = attack_test_conditional_einet(RESTRICTED_LOCAL_SEARCH, 5, dataset_name,
																	trained_adv_einet, evidence_percentage,
																	rls5_test_x,
																	batch_size=DEFAULT_EVAL_BATCH_SIZE)
		#
		# # ---------- WEAKER MODEL AREA ------

		# 8. Weaker model - 1
		w1_mean_cll, w1_std_cll = attack_test_conditional_einet(WEAKER_MODEL, 1, dataset_name,
																trained_adv_einet, evidence_percentage,
																w1_test_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)

		# 9. Weaker model - 3
		w3_mean_cll, w3_std_cll = attack_test_conditional_einet(WEAKER_MODEL, 3, dataset_name,
																trained_adv_einet, evidence_percentage,
																w3_test_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)

		# 10. Weaker model - 5
		w5_mean_cll, w5_std_cll = attack_test_conditional_einet(WEAKER_MODEL, 5, dataset_name,
																trained_adv_einet, evidence_percentage,
																w5_test_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)

		# -------------------------------- LOG CONDITIONALS TO WANDB TABLES ------------------------------------

		cll_table.add_data(train_attack_type, perturbations, standard_mean_cll,
						   ls1_mean_cll, ls3_mean_cll, ls5_mean_cll,
						   rls1_mean_cll, rls3_mean_cll, rls5_mean_cll,
						   av_mean_cll_dict[1][evidence_percentage], av_mean_cll_dict[3][evidence_percentage],
						   av_mean_cll_dict[5][evidence_percentage], w1_mean_cll, w3_mean_cll, w5_mean_cll)


def train_random_dro_spn_sgd(run_id, device, specific_datasets=None, train_attack_type=None, perturbations=None):
	if specific_datasets is None:
		specific_datasets = DISCRETE_DATASETS
	else:
		specific_datasets = [specific_datasets] if type(specific_datasets) is not list else specific_datasets

	for dataset_name in specific_datasets:

		dataset_wandb_tables = fetch_wandb_table(dataset_name)
		ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]
		cll_tables = dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES]

		evaluation_message("Dataset : {}".format(dataset_name))
		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name, device)

		exponential_family, exponential_family_args, structures = None, None, None
		if dataset_name in DISCRETE_DATASETS:
			if dataset_name == BINARY_MNIST:
				structures = [POON_DOMINGOS]
			else:
				structures = [BINARY_TREES]
			if dataset_name in DEBD_DATASETS:
				exponential_family = BinomialArray
			else:
				exponential_family = CategoricalArray
			exponential_family_args = SPN.generate_exponential_family_args(exponential_family, dataset_name)

		for structure in structures:

			evaluation_message("Using the structure {}".format(structure))

			structure_args = dict()
			structure_args[NUM_VAR] = train_x.shape[1]
			structure_args[DEPTH] = DEFAULT_DEPTH
			structure_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS
			graph = SPN.load_structure(run_id, structure, dataset_name, structure_args)

			einet_args = fetch_einet_args_discrete(dataset_name, train_x.shape[1], exponential_family,
												   exponential_family_args)

			trained_clean_einet = SPN.load_pretrained_einet(run_id, structure, dataset_name, einet_args, device,
															attack_type=CLEAN)

			if trained_clean_einet is None:
				evaluation_message("Loading Einet")
				clean_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph, device)
				evaluation_message("Training clean einet")
				trained_clean_einet = train_einet(dataset_name, clean_einet, train_x, valid_x, test_x,
												  einet_args[BATCH_SIZE],
												  einet_args[LEARNING_RATE], einet_args[WEIGHT_DECAY])
				SPN.save_model(run_id, trained_clean_einet, dataset_name, structure, einet_args, False, CLEAN, 0)

			trained_adv_einet = trained_clean_einet

			if perturbations == 1:
				# Test the clean einet only once
				test_trained_einet(dataset_name, trained_clean_einet, trained_clean_einet, train_x, test_x, test_labels,
								   ll_table, cll_tables, CLEAN, 0)

			for samples in [10, 30, 50]:
				perturbed_training_datasets = []
				evaluation_message("Training using samples {}".format(samples))
				for sample in range(samples):
					# 1. Generate randomly perturbed dataset
					perturbed_train_x = random_attack.generate_random_perturbed_dataset(train_x, perturbations, device)
					perturbed_training_datasets.append(perturbed_train_x)

				# Approach-1
				# Use the raw perturbed training samples in inner minimization problem
				specific_filename = "{}_{}_{}".format(dataset_name, perturbations, samples)

				print(specific_filename)
				trained_random_dro_einet = SPN.load_pretrained_einet(run_id, structure, dataset_name, einet_args,
																	 device, attack_type=train_attack_type,
																	 perturbations=perturbations,
																	 specific_filename=specific_filename)

				if trained_random_dro_einet is None:
					print("Could not find trained models")
					random_dro_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph, device)
					trained_random_dro_einet = train_random_dro_from_perturbed_datasets(dataset_name, perturbed_training_datasets, random_dro_einet, train_x, valid_x, test_x,
																						einet_args[BATCH_SIZE],
																						einet_args[LEARNING_RATE], einet_args[WEIGHT_DECAY])

				test_trained_einet(dataset_name, trained_random_dro_einet, trained_clean_einet, train_x, test_x,
								   test_labels, ll_table, cll_tables, "random_dro_samples_{}".format(samples),
								   perturbations)

				SPN.save_model(run_id, trained_adv_einet, dataset_name, structure, einet_args, True, train_attack_type,
							   perturbations)


# adv_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph, device)
#
# print("Considering adv einet")
# trained_adv_einet = SPN.load_pretrained_einet(run_id, structure, dataset_name, einet_args, device,
# 											  attack_type=train_attack_type, perturbations=perturbations)
# if trained_adv_einet is None:
# 	# Test the adversarial einet
# 	evaluation_message(
# 		"Training adversarial einet with attack type {}-{}".format(train_attack_type, perturbations))
# 	trained_adv_einet = train_einet_random_dro(dataset_name, adv_einet, perturbations, train_x, valid_x,
# 											   test_x, einet_args[BATCH_SIZE], einet_args[LEARNING_RATE],
# 											   einet_args[WEIGHT_DECAY])
# 	SPN.save_model(run_id, trained_adv_einet, dataset_name, structure, einet_args, True, train_attack_type,
# 				   perturbations)
#
# test_trained_einet(dataset_name, trained_adv_einet, trained_clean_einet, train_x, test_x, test_labels,
# 				   ll_table, cll_tables, train_attack_type, perturbations)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dataset', type=str, required=True, help="dataset name")
	parser.add_argument('--runid', type=str, required=True, help="run id")
	ARGS = parser.parse_args()
	print(ARGS)

	dataset_name = ARGS.dataset
	run_id = ARGS.runid

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	for perturbation in DRO_PERTURBATIONS:
		evaluation_message(
			"Logging values for {}, perturbation {}, train attack type {}".format(dataset_name, perturbation,
																				  WASSERSTEIN_RANDOM_SAMPLES))
		train_random_dro_spn_sgd(run_id=run_id, device=device, specific_datasets=dataset_name,
								 train_attack_type=WASSERSTEIN_RANDOM_SAMPLES, perturbations=perturbation)

	dataset_wandb_tables = fetch_wandb_table(dataset_name)
	ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]
	cll_tables = dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES]

	wandb_run.log({"{}-META-DRO-LL".format(dataset_name): ll_table})
	for evidence_percentage in EVIDENCE_PERCENTAGES:
		cll_ev_table = cll_tables[evidence_percentage]
		wandb_run.log({"{}-META_DRO-CLL-{}".format(dataset_name, evidence_percentage): cll_ev_table})
