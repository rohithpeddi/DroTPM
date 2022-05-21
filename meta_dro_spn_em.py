import copy

import numpy as np
import argparse
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import wandb
import einet_base_spn as SPN
from EinsumNetwork import EinsumNetwork
from EinsumNetwork.ExponentialFamilyArray import NormalArray, CategoricalArray, BinomialArray
from constants import *
from deeprob.torch.callbacks import EarlyStopping

# wandb_run = wandb.init(project="DROSPN", entity="utd-ml-pgm")

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

	num_distributions, online_em_frequency, batch_size = None, DEFAULT_ONLINE_EM_FREQUENCY, DEFAULT_TRAIN_BATCH_SIZE
	if dataset_name in ['plants', 'accidents', 'tretail']:
		num_distributions = 20
		batch_size = 100
		online_em_frequency = 50
	elif dataset_name in ['nltcs', 'msnbc', 'kdd', 'pumsb_star', 'kosarek', 'msweb']:
		num_distributions = 10
		batch_size = 100
		online_em_frequency = 50
	elif dataset_name in ['baudio', 'jester', 'bnetflix', 'book']:
		num_distributions = 10
		batch_size = 50
		online_em_frequency = 5
	elif dataset_name in ['tmovie', 'dna']:
		num_distributions = 20
		batch_size = 50
		online_em_frequency = 1
	elif dataset_name in ['cwebkb', 'bbc', 'cr52']:
		num_distributions = 10
		batch_size = 50
		online_em_frequency = 1
	elif dataset_name in ['c20ng']:
		num_distributions = 10
		batch_size = 50
		online_em_frequency = 5
	elif dataset_name in ['ad']:
		num_distributions = 10
		batch_size = 50
		online_em_frequency = 5
	elif dataset_name in [BINARY_MNIST, BINARY_FASHION_MNIST]:
		num_distributions = 10
		batch_size = 100
		online_em_frequency = 10

	einet_args[NUM_SUMS] = num_distributions
	einet_args[USE_EM] = True
	einet_args[NUM_CLASSES] = GENERATIVE_NUM_CLASSES
	einet_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
	einet_args[EXPONENTIAL_FAMILY] = exponential_family
	einet_args[EXPONENTIAL_FAMILY_ARGS] = exponential_family_args
	einet_args[ONLINE_EM_FREQUENCY] = online_em_frequency
	einet_args[ONLINE_EM_STEPSIZE] = DEFAULT_ONLINE_EM_STEPSIZE
	einet_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS
	einet_args[BATCH_SIZE] = batch_size

	return einet_args


def train_einet(dataset_name, einet, train_x, valid_x, test_x, batch_size=DEFAULT_TRAIN_BATCH_SIZE):
	early_stopping = EarlyStopping(einet, patience=DEFAULT_EINET_PATIENCE, filepath=EARLY_STOPPING_FILE,
								   delta=EARLY_STOPPING_DELTA)

	train_dataset = TensorDataset(train_x)
	for epoch in range(MAX_NUM_EPOCHS):
		train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
		train_dataloader = tqdm(
			train_dataloader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
			desc='Training epoch : {}, for dataset : {}'.format(epoch, dataset_name),
			unit='batch'
		)
		einet.train()
		for inputs in train_dataloader:
			outputs = einet.forward(inputs[0])
			ll_sample = EinsumNetwork.log_likelihoods(outputs)
			log_likelihood = ll_sample.sum()

			objective = log_likelihood
			objective.backward()

			einet.em_process_batch()
		einet.em_update()

		train_ll, valid_ll, test_ll = SPN.evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=epoch)
		if epoch > 1:
			early_stopping(-valid_ll, epoch)
			if early_stopping.should_stop and epoch > 5:
				print("Early Stopping... {}".format(early_stopping))
				break
	return einet


def fetch_adv_data_using_gradients(einet, train_x, batch_size, outer_epoch, perturbations):
	# 1. Copy the whole einet to a new einet object
	gradient_einet = copy.deepcopy(einet)
	meta_gradients = torch.zeros(train_x.shape, device=device)
	adv_train_x = torch.zeros(train_x.shape, device=device)
	batch_counter = 0
	gradient_einet.train()

	# 2. Attach optimizer to the model inorder to zero out the gradients
	optimizer = optim.Adam(list(gradient_einet.parameters()), lr=0.1, weight_decay=1e-4)

	# 3. Use gradients obtained from doing a loss backward in creating new adversarial training data
	train_dataset = TensorDataset(train_x)
	train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
	train_dataloader = tqdm(
		train_dataloader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Training epoch : {}, for dataset : {}'.format(outer_epoch, dataset_name), unit='batch'
	)

	for inputs in train_dataloader:
		optimizer.zero_grad()

		batch_input = inputs[0]
		adv_batch = batch_input.clone()

		batch_input.requires_grad = True
		outputs = gradient_einet.forward(batch_input)
		log_likelihood = outputs.sum()
		gradient_objective = -log_likelihood
		gradient_objective.backward()

		batch_input_grad = batch_input.grad
		meta_gradients[(batch_size * batch_counter): min(train_x.shape[0], (batch_size * (batch_counter + 1))),
		:] += batch_input_grad

		# 1. Construct new batch
		# Greedy step in choosing the places to flip based on gradients
		scored_input = torch.mul((-2 * adv_batch + 1), meta_gradients[
													   (batch_size * batch_counter): min(train_x.shape[0], (
															   batch_size * (batch_counter + 1))), :])
		num_dims = train_x.shape[1]
		(values, indices) = torch.topk(scored_input.view(1, -1), (adv_batch.shape[0]) * perturbations)
		row_change_count_dict = {}
		for index in list(indices.cpu().numpy()[-1]):
			row = index // num_dims
			column = index % num_dims
			if not row in row_change_count_dict:
				row_change_count_dict[row] = 1
			elif row_change_count_dict[row] >= 5:
				continue
			else:
				row_change_count_dict[row] += 1
			adv_batch[row, column] = 1.0 - adv_batch[row, column]

		adv_train_x[
		(batch_size * batch_counter): min(train_x.shape[0], (batch_size * (batch_counter + 1))),
		:] += adv_batch

		batch_counter = batch_counter + 1

	return adv_train_x


def train_einet_meta_dro(dataset_name, einet, perturbations, train_x, valid_x, test_x, batch_size):
	early_stopping = EarlyStopping(einet, patience=DEFAULT_EINET_PATIENCE, filepath=EARLY_STOPPING_FILE,
								   delta=EARLY_STOPPING_DELTA)

	meta_adv_epoch_count = 1
	adv_train_x = train_x
	for epoch in range(MAX_NUM_EPOCHS):
		adv_train_dataset = TensorDataset(adv_train_x)
		adv_train_dataloader = DataLoader(adv_train_dataset, batch_size, shuffle=False)
		adv_train_dataloader = tqdm(
			adv_train_dataloader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
			desc='Training epoch : {}, for dataset : {}'.format(epoch, dataset_name),
			unit='batch'
		)
		einet.train()
		for adv_inputs in adv_train_dataloader:
			adv_outputs = einet.forward(adv_inputs[0])
			adv_ll_sample = EinsumNetwork.log_likelihoods(adv_outputs)
			adv_log_likelihood = adv_ll_sample.sum()

			adv_objective = adv_log_likelihood
			adv_objective.backward()

			einet.em_process_batch()
		einet.em_update()

		if epoch % meta_adv_epoch_count == 0:
			adv_train_x = fetch_adv_data_using_gradients(einet, train_x, batch_size, epoch, perturbations)

		train_ll, valid_ll, test_ll = SPN.evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=epoch)
		if epoch > 1:
			early_stopping(-valid_ll, epoch)
			if early_stopping.should_stop and epoch > 5:
				print("Early Stopping... {}".format(early_stopping))
				break

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

	ll_table.add_data(train_attack_type, perturbations, standard_mean_ll, standard_std_ll,
					  ls1_mean_ll, ls1_std_ll, ls3_mean_ll, ls3_std_ll, ls5_mean_ll, ls5_std_ll,
					  rls1_mean_ll, rls1_std_ll, rls3_mean_ll, rls3_std_ll, rls5_mean_ll, rls5_std_ll,
					  av_mean_ll_dict[1], av_std_ll_dict[1], av_mean_ll_dict[3], av_std_ll_dict[3],
					  av_mean_ll_dict[5], av_std_ll_dict[5],
					  w1_mean_ll, w1_std_ll, w3_mean_ll, w3_std_ll, w5_mean_ll, w5_std_ll)

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

		cll_table.add_data(train_attack_type, perturbations, standard_mean_cll, standard_std_cll,
						   ls1_mean_cll, ls1_std_cll, ls3_mean_cll, ls3_std_cll, ls5_mean_cll, ls5_std_cll,
						   rls1_mean_cll, rls1_std_cll, rls3_mean_cll, rls3_std_cll, rls5_mean_cll,
						   rls5_std_cll,
						   av_mean_cll_dict[1][evidence_percentage], av_std_cll_dict[1][evidence_percentage],
						   av_mean_cll_dict[3][evidence_percentage], av_std_cll_dict[3][evidence_percentage],
						   av_mean_cll_dict[5][evidence_percentage], av_std_cll_dict[5][evidence_percentage],
						   w1_mean_cll, w1_std_cll, w3_mean_cll, w3_std_cll, w5_mean_cll, w5_std_cll)


def train_meta_dro_spn(run_id, device, specific_datasets=None, train_attack_type=None, perturbations=None):
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
												  einet_args[BATCH_SIZE])
				SPN.save_model(run_id, trained_clean_einet, dataset_name, structure, einet_args, False, CLEAN, 0)

			trained_adv_einet = trained_clean_einet

			if perturbations == 1:
				# Test the clean einet only once
				test_trained_einet(dataset_name, trained_clean_einet, trained_clean_einet, train_x, test_x, test_labels,
								   ll_table, cll_tables, train_attack_type, perturbations)

			adv_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph, device)

			print("Considering adv einet")
			trained_adv_einet = SPN.load_pretrained_einet(run_id, structure, dataset_name, einet_args, device,
														  attack_type=WASSERSTEIN_META, perturbations=perturbations)
			if trained_adv_einet is None:
				# Test the adversarial einet
				evaluation_message("Training adversarial einet with attack type {}".format(train_attack_type))
				trained_adv_einet = train_einet_meta_dro(dataset_name, adv_einet, perturbations, train_x, valid_x,
														 test_x, einet_args[BATCH_SIZE])
				SPN.save_model(run_id, trained_adv_einet, dataset_name, structure, einet_args, True, WASSERSTEIN_META,
							   perturbations)

			test_trained_einet(dataset_name, trained_adv_einet, trained_clean_einet, train_x, test_x, test_labels,
							   ll_table, cll_tables, train_attack_type, perturbations)


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
																				  WASSERSTEIN_META))
		train_meta_dro_spn(run_id=run_id, device=device, specific_datasets=dataset_name,
						   train_attack_type=WASSERSTEIN_META, perturbations=perturbation)

	dataset_wandb_tables = fetch_wandb_table(dataset_name)
	ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]
	cll_tables = dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES]

# wandb_run.log({"{}-META-DRO-EM-LL".format(dataset_name): ll_table})
# for evidence_percentage in EVIDENCE_PERCENTAGES:
# 	cll_ev_table = cll_tables[evidence_percentage]
# 	wandb_run.log({"{}-META_DRO-EM-CLL-{}".format(dataset_name, evidence_percentage): cll_ev_table})
