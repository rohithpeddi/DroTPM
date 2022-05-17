import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import datasets
from EinsumNetworkSGD import EinsumNetwork, Graph
from EinsumNetworkSGD.ExponentialFamilyArray import NormalArray, CategoricalArray, BinomialArray
from attacks.SPN.localrestrictedsearch import attack as local_restricted_search_attack
from attacks.SPN.localsearch import attack as local_search_attack
from attacks.SPN.weakermodel import attack as weaker_attack
from attacks.SPN.amuniform import attack as am_uniform
from constants import *
from deeprob.torch.callbacks import EarlyStopping
from utils import mkdir_p
from utils import predict_labels_mnist


def generate_exponential_family_args(exponential_family, dataset_name):
	exponential_family_args = None
	if exponential_family == BinomialArray:
		if dataset_name in DEBD_DATASETS:
			exponential_family_args = {'N': 1}
		else:
			exponential_family_args = {'N': 255}
	elif exponential_family == CategoricalArray:
		if dataset_name == BINARY_MNIST or dataset_name in DEBD_DATASETS:
			exponential_family_args = {'K': 2}
		else:
			exponential_family_args = {'K': 256}
	elif exponential_family == NormalArray:
		exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}
	return exponential_family_args


def load_dataset(dataset_name, device):
	train_x, train_labels, test_x, test_labels, valid_x, valid_labels = None, None, None, None, None, None
	if dataset_name == FASHION_MNIST:
		train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()
	elif dataset_name == MNIST:
		train_x, train_labels, test_x, test_labels = datasets.load_mnist()
		train_x /= 255.
		test_x /= 255.
		train_x -= 0.1307
		test_x -= 0.1307
		train_x /= 0.3081
		test_x /= 0.3081
		valid_x = train_x[-10000:, :]
		train_x = train_x[:-10000, :]
		valid_labels = train_labels[-10000:]
		train_labels = train_labels[:-10000]

		train_x = torch.from_numpy(train_x).to(torch.device(device))
		valid_x = torch.from_numpy(valid_x).to(torch.device(device))
		test_x = torch.from_numpy(test_x).to(torch.device(device))

		train_labels = ((torch.from_numpy(train_labels)).type(torch.int64)).to(torch.device(device))
		valid_labels = ((torch.from_numpy(valid_labels)).type(torch.int64)).to(torch.device(device))
		test_labels = ((torch.from_numpy(test_labels)).type(torch.int64)).to(torch.device(device))

	elif dataset_name == BINARY_MNIST:
		train_x, valid_x, test_x = datasets.load_binarized_mnist_dataset()
		train_labels = predict_labels_mnist(train_x)
		valid_labels = predict_labels_mnist(valid_x)
		test_labels = predict_labels_mnist(test_x)

		train_x = torch.from_numpy(train_x).to(torch.device(device))
		valid_x = torch.from_numpy(valid_x).to(torch.device(device))
		test_x = torch.from_numpy(test_x).to(torch.device(device))

		train_labels = torch.from_numpy(train_labels.reshape(-1, 1)).to(torch.device(device))
		valid_labels = torch.from_numpy(valid_labels.reshape(-1, 1)).to(torch.device(device))
		test_labels = torch.from_numpy(test_labels.reshape(-1, 1)).to(torch.device(device))

	elif dataset_name in DEBD_DATASETS:
		train_x, test_x, valid_x = datasets.load_debd(dataset_name)
		train_labels, valid_labels, test_labels = None, None, None

		train_x = torch.tensor(train_x, dtype=torch.float32, device=torch.device(device))
		valid_x = torch.tensor(valid_x, dtype=torch.float32, device=torch.device(device))
		test_x = torch.tensor(test_x, dtype=torch.float32, device=torch.device(device))

	return train_x, valid_x, test_x, train_labels, valid_labels, test_labels


def load_structure(run_id, structure, dataset_name, structure_args):
	RUN_STRUCTURE_DIRECTORY = os.path.join("run_{}".format(run_id), STRUCTURE_DIRECTORY)
	# RUN_STRUCTURE_DIRECTORY = os.path.join("", STRUCTURE_DIRECTORY)
	mkdir_p(RUN_STRUCTURE_DIRECTORY)
	graph = None
	if structure == POON_DOMINGOS:
		height = structure_args[HEIGHT]
		width = structure_args[WIDTH]
		pd_num_pieces = structure_args[PD_NUM_PIECES]

		file_name = os.path.join(RUN_STRUCTURE_DIRECTORY, "_".join([structure, dataset_name]) + ".pc")
		if os.path.exists(file_name):
			graph = Graph.read_gpickle(file_name)
		else:
			pd_delta = [[height / d, width / d] for d in pd_num_pieces]
			graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
			Graph.write_gpickle(graph, file_name)
	else:
		# Structure - Binary Trees
		num_var = structure_args[NUM_VAR]
		depth = structure_args[DEPTH]
		num_repetitions = structure_args[NUM_REPETITIONS]

		mkdir_p(STRUCTURE_DIRECTORY)
		file_name = os.path.join(RUN_STRUCTURE_DIRECTORY,
								 "{}_{}_{}.pc".format(structure, dataset_name, num_repetitions))
		if os.path.exists(file_name):
			graph = Graph.read_gpickle(file_name)
		else:
			graph = Graph.random_binary_trees(num_var=num_var, depth=depth, num_repetitions=num_repetitions)
			Graph.write_gpickle(graph, file_name)
	return graph


def load_einet(run_id, structure, dataset_name, einet_args, graph, device):
	args = EinsumNetwork.Args(
		num_var=einet_args[NUM_VAR],
		num_dims=1,
		num_classes=einet_args[NUM_CLASSES],
		num_sums=einet_args[NUM_SUMS],
		num_input_distributions=einet_args[NUM_INPUT_DISTRIBUTIONS],
		exponential_family=einet_args[EXPONENTIAL_FAMILY],
		exponential_family_args=einet_args[EXPONENTIAL_FAMILY_ARGS])
	einet = EinsumNetwork.EinsumNetwork(graph, args)
	einet.initialize()
	einet.to(device)
	return einet


def load_pretrained_einet(run_id, structure, dataset_name, einet_args, device, attack_type=None, perturbations=None,
						  specific_filename=None):
	einet = None

	if attack_type is None or attack_type == CLEAN:
		RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), CLEAN_EINET_MODEL_DIRECTORY)

		mkdir_p(RUN_MODEL_DIRECTORY)

		file_name = os.path.join(RUN_MODEL_DIRECTORY,
								 "{}_{}_{}_{}_{}.mdl".format(structure, dataset_name, einet_args[NUM_SUMS],
															 einet_args[NUM_INPUT_DISTRIBUTIONS],
															 einet_args[NUM_REPETITIONS]))

	else:
		if attack_type == NEURAL_NET:
			RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id),
											   NN_EINET_MODEL_DIRECTORY + "/{}".format(perturbations))
		elif attack_type == LOCAL_SEARCH:
			RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id),
											   LS_EINET_MODEL_DIRECTORY + "/{}".format(perturbations))
		elif attack_type == RESTRICTED_LOCAL_SEARCH:
			RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id),
											   RLS_EINET_MODEL_DIRECTORY + "/{}".format(perturbations))
		elif attack_type == AMBIGUITY_SET_UNIFORM:
			RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id),
											   AMU_EINET_MODEL_DIRECTORY + "/{}".format(perturbations))
		elif attack_type == WASSERSTEIN_META:
			RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id),
											   WASSERSTEIN_META_EINET_MODEL_DIRECTORY + "/{}".format(perturbations))
		elif attack_type == WASSERSTEIN_RANDOM_SAMPLES:
			RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id),
											   WASSERSTEIN_SAMPLES_EINET_MODEL_DIRECTORY + "/{}".format(perturbations))

		mkdir_p(RUN_MODEL_DIRECTORY)

		if attack_type == WASSERSTEIN_RANDOM_SAMPLES:
			file_name = os.path.join(RUN_MODEL_DIRECTORY,
									 "{}_{}_{}_{}_{}_adv.mdl".format(structure, specific_filename, einet_args[NUM_SUMS],
																	 einet_args[NUM_INPUT_DISTRIBUTIONS],
																	 einet_args[NUM_REPETITIONS]))
		else:
			file_name = os.path.join(RUN_MODEL_DIRECTORY,
									 "{}_{}_{}_{}_{}_adv.mdl".format(structure, dataset_name, einet_args[NUM_SUMS],
																	 einet_args[NUM_INPUT_DISTRIBUTIONS],
																	 einet_args[NUM_REPETITIONS]))

	if os.path.exists(file_name):
		einet = torch.load(file_name).to(device)
		return einet
	else:
		AssertionError("Einet for the corresponding structure is not stored, train first")
		return None


# def epoch_einet_train(train_dataloader, einet, epoch, dataset_name, weight=1, optimizer=None):
# 	train_dataloader = tqdm(
# 		train_dataloader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
# 		desc='Training epoch : {}, for dataset : {}'.format(epoch, dataset_name),
# 		unit='batch'
# 	)
# 	einet.train()
# 	for inputs in train_dataloader:
# 		optimizer.zero_grad()
# 		outputs = einet.forward(inputs[0])
# 		ll_sample = weight * EinsumNetwork.log_likelihoods(outputs)
# 		log_likelihood = ll_sample.sum()
#
# 		objective = -log_likelihood
# 		objective.backward()
# 		optimizer.step()


def evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=0):
	# Evaluate
	einet.eval()
	train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=EVAL_BATCH_SIZE)
	valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=EVAL_BATCH_SIZE)
	test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=EVAL_BATCH_SIZE)
	print("[{}] train LL {} valid LL {} test LL {}".format(epoch_count, train_ll / train_x.shape[0],
														   valid_ll / valid_x.shape[0], test_ll / test_x.shape[0]))
	return train_ll / train_x.shape[0], valid_ll / valid_x.shape[0], test_ll / test_x.shape[0]


def save_model(run_id, einet, dataset_name, structure, einet_args, is_adv, attack_type=CLEAN, perturbations=None,
			   specific_filename=None):
	RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), EINET_MODEL_DIRECTORY)

	if attack_type in [AMBIGUITY_SET_UNIFORM, NEURAL_NET, LOCAL_SEARCH, RESTRICTED_LOCAL_SEARCH,
					   WASSERSTEIN_RANDOM_SAMPLES, WASSERSTEIN_META]:
		if attack_type == NEURAL_NET:
			sub_directory_name = NEURAL_NETWORK_ATTACK_MODEL_SUB_DIRECTORY
		elif attack_type == LOCAL_SEARCH:
			sub_directory_name = LOCAL_SEARCH_ATTACK_MODEL_SUB_DIRECTORY
		elif attack_type == RESTRICTED_LOCAL_SEARCH:
			sub_directory_name = LOCAL_RESTRICTED_SEARCH_ATTACK_MODEL_SUB_DIRECTORY
		elif attack_type == AMBIGUITY_SET_UNIFORM:
			sub_directory_name = AMBIGUITY_SET_UNIFORM_ATTACK_MODEL_SUB_DIRECTORY
		elif attack_type == WASSERSTEIN_RANDOM_SAMPLES:
			sub_directory_name = WASSERSTEIN_RANDOM_SAMPLES_ATTACK_MODEL_SUB_DIRECTORY
		elif attack_type == WASSERSTEIN_META:
			sub_directory_name = WASSERSTEIN_META_ATTACK_MODEL_SUB_DIRECTORY

		ATTACK_SUB_DIRECTORY = os.path.join(RUN_MODEL_DIRECTORY, sub_directory_name)
		ATTACK_MODEL_DIRECTORY = os.path.join(ATTACK_SUB_DIRECTORY, "{}".format(perturbations))
	else:
		ATTACK_MODEL_DIRECTORY = os.path.join(RUN_MODEL_DIRECTORY, CLEAN_MODEL_SUB_DIRECTORY)

	mkdir_p(ATTACK_MODEL_DIRECTORY)

	file_name = None
	if is_adv:
		if attack_type == WASSERSTEIN_RANDOM_SAMPLES:
			file_name = os.path.join(ATTACK_MODEL_DIRECTORY,
									 "{}_{}_{}_{}_{}_adv.mdl".format(structure, specific_filename,
																	 einet_args[NUM_SUMS],
																	 einet_args[NUM_INPUT_DISTRIBUTIONS],
																	 einet_args[NUM_REPETITIONS]))
		else:
			file_name = os.path.join(ATTACK_MODEL_DIRECTORY,
									 "{}_{}_{}_{}_{}_adv.mdl".format(structure, dataset_name,
																	 einet_args[NUM_SUMS],
																	 einet_args[NUM_INPUT_DISTRIBUTIONS],
																	 einet_args[NUM_REPETITIONS]))
	else:
		file_name = os.path.join(ATTACK_MODEL_DIRECTORY,
								 "{}_{}_{}_{}_{}.mdl".format(structure, dataset_name, einet_args[NUM_SUMS],
															 einet_args[NUM_INPUT_DISTRIBUTIONS],
															 einet_args[NUM_REPETITIONS]))

	torch.save(einet, file_name)
	return


def train_einet_meta_dro(run_id, structure, dataset_name, einet, train_labels, train_x, valid_x, test_x, einet_args,
						 perturbations, device, attack_type=CLEAN, batch_size=DEFAULT_TRAIN_BATCH_SIZE, is_adv=False):
	early_stopping = EarlyStopping(einet, patience=DEFAULT_EINET_PATIENCE, filepath=EARLY_STOPPING_FILE,
								   delta=EARLY_STOPPING_DELTA)
	optimizer = optim.Adam(list(einet.parameters()), lr=einet_args[BATCH_SIZE], weight_decay=einet_args[WEIGHT_DECAY])

	train_dataset = TensorDataset(train_x)
	train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
	einet.train()

	meta_adv_epoch_count = 1
	meta_gradients = torch.zeros(train_x.shape, device=device)
	for epoch_count in range(1, MAX_NUM_EPOCHS):
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
			batch_input.requires_grad = True
			outputs = einet.forward(batch_input)
			log_likelihood = outputs.sum()
			objective = -log_likelihood
			objective.backward()

			batch_input_grad = batch_input.grad
			meta_gradients[(batch_size * batch_counter): min(train_x.shape[0], (batch_size * (batch_counter + 1))), :] += batch_input_grad

			if epoch_count % meta_adv_epoch_count == 0:
				# 1. Construct new batch
				# Greedy step in choosing the places to flip based on gradients
				adv_batch = batch_input.detach().clone()
				scored_input = torch.mul((-2*adv_batch+1), meta_gradients[(batch_size * batch_counter): min(train_x.shape[0], (batch_size * (batch_counter + 1))), :])
				num_dims = train_x.shape[1]
				(values, indices) = torch.topk(scored_input.view(1, -1), (adv_batch.shape[0])*perturbations)
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

				# # Evaluation check
				# einet.eval()
				# original_log_likelihood = EinsumNetwork.eval_loglikelihood_batched(einet, batch_input, batch_size=EVAL_BATCH_SIZE)
				# adv_log_likelihood = EinsumNetwork.eval_loglikelihood_batched(einet, adv_batch, batch_size=EVAL_BATCH_SIZE)
				# print("Decreased log score on adv_batch {} ".format(original_log_likelihood-adv_log_likelihood))

				# 2. Forward new batch
				# 3. Define objective
				# 4. Perform optimizer step
				einet.train()
				optimizer.zero_grad()
				adv_outputs = einet.forward(adv_batch)
				adv_log_likelihood = adv_outputs.sum()
				adv_objective = -adv_log_likelihood
				adv_objective.backward()

			optimizer.step()
			batch_counter = batch_counter + 1

		if epoch_count % meta_adv_epoch_count == 0:
			meta_gradients = torch.zeros(train_x.shape, device=device)

		train_ll, valid_ll, test_ll = evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=epoch_count)
		if epoch_count > 1:
			early_stopping(-valid_ll, epoch_count)
			if early_stopping.should_stop and epoch_count > 5:
				print("Early Stopping... {}".format(early_stopping))
				break

	einet.load_state_dict(early_stopping.get_best_state())
	train_ll, valid_ll, test_ll = evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=epoch_count)
	save_model(run_id, einet, dataset_name, structure, einet_args, is_adv, attack_type, perturbations)
	return einet


# def train_einet(run_id, structure, dataset_name, einet, train_labels, train_x, valid_x, test_x, einet_args,
# 						 perturbations, device, attack_type=CLEAN, batch_size=DEFAULT_TRAIN_BATCH_SIZE, is_adv=False):
# 	early_stopping = EarlyStopping(einet, patience=DEFAULT_EINET_PATIENCE, filepath=EARLY_STOPPING_FILE,
# 								   delta=EARLY_STOPPING_DELTA)
# 	optimizer = optim.Adam(list(einet.parameters()), lr=einet_args[BATCH_SIZE], weight_decay=einet_args[WEIGHT_DECAY])
# 	train_dataset = TensorDataset(train_x)
# 	for epoch_count in range(MAX_NUM_EPOCHS):
# 		train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
# 		train_dataloader = tqdm(
# 			train_dataloader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
# 			desc='Training epoch : {}, for dataset : {}'.format(epoch_count, dataset_name),
# 			unit='batch'
# 		)
# 		einet.train()
# 		for inputs in train_dataloader:
# 			optimizer.zero_grad()
# 			outputs = einet.forward(inputs[0])
# 			log_likelihood = outputs.sum()
# 			objective = -log_likelihood
# 			objective.backward()
# 			optimizer.step()
# 		train_ll, valid_ll, test_ll = evaluate_lls(einet, train_x, valid_x, test_x,
# 													   epoch_count=epoch_count)
# 		if epoch_count > 1:
# 			early_stopping(-valid_ll, epoch_count)
# 			if early_stopping.should_stop:
# 				print("Early Stopping... {}".format(early_stopping))
# 				break
# 	return einet


def fetch_attack_method(attack_type):
	if attack_type == RESTRICTED_LOCAL_SEARCH:
		return local_restricted_search_attack
	elif attack_type == LOCAL_SEARCH:
		return local_search_attack
	elif attack_type == WEAKER_MODEL:
		return weaker_attack
	elif attack_type == AMBIGUITY_SET_UNIFORM:
		return am_uniform


def fetch_adv_data(einet, dataset_name, train_data, test_data, test_labels, perturbations, attack_type, device,
				   file_name=None, combine=False):
	attack = fetch_attack_method(attack_type)
	adv_data = attack.generate_adv_dataset(einet, dataset_name, test_data, test_labels, perturbations, device,
										   combine=combine, batched=True, train_data=train_data)

	print("Fetched adversarial examples : {}/{}".format(adv_data.shape[0], test_data.shape[0]))

	adv_data = TensorDataset(adv_data)

	return adv_data


def get_stats(likelihoods):
	mean_ll = (torch.mean(likelihoods)).cpu().item()
	stddev_ll = (2.0 * torch.std(likelihoods) / np.sqrt(len(likelihoods))).cpu().item()
	return mean_ll, stddev_ll


def fetch_average_likelihoods_for_data(dataset_name, trained_einet, device, test_x,
									   average_repeat_size=DEFAULT_AVERAGE_REPEAT_SIZE):
	test_dataset = TensorDataset(test_x)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
	data_loader = tqdm(
		test_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Evaluating average neighbourhood lls for {}'.format(dataset_name),
		unit='batch'
	)

	likelihoods = dict()
	for inputs in data_loader:
		test_inputs = inputs[0].detach().clone()
		batch_size, num_dims = test_inputs.shape
		iteration_inputs = test_inputs

		num_repetitions = min(num_dims, average_repeat_size)

		for perturbation in range(1, 6):
			# Always retain the current element for comparison
			dim_idx = random.sample(range(1, num_dims + 1), num_repetitions)
			dim_idx.append(0)

			identity = torch.cat((torch.zeros(num_dims, device=torch.device(device)).reshape((1, -1)),
								  torch.eye(num_dims, device=torch.device(device))))

			# Pick (k+1) nearest neighbours and use them for search
			identity = identity[dim_idx, :]
			identity = identity.repeat((batch_size, 1))

			if perturbation == 1:
				perturbed_set = torch.repeat_interleave(iteration_inputs,
														(num_repetitions + 1) * (
															torch.ones(batch_size,
																	   device=torch.device(device)).int()),
														dim=0)
			perturbed_set = identity + perturbed_set - 2 * torch.mul(identity, perturbed_set)
			iteration_inputs = perturbed_set

			if perturbation in PERTURBATIONS:
				ll_sample = EinsumNetwork.fetch_likelihoods_for_data(trained_einet, iteration_inputs,
																	 batch_size=average_repeat_size)

				if perturbation not in likelihoods:
					likelihoods[perturbation] = []

				(likelihoods[perturbation]).append(ll_sample.mean())

	av_mean_dict = dict()
	av_std_dict = dict()

	for perturbation in [1, 3, 5]:
		lls = torch.tensor(likelihoods[perturbation])
		mean_ll, stddev_ll = get_stats(lls)

		av_mean_dict[perturbation] = mean_ll
		av_std_dict[perturbation] = stddev_ll

	return av_mean_dict, av_std_dict


def test_einet(dataset_name, trained_einet, data_einet, train_x, test_x, test_labels, perturbations, device,
			   attack_type=None, batch_size=1, is_adv=False):
	trained_einet.eval()
	if is_adv:
		test_x = fetch_adv_data(data_einet, dataset_name, train_x, test_x, test_labels, perturbations, attack_type,
								device, TEST_DATASET, combine=False).tensors[0]

	test_lls = EinsumNetwork.fetch_likelihoods_for_data(trained_einet, test_x, batch_size=batch_size)

	mean_ll, stddev_ll = get_stats(test_lls)
	return mean_ll, stddev_ll, test_x


def fetch_average_conditional_likelihoods_for_data(dataset_name, trained_einet, device, test_x,
												   average_repeat_size=DEFAULT_AVERAGE_REPEAT_SIZE):
	test_dataset = TensorDataset(test_x)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
	data_loader = tqdm(
		test_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Evaluating average neighbourhood conditional lls for {}'.format(dataset_name),
		unit='batch'
	)

	likelihoods = dict()
	for inputs in data_loader:
		test_inputs = inputs[0].detach().clone()
		batch_size, num_dims = test_inputs.shape
		iteration_inputs = test_inputs

		num_repetitions = min(num_dims, average_repeat_size)

		for perturbation in range(1, 6):
			# Always retain the current element for comparison
			dim_idx = random.sample(range(1, num_dims + 1), num_repetitions)
			dim_idx.append(0)

			identity = torch.cat((torch.zeros(num_dims, device=torch.device(device)).reshape((1, -1)),
								  torch.eye(num_dims, device=torch.device(device))))

			# Pick (k+1) nearest neighbours and use them for search
			identity = identity[dim_idx, :]
			identity = identity.repeat((batch_size, 1))

			if perturbation == 1:
				perturbed_set = torch.repeat_interleave(iteration_inputs,
														(num_repetitions + 1) * (
															torch.ones(batch_size, device=torch.device(device)).int()),
														dim=0)
			perturbed_set = identity + perturbed_set - 2 * torch.mul(identity, perturbed_set)
			iteration_inputs = perturbed_set

			if perturbation in PERTURBATIONS:
				for evidence_percentage in EVIDENCE_PERCENTAGES:
					marginalize_idx = list(np.arange(int(num_dims * evidence_percentage), num_dims))
					ll_sample = EinsumNetwork.fetch_conditional_likelihoods_for_data(trained_einet, iteration_inputs,
																					 marginalize_idx=marginalize_idx,
																					 batch_size=average_repeat_size)
					if perturbation not in likelihoods:
						likelihoods[perturbation] = dict()

					if evidence_percentage not in likelihoods[perturbation]:
						likelihoods[perturbation][evidence_percentage] = []

					(likelihoods[perturbation][evidence_percentage]).append(ll_sample.mean())

	av_mean_dict = dict()
	av_std_dict = dict()

	for perturbation in [1, 3, 5]:
		if perturbation not in av_mean_dict:
			av_mean_dict[perturbation] = dict()
			av_std_dict[perturbation] = dict()

		for evidence_percentage in EVIDENCE_PERCENTAGES:
			lls = torch.tensor(likelihoods[perturbation][evidence_percentage])
			mean_ll, stddev_ll = get_stats(lls)

			av_mean_dict[perturbation][evidence_percentage] = mean_ll
			av_std_dict[perturbation][evidence_percentage] = stddev_ll

	return av_mean_dict, av_std_dict


def test_conditional_einet(test_attack_type, perturbations, dataset_name, einet, evidence_percentage, test_x,
						   batch_size=DEFAULT_EVAL_BATCH_SIZE):
	marginalize_idx = None
	if dataset_name in DEBD_DATASETS:
		test_N, num_dims = test_x.shape
		marginalize_idx = list(np.arange(int(num_dims * evidence_percentage), num_dims))
	elif dataset_name == MNIST or dataset_name == BINARY_MNIST:
		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT * (1 - evidence_percentage)), :].reshape(-1))

	einet.eval()

	test_lls = EinsumNetwork.fetch_conditional_likelihoods_for_data(einet, test_x, marginalize_idx=marginalize_idx,
																	batch_size=batch_size)
	mean_ll, stddev_ll = get_stats(test_lls)
	return mean_ll, stddev_ll
