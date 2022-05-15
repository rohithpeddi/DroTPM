# import wandb
# import cn_base as CN
# from constants import *
# import argparse
#
# from deeprob.torch.callbacks import EarlyStopping
#
# wandb_run = wandb.init(project="DROCN", entity="utd-ml-pgm")
#
# columns = ["attack_type", "perturbations", "standard_mean_ll", "standard_std_ll",
# 		   "ls1_mean_ll", "ls1_std_ll", "ls3_mean_ll", "ls3_std_ll", "ls5_mean_ll", "ls5_std_ll",
# 		   "rls1_mean_ll", "rls1_std_ll", "rls3_mean_ll", "rls3_std_ll", "rls5_mean_ll", "rls5_std_ll",
# 		   "av1_mean_ll", "av1_std_ll", "av3_mean_ll", "av3_std_ll", "av5_mean_ll", "av5_std_ll",
# 		   "w1_mean_ll", "w1_std_ll", "w3_mean_ll", "w3_std_ll", "w5_mean_ll", "w5_std_ll"]
#
# wandb_tables = dict()
#
#
# def fetch_wandb_table(dataset_name):
# 	if dataset_name not in wandb_tables:
# 		dataset_wandb_tables = dict()
# 		ll_table = wandb.Table(columns=columns)
# 		dataset_wandb_tables[LOGLIKELIHOOD_TABLE] = ll_table
# 		wandb_tables[dataset_name] = dataset_wandb_tables
# 	return wandb_tables[dataset_name]
#
#
# def evaluation_message(message):
# 	print("\n")
# 	print("-----------------------------------------------------------------------------")
# 	print("#" + message)
# 	print("-----------------------------------------------------------------------------")
#
#
# # ------------------------------------------------------------------------------------------
# # ------  TESTING AREA ------
# def test_trained_einet(dataset_name, trained_adv_cnet, trained_clean_cnet, train_x, test_x, perturbations):
#
# 	av_mean_ll_dict, av_std_ll_dict = CN.fetch_average_likelihoods_for_data(dataset_name, trained_adv_cnet, test_x)
#
# 	def attack_test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet, train_x, test_x,
# 						 perturbations, attack_type, is_adv):
# 		mean_ll, std_ll, attack_test_x = CN.test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet,
# 													  train_x, test_x, perturbations=perturbations,
# 													  attack_type=attack_type, is_adv=is_adv)
# 		evaluation_message("{}-{} Mean LL : {}, Std LL : {}".format(attack_type, perturbations, mean_ll, std_ll))
# 		return mean_ll, std_ll, attack_test_x
#
# 	# 1. Original Test Set
# 	standard_mean_ll, standard_std_ll, standard_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
# 																		  trained_clean_cnet, train_x, test_x,
# 																		  perturbations=0,
# 																		  attack_type=CLEAN, is_adv=False)
#
# 	# ------  LOCAL SEARCH AREA ------
#
# 	# 2. Local Search - 1 Test Set
# 	ls1_mean_ll, ls1_std_ll, ls1_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
# 														   trained_clean_cnet, train_x, test_x,
# 														   perturbations=1, attack_type=LOCAL_SEARCH,
# 														   is_adv=True)
#
# 	# 3. Local Search - 3 Test Set
# 	ls3_mean_ll, ls3_std_ll, ls3_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
# 														   trained_clean_cnet, train_x, test_x,
# 														   perturbations=3, attack_type=LOCAL_SEARCH,
# 														   is_adv=True)
#
# 	# 4. Local Search - 5 Test Set
# 	ls5_mean_ll, ls5_std_ll, ls5_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
# 														   trained_clean_cnet, train_x, test_x,
# 														   perturbations=5, attack_type=LOCAL_SEARCH,
# 														   is_adv=True)
#
# 	# ------  RESTRICTED LOCAL SEARCH AREA ------
#
# 	# 5. Restricted Local Search - 1 Test Set
# 	rls1_mean_ll, rls1_std_ll, rls1_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
# 															  trained_clean_cnet, train_x, test_x,
# 															  perturbations=1,
# 															  attack_type=RESTRICTED_LOCAL_SEARCH, is_adv=True)
#
# 	# 6. Restricted Local Search - 3 Test Set
# 	rls3_mean_ll, rls3_std_ll, rls3_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
# 															  trained_clean_cnet, train_x, test_x,
# 															  perturbations=3,
# 															  attack_type=RESTRICTED_LOCAL_SEARCH, is_adv=True)
#
# 	# 7. Restricted Local Search - 5 Test Set
# 	rls5_mean_ll, rls5_std_ll, rls5_test_x = attack_test_cnet(dataset_name, trained_adv_cnet,
# 															  trained_clean_cnet, train_x, test_x,
# 															  perturbations=5,
# 															  attack_type=RESTRICTED_LOCAL_SEARCH, is_adv=True)
#
# 	# ------------- WEAKER MODEL ATTACK --------
#
# 	# 11. Weaker model attack - 1 Test Set
# 	w1_mean_ll, w1_std_ll, w1_test_x = attack_test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet,
# 														train_x, test_x, perturbations=1,
# 														attack_type=WEAKER_MODEL, is_adv=True)
#
# 	# 12. Weaker model attack - 3 Test Set
# 	w3_mean_ll, w3_std_ll, w3_test_x = attack_test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet,
# 														train_x, test_x, perturbations=3,
# 														attack_type=WEAKER_MODEL, is_adv=True)
#
# 	# 13. Weaker model attack - 5 Test Set
# 	w5_mean_ll, w5_std_ll, w5_test_x = attack_test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet,
# 														train_x, test_x, perturbations=5,
# 														attack_type=WEAKER_MODEL, is_adv=True)
#
# 	# -------------------------------- LOG LIKELIHOOD TO WANDB TABLES ------------------------------------
#
# 	ll_table.add_data(train_attack_type, perturbations, standard_mean_ll, standard_std_ll,
# 					  ls1_mean_ll, ls1_std_ll, ls3_mean_ll, ls3_std_ll, ls5_mean_ll, ls5_std_ll,
# 					  rls1_mean_ll, rls1_std_ll, rls3_mean_ll, rls3_std_ll, rls5_mean_ll, rls5_std_ll,
# 					  av_mean_ll_dict[1], av_std_ll_dict[1], av_mean_ll_dict[3], av_std_ll_dict[3],
# 					  av_mean_ll_dict[5], av_std_ll_dict[5],
# 					  w1_mean_ll, w1_std_ll, w3_mean_ll, w3_std_ll, w5_mean_ll, w5_std_ll)
#
#
# def train_random_dro_from_perturbed_datasets(run_id, dataset_name, perturbed_training_datasets, einet, train_x, valid_x,
# 											 test_x, einet_args):
# 	patience = DEFAULT_EINET_PATIENCE
# 	early_stopping = EarlyStopping(einet, patience=patience, filepath=EARLY_STOPPING_FILE, delta=EARLY_STOPPING_DELTA)
#
#
# 	for epoch_count in range(MAX_NUM_EPOCHS):
# 		SPN.epoch_einet_train(train_dataloader, einet, epoch_count, dataset_name, weight=1)
# 		train_ll, valid_ll, test_ll = SPN.evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=epoch_count)
# 		if epoch_count > 1:
# 			early_stopping(-valid_ll, epoch_count)
# 			if early_stopping.should_stop and epoch_count > 5:
# 				print("Early Stopping... {}".format(early_stopping))
# 				break
# 		print("Fetching adversarial data, training epoch {}".format(epoch_count))
#
# 		einet.eval()
# 		perturbed_likelihoods = []
# 		for i in range(len(perturbed_training_datasets)):
# 			perturbed_dataset = perturbed_training_datasets[i]
# 			perturbed_data_likelihood = EinsumNetwork.eval_loglikelihood_batched(einet, perturbed_dataset,
# 																				 batch_size=EVAL_BATCH_SIZE) / \
# 										train_x.shape[0]
# 			perturbed_likelihoods.append(perturbed_data_likelihood)
# 		min_likelihood_dataset_idx = torch.argmin(torch.tensor(perturbed_likelihoods))
# 		print("Minimum Likelihood is observed at idx : {}".format(min_likelihood_dataset_idx))
# 		train_dataset = TensorDataset(perturbed_training_datasets[min_likelihood_dataset_idx])
# 	return einet
#
#
# def random_dro_spn(run_id, specific_datasets=None, perturbations=None, device=None):
# 	if specific_datasets is None:
# 		specific_datasets = DISCRETE_DATASETS
# 	else:
# 		specific_datasets = [specific_datasets] if type(specific_datasets) is not list else specific_datasets
#
# 	for dataset_name in specific_datasets:
#
# 		dataset_wandb_tables = fetch_wandb_table(dataset_name)
# 		ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]
# 		cll_tables = dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES]
#
# 		evaluation_message("Dataset : {}".format(dataset_name))
# 		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name, device)
# 		exponential_family, exponential_family_args, structures = None, None, None
# 		if dataset_name in DISCRETE_DATASETS:
# 			if dataset_name == BINARY_MNIST:
# 				structures = [POON_DOMINGOS]
# 			else:
# 				structures = [BINARY_TREES]
# 			exponential_family = CategoricalArray
# 			exponential_family_args = SPN.generate_exponential_family_args(exponential_family, dataset_name)
#
# 		for structure in structures:
# 			evaluation_message("Using the structure {}".format(structure))
# 			graph = None
# 			if structure == POON_DOMINGOS:
# 				structure_args = dict()
# 				structure_args[HEIGHT] = MNIST_HEIGHT
# 				structure_args[WIDTH] = MNIST_WIDTH
# 				structure_args[PD_NUM_PIECES] = DEFAULT_PD_NUM_PIECES
# 				graph = SPN.load_structure(run_id, structure, dataset_name, structure_args)
# 			else:
# 				structure_args = dict()
# 				structure_args[NUM_VAR] = train_x.shape[1]
# 				structure_args[DEPTH] = DEFAULT_DEPTH
# 				structure_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS
# 				graph = SPN.load_structure(run_id, structure, dataset_name, structure_args)
# 			einet_args = fetch_einet_args_discrete(dataset_name, train_x.shape[1], exponential_family,
# 												   exponential_family_args)
#
# 			evaluation_message("Loading Clean Einet")
# 			trained_clean_einet = SPN.load_pretrained_einet(run_id, structure, dataset_name, einet_args, device)
#
# 			# Train a clean EiNET on original training data
# 			if trained_clean_einet is None:
# 				clean_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph, device)
# 				evaluation_message("Training clean einet")
# 				trained_clean_einet = SPN.train_einet(run_id, structure, dataset_name, clean_einet, train_labels,
# 													  train_x, valid_x, test_x, einet_args, perturbations, device,
# 													  CLEAN, batch_size=einet_args[BATCH_SIZE], is_adv=False)
# 			if perturbations == 1:
# 				test_trained_einet(dataset_name, trained_clean_einet, trained_clean_einet, train_x, test_x, test_labels,
# 								   ll_table, cll_tables, "clean", 0)
#
# 			for samples in [10, 30, 50, 100]:
# 				perturbed_training_datasets = []
# 				evaluation_message("Training using samples {}".format(samples))
# 				for sample in range(samples):
# 					# 1. Generate randomly perturbed dataset
# 					perturbed_train_x = random_attack.generate_random_perturbed_dataset(train_x, perturbations, device)
# 					perturbed_training_datasets.append(perturbed_train_x)
#
# 				# Approach-1
# 				# Use the raw perturbed training samples in inner minimization problem
# 				specific_filename = "{}_{}_{}".format(dataset_name, perturbations, samples)
#
# 				print(specific_filename)
# 				trained_random_dro_einet = SPN.load_pretrained_einet(run_id, structure, dataset_name, einet_args,
# 																	 device, attack_type=WASSERSTEIN_RANDOM_SAMPLES,
# 																	 perturbations=perturbations,
# 																	 specific_filename=specific_filename)
#
# 				if trained_random_dro_einet is None:
# 					print("Could not find trained models")
# 					random_dro_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph, device)
# 					trained_random_dro_einet = train_random_dro_from_perturbed_datasets(run_id, dataset_name,
# 																						perturbed_training_datasets,
# 																						random_dro_einet, train_x,
# 																						valid_x, test_x, einet_args)
#
# 				test_trained_einet(dataset_name, trained_random_dro_einet, trained_clean_einet, train_x, test_x,
# 								   test_labels, ll_table, cll_tables, "random_dro_samples_{}".format(samples),
# 								   perturbations)
#
# 				SPN.save_model(run_id, trained_random_dro_einet, dataset_name, structure, einet_args, True,
# 							   WASSERSTEIN_RANDOM_SAMPLES,
# 							   perturbations, specific_filename=specific_filename)
#
#
# if __name__ == '__main__':
#
# 	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 	parser.add_argument('--dataset', type=str, required=True, help="dataset name")
# 	parser.add_argument('--runid', type=str, required=True, help="run id")
# 	parser.add_argument('--lr', type=str, required=True, help="learning rate")
#
# 	ARGS = parser.parse_args()
# 	print(ARGS)
#
# 	dataset_name = ARGS.dataset
# 	run_id = ARGS.runid
# 	lr = float(ARGS.lr)
#
# 	# for dataset_name in DEBD_DATASETS:
# 	for perturbation in PERTURBATIONS:
# 		if perturbation == 0:
# 			continue
# 			TRAIN_ATTACKS = [CLEAN]
# 		else:
# 			TRAIN_ATTACKS = [AMBIGUITY_SET_UNIFORM]
# 		# TRAIN_ATTACKS = [LOCAL_SEARCH, RESTRICTED_LOCAL_SEARCH]
#
# 		for train_attack_type in TRAIN_ATTACKS:
# 			evaluation_message(
# 				"Logging values for {}, perturbation {}, train attack type {}".format(dataset_name, perturbation,
# 																					  train_attack_type))
# 			if perturbation != 0:
# 				test_cnet(run_id=1451, specific_datasets=dataset_name, is_adv=True,
# 						  train_attack_type=train_attack_type, perturbations=perturbation, learning_rate=lr)
# 			elif perturbation == 0:
# 				test_cnet(run_id=1451, specific_datasets=dataset_name, is_adv=False,
# 						  train_attack_type=train_attack_type, perturbations=perturbation, learning_rate=lr)
#
# 	dataset_wandb_tables = fetch_wandb_table(dataset_name)
# 	ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]
#
# # wandb_run.log({"{}-{}-RANDOM-DRO-LL".format(dataset_name, lr): ll_table})
