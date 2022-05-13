import torch
import wandb
import einet_sgd_base_spn as SPN
from EinsumNetworkSGD.ExponentialFamilyArray import NormalArray, CategoricalArray, BinomialArray
from constants import *
from utils import pretty_print_dictionary, dictionary_to_file


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
	elif dataset_name in ['nltcs', 'msnbc', 'kdd', 'pumsb_star', 'kosarek', 'msweb']:
		num_distributions = 10
		batch_size = 100
	elif dataset_name in ['baudio', 'jester', 'bnetflix', 'book']:
		num_distributions = 10
		batch_size = 50
	elif dataset_name in ['tmovie', 'dna']:
		num_distributions = 20
		batch_size = 50
	elif dataset_name in ['cwebkb', 'bbc', 'cr52']:
		num_distributions = 10
		batch_size = 50
	elif dataset_name in ['c20ng']:
		num_distributions = 10
		batch_size = 50
	elif dataset_name in ['ad']:
		num_distributions = 10
		batch_size = 50
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
	return einet_args


def test_standard_spn_discrete(run_id, device, specific_datasets=None, is_adv=False, train_attack_type=None,
							   perturbations=None, test_attack_type=None):
	if specific_datasets is None:
		specific_datasets = DISCRETE_DATASETS
	else:
		specific_datasets = [specific_datasets] if type(specific_datasets) is not list else specific_datasets

	for dataset_name in specific_datasets:

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

			evaluation_message("Loading Einet")
			trained_clean_einet = SPN.load_pretrained_einet(run_id, structure, dataset_name, einet_args, device)

			if trained_clean_einet is None:
				clean_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph, device)
				evaluation_message("Training clean einet")
				trained_clean_einet = SPN.train_einet(run_id, structure, dataset_name, clean_einet, train_labels,
													  train_x, valid_x, test_x, einet_args, perturbations, device,
													  CLEAN, batch_size=einet_args[BATCH_SIZE], is_adv=False)

			trained_adv_einet = trained_clean_einet
			adv_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph, device)

			if is_adv:
				print("Considering adv einet")
				trained_adv_einet = SPN.load_pretrained_einet(run_id, structure, dataset_name, einet_args, device,
															  train_attack_type, perturbations)
				if trained_adv_einet is None:
					evaluation_message("Training adversarial einet with attack type {}".format(train_attack_type))
					trained_adv_einet = SPN.train_einet_dro(run_id, structure, dataset_name, adv_einet, train_labels,
														train_x, valid_x, test_x, einet_args, perturbations, device,
														train_attack_type, batch_size=einet_args[BATCH_SIZE],
														is_adv=True)
				else:
					evaluation_message("Loaded pretrained einet for the configuration")

			def attack_test_einet(dataset_name, trained_adv_einet, trained_clean_einet, train_x, test_x, test_labels,
								  perturbations, attack_type, batch_size, is_adv):
				mean_ll, std_ll, attack_test_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet,
																train_x, test_x, test_labels,
																perturbations=perturbations, device=device,
																attack_type=attack_type, batch_size=batch_size,
																is_adv=is_adv)
				evaluation_message("{} Mean LL : {}, Std LL : {}".format(attack_type, mean_ll, std_ll))
				return mean_ll, std_ll, attack_test_x

			# 1. Original Test Set
			standard_mean_ll, standard_std_ll, standard_test_x = attack_test_einet(dataset_name, trained_adv_einet,
																				   trained_clean_einet, train_x, test_x,
																				   test_labels, perturbations=0,
																				   attack_type=CLEAN,
																				   batch_size=DEFAULT_EVAL_BATCH_SIZE,
																				   is_adv=False)


if __name__ == '__main__':

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	for dataset_name in DEBD_DATASETS:
		for perturbation in PERTURBATIONS:
			if perturbation == 0:
				TRAIN_ATTACKS = [CLEAN]
			else:
				TRAIN_ATTACKS = [WASSERSTEIN_META]

			for train_attack_type in TRAIN_ATTACKS:
				evaluation_message(
					"Logging values for {}, perturbation {}, train attack type {}".format(dataset_name, perturbation,
																						  train_attack_type))
				if perturbation != 0:
					test_standard_spn_discrete(run_id=951, device=device, specific_datasets=dataset_name, is_adv=True,
											   train_attack_type=train_attack_type, perturbations=perturbation)
				elif perturbation == 0:
					test_standard_spn_discrete(run_id=951, device=device, specific_datasets=dataset_name, is_adv=False,
											   train_attack_type=train_attack_type, perturbations=perturbation)
