import numpy as np
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

###################################################################################################

wandb_run = wandb.init(project="DROSPN", entity="utd-ml-pgm")

columns = ["batch_size", "learning_rate", "weight_decay", "avg_mean", "avg_std", "run_1_mean", "run_2_mean",
		   "run_3_mean"]

wandb_tables = dict()


###################################################################################################

def evaluation_message(message):
	print("\n")
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


def fetch_wandb_table(dataset_name):
	if dataset_name not in wandb_tables:
		dataset_wandb_tables = dict()
		hp_tuning_table = wandb.Table(columns=columns)
		dataset_wandb_tables[HYPER_PARAMETER_TUNING_TABLE] = hp_tuning_table
		wandb_tables[dataset_name] = dataset_wandb_tables
	return wandb_tables[dataset_name]


def fetch_einet_args_discrete(dataset_name, num_var, exponential_family, exponential_family_args):
	einet_args = dict()
	einet_args[NUM_VAR] = num_var

	num_distributions = None
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


def test_standard_spn_discrete(run_id, device, batch_size, optimizer_learning_rate, optimizer_weight_decay,
							   specific_datasets=None):
	if specific_datasets is None:
		specific_datasets = DISCRETE_DATASETS
	else:
		specific_datasets = [specific_datasets] if type(specific_datasets) is not list else specific_datasets

	for dataset_name in specific_datasets:
		dataset_wandb_tables = fetch_wandb_table(dataset_name)
		hp_tuning_table = dataset_wandb_tables[HYPER_PARAMETER_TUNING_TABLE]

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

			def attack_test_einet(trained_einet, test_x, batch_size, run_no):
				trained_einet.eval()
				test_lls = EinsumNetwork.fetch_likelihoods_for_data(trained_einet, test_x, batch_size=batch_size)
				mean_ll, std_ll = SPN.get_stats(test_lls)
				evaluation_message("{} Run {} Mean LL : {}, Std LL : {}".format(dataset_name, run_no, mean_ll, std_ll))
				return mean_ll, std_ll, test_x

			def train_einet(dataset_name, einet, train_x, valid_x, test_x, batch_size, learning_rate, weight_decay):
				early_stopping = EarlyStopping(einet, patience=DEFAULT_EINET_PATIENCE, filepath=EARLY_STOPPING_FILE,
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

			mean_list = []
			std_list = []
			# Run three times and take average of the results and choose best amongst them
			for run_no in range(3):
				evaluation_message("Loading Einet {}".format(run_no))
				clean_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph, device)
				evaluation_message("Training clean einet {}".format(run_no))
				trained_clean_einet = train_einet(dataset_name, clean_einet, train_x, valid_x, test_x, batch_size,
												  learning_rate, weight_decay)
				# 1. Original Test Set
				standard_mean_ll, standard_std_ll, standard_test_x = attack_test_einet(trained_clean_einet, test_x,
																					   batch_size, run_no)
				mean_list.append(standard_mean_ll)
				std_list.append(standard_std_ll)

			# Compilation of results to log into the wandb table
			avg_mean = np.mean(np.array(mean_list))
			avg_std = np.mean(np.array(std_list))
			hp_tuning_table.add_data(batch_size, learning_rate, weight_decay, avg_mean, avg_std, mean_list[0],
									 mean_list[1], mean_list[2])
			evaluation_message("Logging into table: Batch size {}, Learning rate {}, Weight decay {}, Avg Mean {}, Avg Std {}, Mean run 1 {}, Mean run 2 {}, Mean run 3 {}".format(batch_size, learning_rate, weight_decay, avg_mean, avg_std, mean_list[0],
									 mean_list[1], mean_list[2]))


if __name__ == '__main__':

	# Hyper parameters to tune
	# 1. Batch Size
	# 2. Optimizer Learning rate
	# 3. Optimizer Weight Decay
	# For every combination of the hyper-parameters we run thrice and take average

	BATCH_SIZES = [100, 200, 300, 500, 600, 700]
	OPTIMIZER_LEARNING_RATES = [5e-3, 1e-2, 5e-2, 1e-1]
	OPTIMIZER_WEIGHT_DECAYS = [1e-4, 1e-3, 1e-2]

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	for dataset_name in DEBD_DATASETS:

		for batch_size in BATCH_SIZES:
			for learning_rate in OPTIMIZER_LEARNING_RATES:
				for weight_decay in OPTIMIZER_WEIGHT_DECAYS:
					evaluation_message(
						"Training dataset {}, using batch_size {}, learning_rate {}, weight_decay {}".format(
							dataset_name, batch_size, learning_rate, weight_decay))
					test_standard_spn_discrete(run_id=852, device=device, batch_size=batch_size,
											   optimizer_learning_rate=learning_rate,
											   optimizer_weight_decay=weight_decay, specific_datasets=dataset_name)

		# Log the results into wandb table for each dataset
		dataset_wandb_tables = fetch_wandb_table(dataset_name)
		hp_tuning_table = dataset_wandb_tables[HYPER_PARAMETER_TUNING_TABLE]

		wandb_run.log({"{}-LL".format(dataset_name): hp_tuning_table})
