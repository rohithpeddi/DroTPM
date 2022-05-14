import wandb
import cn_base as CN
from constants import *
from utils import pretty_print_dictionary, dictionary_to_file

############################################################################


# wandb_run = wandb.init(project="DROCN", entity="utd-ml-pgm")

columns = ["attack_type", "perturbations", "standard_mean_ll", "standard_std_ll",
		   "ls1_mean_ll", "ls1_std_ll", "ls3_mean_ll", "ls3_std_ll", "ls5_mean_ll", "ls5_std_ll",
		   "rls1_mean_ll", "rls1_std_ll", "rls3_mean_ll", "rls3_std_ll", "rls5_mean_ll", "rls5_std_ll",
		   "av1_mean_ll", "av1_std_ll", "av3_mean_ll", "av3_std_ll", "av5_mean_ll", "av5_std_ll",
		   "w1_mean_ll", "w1_std_ll", "w3_mean_ll", "w3_std_ll", "w5_mean_ll", "w5_std_ll"]

wandb_tables = dict()


############################################################################

def evaluation_message(message):
	print("\n")
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


def fetch_wandb_table(dataset_name):
	if dataset_name not in wandb_tables:
		dataset_wandb_tables = dict()

		ll_table = wandb.Table(columns=columns)
		dataset_wandb_tables[LOGLIKELIHOOD_TABLE] = ll_table

		wandb_tables[dataset_name] = dataset_wandb_tables

	return wandb_tables[dataset_name]


def test_cnet(run_id, specific_datasets=None, is_adv=False, train_attack_type=None, perturbations=None):
	if specific_datasets is None:
		specific_datasets = DISCRETE_DATASETS
	else:
		specific_datasets = [specific_datasets] if type(specific_datasets) is not list else specific_datasets

	results = dict()
	for dataset_name in specific_datasets:
		dataset_wandb_tables = fetch_wandb_table(dataset_name)
		ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]

		evaluation_message("Dataset : {}".format(dataset_name))
		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = CN.load_dataset(dataset_name)

		evaluation_message("Loading Cnet")

		trained_clean_cnet = CN.load_pretrained_cnet(run_id, dataset_name, attack_type=CLEAN, perturbations=0)
		if trained_clean_cnet is None:
			trained_clean_cnet = CN.train_cnet(run_id, dataset_name, train_x, valid_x, test_x, perturbations,
											   attack_type=CLEAN, is_adv=False)

		trained_adv_cnet = trained_clean_cnet
		if is_adv:
			print("Considering adversarial trained cutset network")
			trained_adv_cnet = CN.load_pretrained_cnet(run_id, dataset_name, attack_type=train_attack_type,
													   perturbations=perturbations)
			if trained_adv_cnet is None:
				evaluation_message("Training adversarial cnet with attack type {}".format(train_attack_type))
				trained_adv_cnet = CN.train_cnet(run_id, dataset_name, train_x, valid_x, test_x,
												 perturbations, attack_type=train_attack_type, is_adv=True)
			else:
				evaluation_message("Loaded pretrained cnet for the configuration")

		# ------------------------------------------------------------------------------------------
		# ------  TESTING AREA ------

		# ------  AVERAGE ATTACK AREA ------

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


if __name__ == '__main__':

	for dataset_name in DEBD_DATASETS:
		for perturbation in PERTURBATIONS:
			if perturbation == 0:
				TRAIN_ATTACKS = [CLEAN]
			else:
				TRAIN_ATTACKS = [AMBIGUITY_SET_UNIFORM]
			# TRAIN_ATTACKS = [LOCAL_SEARCH, RESTRICTED_LOCAL_SEARCH]

			for train_attack_type in TRAIN_ATTACKS:
				evaluation_message(
					"Logging values for {}, perturbation {}, train attack type {}".format(dataset_name, perturbation,
																						  train_attack_type))
				if perturbation != 0:
					test_cnet(run_id=1351, specific_datasets=dataset_name, is_adv=True,
							  train_attack_type=train_attack_type, perturbations=perturbation)
				elif perturbation == 0:
					test_cnet(run_id=1351, specific_datasets=dataset_name, is_adv=False,
							  train_attack_type=train_attack_type, perturbations=perturbation)

		dataset_wandb_tables = fetch_wandb_table(dataset_name)
		ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]

	# wandb_run.log({"{}-LL".format(dataset_name): ll_table})
