import wandb
import cn_base as CN
from constants import *
from utils import pretty_print_dictionary, dictionary_to_file

############################################################################


wandb_run = wandb.init(project="DROCN", entity="utd-ml-pgm")

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

		cll_wandb_tables = dict()
		for evidence_percentage in EVIDENCE_PERCENTAGES:
			cll_table = wandb.Table(columns=columns)
			cll_wandb_tables[evidence_percentage] = cll_table
		dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES] = cll_wandb_tables

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
		cll_tables = dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES]

		evaluation_message("Dataset : {}".format(dataset_name))
		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = CN.load_dataset(dataset_name)

		evaluation_message("Loading Cnet")

		trained_clean_cnet = CN.load_pretrained_cnet(run_id, dataset_name, attack_type=CLEAN, perturbations=0)
		if trained_clean_cnet is None:
			trained_clean_cnet = CN.train_cnet(run_id, dataset_name, train_x, valid_x, test_x, perturbations,
											   attack_type=CLEAN, is_adv=False)

		if is_adv:
			print("Considering adversarial trained cutset network")
			trained_adv_cnet = CN.load_pretrained_cnet(run_id, dataset_name, attack_type=train_attack_type,
													   perturbations=perturbations)
			if trained_adv_cnet is None:
				evaluation_message("Training adversarial einet with attack type {}".format(train_attack_type))
				trained_adv_cnet = CN.train_cnet(run_id, dataset_name, train_x, valid_x, test_x,
												 perturbations, attack_type=train_attack_type, is_adv=True)
			else:
				trained_adv_cnet = trained_clean_cnet
				evaluation_message("Loaded pretrained einet for the configuration")

			# ------------------------------------------------------------------------------------------
			# ------  QUALITATIVE EXAMPLE AREA ------

			# SPN.generate_conditional_samples(trained_adv_cnet, structure, dataset_name, einet_args, test_x,
			# 								 train_attack_type)

			# ------------------------------------------------------------------------------------------
			# ------  TESTING AREA ------

			# ------  AVERAGE ATTACK AREA ------

			av_mean_ll_dict, av_std_ll_dict = CN.fetch_average_likelihoods_for_data(dataset_name, trained_adv_cnet,
																					test_x)

			def attack_test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet, train_x, test_x,
								 perturbations, attack_type, is_adv):
				mean_ll, std_ll, attack_test_x = CN.test_cnet(dataset_name, trained_adv_cnet, trained_clean_cnet,
															  train_x, test_x, perturbations=perturbations,
															  attack_type=attack_type, is_adv=is_adv)
				evaluation_message("{} Mean LL : {}, Std LL : {}".format(attack_type, mean_ll, std_ll))
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

			# ------------------------------------------------------------------------------------------------
			# ----------------------------- CONDITIONAL LIKELIHOOD AREA --------------------------------------
			# ------------------------------------------------------------------------------------------------

			def attack_test_conditional_cnet(dataset_name, trained_adv_cnet, test_attack_type, perturbations,
											 evidence_percentage, test_x):

				mean_ll, std_ll = CN.test_conditional_cnet(dataset_name, trained_adv_cnet, test_attack_type,
														   perturbations, evidence_percentage, test_x)
				evaluation_message(
					"{}-{},  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(test_attack_type,
																						  perturbations,
																						  evidence_percentage,
																						  mean_ll, std_ll))
				return mean_ll, std_ll

			# ---------- AVERAGE ATTACK AREA ------

			# 8. Average attack dictionary
			av_mean_cll_dict, av_std_cll_dict = CN.fetch_average_conditional_likelihoods_for_data(dataset_name,
																								  trained_adv_cnet,
																								  test_x)

			for evidence_percentage in EVIDENCE_PERCENTAGES:
				cll_table = cll_tables[evidence_percentage]

				dataset_distribution_evidence_results = dict()

				# 1. Original Test Set
				standard_mean_cll, standard_std_cll = attack_test_conditional_cnet(dataset_name, trained_adv_cnet,
																				   test_attack_type=CLEAN,
																				   perturbations=0,
																				   evidence_percentage=evidence_percentage,
																				   test_x=test_x)

				# ---------- LOCAL SEARCH AREA ------

				# 2. Local search - 1
				ls1_mean_cll, ls1_std_cll = attack_test_conditional_cnet(dataset_name, trained_adv_cnet,
																				   test_attack_type=LOCAL_SEARCH,
																				   perturbations=1,
																				   evidence_percentage=evidence_percentage,
																				   test_x=ls1_test_x)

				# 3. Local search - 3
				ls3_mean_cll, ls3_std_cll = attack_test_conditional_cnet(dataset_name, trained_adv_cnet,
																				   test_attack_type=LOCAL_SEARCH,
																				   perturbations=3,
																				   evidence_percentage=evidence_percentage,
																				   test_x=ls3_test_x)

				# 4. Local search - 5
				ls5_mean_cll, ls5_std_cll = attack_test_conditional_cnet(dataset_name, trained_adv_cnet,
																				   test_attack_type=LOCAL_SEARCH,
																				   perturbations=5,
																				   evidence_percentage=evidence_percentage,
																				   test_x=ls5_test_x)

				# ---------- RESTRICTED LOCAL SEARCH AREA ------

				# 5. Restricted Local search - 1
				rls1_mean_cll, rls1_std_cll = attack_test_conditional_cnet(dataset_name, trained_adv_cnet,
																				   test_attack_type=RESTRICTED_LOCAL_SEARCH,
																				   perturbations=1,
																				   evidence_percentage=evidence_percentage,
																				   test_x=rls1_test_x)

				# 6. Restricted Local search - 3
				rls3_mean_cll, rls3_std_cll = attack_test_conditional_cnet(dataset_name, trained_adv_cnet,
																				   test_attack_type=RESTRICTED_LOCAL_SEARCH,
																				   perturbations=3,
																				   evidence_percentage=evidence_percentage,
																				   test_x=rls3_test_x)

				# 7. Restricted Local search - 5
				rls5_mean_cll, rls5_std_cll = attack_test_conditional_cnet(dataset_name, trained_adv_cnet,
																				   test_attack_type=RESTRICTED_LOCAL_SEARCH,
																				   perturbations=5,
																				   evidence_percentage=evidence_percentage,
																				   test_x=rls5_test_x)
				#
				# # ---------- WEAKER MODEL AREA ------

				# 8. Weaker model - 1
				w1_mean_cll, w1_std_cll = attack_test_conditional_cnet(dataset_name, trained_adv_cnet,
																				   test_attack_type=WEAKER_MODEL,
																				   perturbations=1,
																				   evidence_percentage=evidence_percentage,
																				   test_x=w1_test_x)

				# 9. Weaker model - 3
				w3_mean_cll, w3_std_cll = attack_test_conditional_cnet(dataset_name, trained_adv_cnet,
																				   test_attack_type=WEAKER_MODEL,
																				   perturbations=3,
																				   evidence_percentage=evidence_percentage,
																				   test_x=w3_test_x)

				# 10. Weaker model - 5
				w5_mean_cll, w5_std_cll = attack_test_conditional_cnet(dataset_name, trained_adv_cnet,
																				   test_attack_type=WEAKER_MODEL,
																				   perturbations=5,
																				   evidence_percentage=evidence_percentage,
																				   test_x=w5_test_x)

				# -------------------------------- LOG CONDITIONALS TO WANDB TABLES ------------------------------------

				cll_table.add_data(train_attack_type, perturbations, standard_mean_cll, standard_std_cll,
								   ls1_mean_cll, ls1_std_cll, ls3_mean_cll, ls3_std_cll, ls5_mean_cll, ls5_std_cll,
								   rls1_mean_cll, rls1_std_cll, rls3_mean_cll, rls3_std_cll, rls5_mean_cll,
								   rls5_std_cll,
								   av_mean_cll_dict[1][evidence_percentage], av_std_cll_dict[1][evidence_percentage],
								   av_mean_cll_dict[3][evidence_percentage], av_std_cll_dict[3][evidence_percentage],
								   av_mean_cll_dict[5][evidence_percentage], av_std_cll_dict[5][evidence_percentage],
								   w1_mean_cll, w1_std_cll, w3_mean_cll, w3_std_cll, w5_mean_cll, w5_std_cll)


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
					test_cnet(run_id=224, specific_datasets=dataset_name, is_adv=True,
											   train_attack_type=train_attack_type, perturbations=perturbation)
				elif perturbation == 0:
					test_cnet(run_id=224, specific_datasets=dataset_name, is_adv=False,
											   train_attack_type=train_attack_type, perturbations=perturbation)

		dataset_wandb_tables = fetch_wandb_table(dataset_name)
		ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]
		cll_tables = dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES]

		wandb_run.log({"{}-LL".format(dataset_name): ll_table})
		for evidence_percentage in EVIDENCE_PERCENTAGES:
			cll_ev_table = cll_tables[evidence_percentage]
			wandb_run.log({"{}-CLL-{}".format(dataset_name, evidence_percentage): cll_ev_table})
