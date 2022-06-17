import wandb
import json
import csv
import os
import numpy as np
import pandas as pd
from constants import *
from utils import mkdir_p

api = wandb.Api()

DRO_RANDOM_SGD_SPN = "DRO-Random-SGD-SPN"
DRO_GRADIENT_SGD_SPN = "DRO-GRADIENT-SGD-SPN"

RANDOM_DRO_SAMPLES = "random_dro_samples"

entities = ["utd-ml-pgm"]
projects = [DRO_GRADIENT_SGD_SPN, DRO_RANDOM_SGD_SPN]

mkdir_p("sup_results")
mkdir_p("results")

directory = 'sup_results'


def generate_csv():
	for i in range(2):
		entity = entities[0]
		project = projects[i]

		runs = api.runs(entity + "/" + project)
		for run in runs:
			summary_list = []
			summary_list.append(run.summary._json_dict)

			tables = list(summary_list[0].keys())
			tables.sort()

			for table in tables:
				print("-------------------------------------------------------------")
				print("Writing the information from project {}, file {}".format(project, table))

				for dataset_name in DEBD_DATASETS:
					if dataset_name in table:
						if "-LL" in table:
							raw_data_file = open("results/{}_{}.csv".format(dataset_name, "LL"), "a")
						elif "-CLL" in table:
							for evidence_percentage in EVIDENCE_PERCENTAGES:
								if str(evidence_percentage) in table:
									raw_data_file = open(
										"results/{}_{}_{}.csv".format(dataset_name, "CLL", evidence_percentage),
										"a")
						else:
							continue

						csv_writer = csv.writer(raw_data_file)

						if project == DRO_GRADIENT_SGD_SPN:
							csv_writer.writerow(
								["MODEL", "C", "LS-1", "LS-3", "LS-5", "RLS-1", "RLS-3", "RLS-5",
								 "AV-1", "AV-3", "AV-5", "W-1", "W-3", "W-5"])

						meta = json.load(run.file(summary_list[0][table]['path']).download())
						table_data = meta["data"]

						for id in range(len(table_data)):
							data_row = table_data[id]
							if project == DRO_RANDOM_SGD_SPN and id == 0:
								print("Ignoring : {}".format(data_row))
								continue

							modified_data_row = []

							# Change attack name
							attack_type = data_row[0]
							if data_row[0] == WASSERSTEIN_META:
								if project == DRO_GRADIENT_SGD_SPN:
									attack_type = "GDRO"
								attack_type = "{}-{}".format(attack_type, data_row[1])
							elif RANDOM_DRO_SAMPLES in data_row[0]:
								if project == DRO_RANDOM_SGD_SPN:
									attack_type = "KDRO"
									samples = (data_row[0].split("_"))[3]
								attack_type = "{}-{}-{}".format(attack_type, samples, data_row[1])
							elif data_row[0] == CLEAN:
								attack_type = "CLEAN"
							data_row[0] = attack_type

							# Adding attack type and perturbations
							modified_data_row.append(attack_type)

							# Adding only mean columns
							for didx in range(2, len(data_row)):
								modified_data_row.append(round(data_row[didx], 2))
							csv_writer.writerow(modified_data_row)


# def generate_results_csv():
# 	for i in range(2):
# 		entity = entities[0]
# 		project = projects[i]
#
# 		runs = api.runs(entity + "/" + project)
# 		for run in runs:
# 			summary_list = []
# 			summary_list.append(run.summary._json_dict)
#
# 			tables = list(summary_list[0].keys())
# 			tables.sort()
#
# 			for table in tables:
# 				print("-------------------------------------------------------------")
# 				print("Writing the information from project {}, file {}".format(project, table))
#
# 				for dataset_name in DEBD_DATASETS:
# 					if dataset_name in table:
# 						if "-LL" in table:
# 							raw_data_file = open("results/{}_{}.csv".format(dataset_name, "LL"), "a")
# 						elif "-CLL" in table:
# 							for evidence_percentage in EVIDENCE_PERCENTAGES:
# 								if str(evidence_percentage) in table:
# 									raw_data_file = open(
# 										"results/{}_{}_{}.csv".format(dataset_name, "CLL", evidence_percentage), "a")
# 						else:
# 							continue
#
# 						csv_writer = csv.writer(raw_data_file)
#
# 						if project == ROSPN:
# 							csv_writer.writerow(
# 								["MODEL", "C", "LS-1", "LS-3", "LS-5", "RLS-1", "RLS-3", "RLS-5",
# 								 "AV-1", "AV-3", "AV-5", "W-1", "W-3", "W-5"])
#
# 						meta = json.load(run.file(summary_list[0][table]['path']).download())
# 						table_data = meta["data"]
#
# 						for id in range(len(table_data)):
# 							data_row = table_data[id]
# 							if project == ROSPN_O and id == 0:
# 								print("Ignoring : {}".format(data_row))
# 								continue
#
# 							modified_data_row = []
#
# 							# Change attack name
# 							attack_type = data_row[0]
# 							if data_row[0] == LOCAL_SEARCH:
# 								if project == ROSPN:
# 									attack_type = "RALS"
# 								elif project == ROSPN_O:
# 									attack_type = "ALS"
# 								attack_type = "{}-{}".format(attack_type, data_row[1])
# 							elif data_row[0] == RESTRICTED_LOCAL_SEARCH:
# 								if project == ROSPN:
# 									attack_type = "RARLS"
# 								elif project == ROSPN_O:
# 									attack_type = "ARLS"
# 								attack_type = "{}-{}".format(attack_type, data_row[1])
# 							elif data_row[0] == CLEAN:
# 								attack_type = "CLEAN"
# 							data_row[0] = attack_type
#
# 							# Adding attack type and perturbations
# 							modified_data_row.append(attack_type)
#
# 							# Adding only mean columns
# 							for didx in range(2, len(data_row)):
# 								if didx % 2 == 0:
# 									modified_data_row.append(round(data_row[didx], 2))
# 							csv_writer.writerow(modified_data_row)


def get_details(filename):
	name, description, type = None, None, None
	for dataset_name in DEBD_DATASETS:
		if dataset_name in filename:
			name = DEBD_display_name[dataset_name]
		if "_LL" in filename:
			description = "negative log likelihood scores"
			type = "LL"
		elif "_CLL" in filename:
			for evidence_percentage in EVIDENCE_PERCENTAGES:
				if str(evidence_percentage) in filename:
					description = "conditional log likelihood scores with evidence percentage {}".format(
						evidence_percentage * 100)
					type = "CLL-{}".format(evidence_percentage)
	return name, description, type


def generate_supplementary_latex_code(filename):
	directory = 'sup_results'
	file = os.path.join(directory, filename)
	print("Converting to file {} to latex".format(filename))
	name, description, type = get_details(filename)
	if os.path.isfile(file):
		df = pd.read_csv(file, header=None)
		with open("sup_table.tex", "a") as f:
			f.write("\n")
			f.write("\\begin{table}\n")
			f.write("\\centering \n")
			f.write("\\caption{\\label{tab:" + name + "-" + type + "}" + name + "-" + description + "}\n")
			f.write("\\begin{tabular}{ |" + " | ".join(["c"] * len(df.columns)) + " |}\n")
			f.write("\\hline \n")
			for i, row in df.iterrows():
				f.write(" & ".join([str(x) for x in row.values]) + " \\\\\n")
				if i == 0:
					f.write("\\hline \n")
			f.write("\\hline \n")
			f.write("\\end{tabular}\n")
			f.write("\\end{table}\n")


def generate_suplementary_latex():
	for type in ["LL", "CLL"]:
		for dataset_name in DEBD_DATASETS:
			if type == "LL":
				generate_supplementary_latex_code("{}_{}.csv".format(dataset_name, type))
			elif type == "CLL":
				for evidence_percentage in EVIDENCE_PERCENTAGES:
					generate_supplementary_latex_code("{}_{}_{}.csv".format(dataset_name, type, evidence_percentage))


# def generate_results_code(filename):
# 	directory = 'results'
# 	mkdir_p("results_final")
# 	file = os.path.join(directory, filename)
# 	print("Parsing file {} to generate final csv".format(filename))
# 	name, description, type = get_details(filename)
# 	if os.path.isfile(file):
# 		df = pd.read_csv(file, header=[0])
# 		with open("results_final/{}.csv".format(type), "a") as f:
# 			result_row = [name]
# 			# D = 1
# 			# Clean SPN [CLEAN]
# 			clean_row = [df.iloc[0]['C'], df.iloc[0]['LS-1'], df.iloc[0]['RLS-1']]
# 			# Adv SPN [ALS]
# 			adv_row = [df.iloc[7]['C'], df.iloc[7]['LS-1'], df.iloc[7]['RLS-1']]
# 			# Regularized Adv SPN [RALS]
# 			radv_row = [df.iloc[1]['C'], df.iloc[1]['LS-1'], df.iloc[1]['RLS-1']]
# 			result_row.extend(clean_row)
# 			result_row.extend(adv_row)
# 			result_row.extend(radv_row)
#
# 			# D = 3
# 			# Clean SPN [CLEAN]
# 			clean_row = [df.iloc[0]['C'], df.iloc[0]['LS-3'], df.iloc[0]['RLS-3']]
# 			# Adv SPN [ALS]
# 			adv_row = [df.iloc[9]['C'], df.iloc[9]['LS-3'], df.iloc[9]['RLS-3']]
# 			# Regularized Adv SPN [RALS]
# 			radv_row = [df.iloc[3]['C'], df.iloc[3]['LS-3'], df.iloc[3]['RLS-3']]
# 			result_row.extend(clean_row)
# 			result_row.extend(adv_row)
# 			result_row.extend(radv_row)
#
# 			# D = 5
# 			# Clean SPN [CLEAN]
# 			clean_row = [df.iloc[0]['C'], df.iloc[0]['LS-5'], df.iloc[0]['RLS-5']]
# 			# Adv SPN [ALS]
# 			adv_row = [df.iloc[11]['C'], df.iloc[11]['LS-5'], df.iloc[11]['RLS-5']]
# 			# Regularized Adv SPN [RALS]
# 			radv_row = [df.iloc[5]['C'], df.iloc[5]['LS-5'], df.iloc[5]['RLS-5']]
# 			result_row.extend(clean_row)
# 			result_row.extend(adv_row)
# 			result_row.extend(radv_row)
#
# 			print("Storing model scores {}".format([df.iloc[0]['MODEL'], df.iloc[7]['MODEL'], df.iloc[1]['MODEL'],
# 													df.iloc[0]['MODEL'], df.iloc[9]['MODEL'], df.iloc[3]['MODEL'],
# 													df.iloc[0]['MODEL'], df.iloc[11]['MODEL'], df.iloc[5]['MODEL']]))
# 			csv_writer = csv.writer(f)
# 			csv_writer.writerow(result_row)


# def generate_results_code(filename):
# 	directory = 'results'
# 	mkdir_p("results_final")
# 	file = os.path.join(directory, filename)
# 	print("Parsing file {} to generate final csv".format(filename))
# 	name, description, type = get_details(filename)
# 	if os.path.isfile(file):
# 		df = pd.read_csv(file, header=[0])
# 		with open("results_final/{}.csv".format(type), "a") as f:
# 			result_row = [name]
# 			# # D = 1
# 			# # Clean Test
# 			# clean_row = [df.iloc[0]['C'], df.iloc[7]['C'], df.iloc[1]['C']]
# 			# # Adv Test
# 			# adv_row = [df.iloc[0]['LS-1'], df.iloc[7]['LS-1'], df.iloc[1]['LS-1']]
# 			# # Rest Test
# 			# rest_row = [df.iloc[0]['RLS-1'], df.iloc[7]['RLS-1'], df.iloc[1]['RLS-1']]
# 			# result_row.extend(clean_row)
# 			# result_row.extend(adv_row)
# 			# result_row.extend(rest_row)
#
# 			# # D = 3
# 			# # Clean Test
# 			# clean_row = [df.iloc[0]['C'], df.iloc[9]['C'], df.iloc[3]['C']]
# 			# # Adv Test
# 			# adv_row = [df.iloc[0]['LS-3'], df.iloc[9]['LS-3'], df.iloc[3]['LS-3']]
# 			# # Rest Test
# 			# rest_row = [df.iloc[0]['RLS-3'], df.iloc[9]['RLS-3'], df.iloc[3]['RLS-3']]
# 			# result_row.extend(clean_row)
# 			# result_row.extend(adv_row)
# 			# result_row.extend(rest_row)
# 			#
# 			#
# 			# D = 5
# 			# Clean Test
# 			clean_row = [df.iloc[0]['C'], df.iloc[11]['C'], df.iloc[5]['C']]
# 			# Adv Test
# 			adv_row = [df.iloc[0]['LS-5'], df.iloc[11]['LS-5'], df.iloc[5]['LS-5']]
# 			# Rest Test
# 			rest_row = [df.iloc[0]['RLS-5'], df.iloc[11]['RLS-5'], df.iloc[5]['RLS-5']]
# 			result_row.extend(clean_row)
# 			result_row.extend(adv_row)
# 			result_row.extend(rest_row)
#
#
# 			print("Storing model scores {}".format([df.iloc[0]['MODEL'], df.iloc[7]['MODEL'], df.iloc[1]['MODEL'],
# 													df.iloc[0]['MODEL'], df.iloc[9]['MODEL'], df.iloc[3]['MODEL'],
# 													df.iloc[0]['MODEL'], df.iloc[11]['MODEL'], df.iloc[5]['MODEL']]))
# 			csv_writer = csv.writer(f)
# 			csv_writer.writerow(result_row)

# def generate_results_code(filename):
# 	directory = 'results'
# 	mkdir_p("results_final")
# 	file = os.path.join(directory, filename)
# 	print("Parsing file {} to generate final csv".format(filename))
# 	name, description, type = get_details(filename)
# 	if os.path.isfile(file):
# 		df = pd.read_csv(file, header=[0])
# 		with open("results_final/{}.csv".format(type), "a") as f:
# 			result_row = [name]
#
# 			# AV = 1
# 			# SPN, SPNa, SPNr
# 			av1_row = [df.iloc[0]['AV-1'], df.iloc[7]['AV-1'], df.iloc[1]['AV-1']]
# 			# AV = 3
# 			# SPN, SPNa, SPNr
# 			av3_row = [df.iloc[0]['AV-3'], df.iloc[9]['AV-3'], df.iloc[3]['AV-3']]
# 			# AV = 5
# 			# SPN, SPNa, SPNr
# 			av5_row = [df.iloc[0]['AV-5'], df.iloc[11]['AV-5'], df.iloc[5]['AV-5']]
#
# 			result_row.extend(av1_row)
# 			result_row.extend(av3_row)
# 			result_row.extend(av5_row)
#
# 			print("Storing model scores {}".format([df.iloc[0]['MODEL'], df.iloc[7]['MODEL'], df.iloc[1]['MODEL'],
# 													df.iloc[0]['MODEL'], df.iloc[9]['MODEL'], df.iloc[3]['MODEL'],
# 													df.iloc[0]['MODEL'], df.iloc[11]['MODEL'], df.iloc[5]['MODEL']]))
# 			csv_writer = csv.writer(f)
# 			csv_writer.writerow(result_row)

def generate_results_code(filename):
	directory = 'results'
	mkdir_p("results_final")
	file = os.path.join(directory, filename)
	print("Parsing file {} to generate final csv".format(filename))
	name, description, type = get_details(filename)
	if os.path.isfile(file):
		df = pd.read_csv(file, header=[0])
		with open("results_final/{}.csv".format(type), "a") as f:
			result_row = [name]

			# W = 1
			# SPN, SPNa, SPNr
			av1_row = [df.iloc[0]['W-1'], df.iloc[7]['W-1'], df.iloc[1]['W-1']]
			# W = 3
			# SPN, SPNa, SPNr
			av3_row = [df.iloc[0]['W-3'], df.iloc[9]['W-3'], df.iloc[3]['W-3']]
			# W = 5
			# SPN, SPNa, SPNr
			av5_row = [df.iloc[0]['W-5'], df.iloc[11]['W-5'], df.iloc[5]['W-5']]

			result_row.extend(av1_row)
			result_row.extend(av3_row)
			result_row.extend(av5_row)

			print("Storing model scores {}".format([df.iloc[0]['MODEL'], df.iloc[7]['MODEL'], df.iloc[1]['MODEL'],
													df.iloc[0]['MODEL'], df.iloc[9]['MODEL'], df.iloc[3]['MODEL'],
													df.iloc[0]['MODEL'], df.iloc[11]['MODEL'], df.iloc[5]['MODEL']]))
			csv_writer = csv.writer(f)
			csv_writer.writerow(result_row)


def generate_paper_results_code(filename):
	directory = 'results'
	mkdir_p("paper_results_final")
	file = os.path.join(directory, filename)
	print("Parsing file {} to generate final csv".format(filename))
	name, description, type = get_details(filename)
	if os.path.isfile(file):
		df = pd.read_csv(file, header=[0])
		with open("paper_results_final/{}.csv".format(type), "a") as f:
			result_row = [name]
			# result_row_1 = [name]

			# e = 1
			# T
			# SPN, SPNk, SPNg
			e1t1_row = [df.iloc[0]['C'], min(df.iloc[4]['C'], df.iloc[5]['C'], df.iloc[6]['C']), df.iloc[1]['C']]

			# Ta
			# SPN, SPNk, SPNg
			e1ta1_row = [df.iloc[0]['LS-1'], min(df.iloc[4]['LS-1'], df.iloc[5]['LS-1'], df.iloc[6]['LS-1']),
						 df.iloc[1]['LS-1']]

			# Tr
			# SPN, SPNk, SPNg
			e1tr1_row = [df.iloc[0]['RLS-1'], min(df.iloc[4]['RLS-1'], df.iloc[5]['RLS-1'], df.iloc[6]['RLS-1']),
						 df.iloc[1]['RLS-1']]

			result_row.extend(e1t1_row)
			result_row.extend(e1ta1_row)
			result_row.extend(e1tr1_row)

			# result_row_1.extend(e1t1_row)
			# result_row_1.extend(e1ta1_row)
			# result_row_1.extend(e1tr1_row)
			#
			# result_row_3 = [name]

			# e = 3
			# T
			# SPN, SPNk, SPNg
			e3t3_row = [df.iloc[0]['C'], min(df.iloc[7]['C'], df.iloc[8]['C'], df.iloc[9]['C']), df.iloc[2]['C']]

			# Ta
			# SPN, SPNk, SPNg
			e3ta3_row = [df.iloc[0]['LS-3'], min(df.iloc[7]['LS-3'], df.iloc[8]['LS-3'], df.iloc[9]['LS-3']),
						 df.iloc[2]['LS-3']]

			# Tr
			# SPN, SPNk, SPNg
			e3tr3_row = [df.iloc[0]['RLS-3'], min(df.iloc[7]['RLS-3'], df.iloc[8]['RLS-3'], df.iloc[9]['RLS-3']),
						 df.iloc[2]['RLS-3']]

			result_row.extend(e3t3_row)
			result_row.extend(e3ta3_row)
			result_row.extend(e3tr3_row)

			# result_row_3.extend(e3t3_row)
			# result_row_3.extend(e3ta3_row)
			# result_row_3.extend(e3tr3_row)
			#
			# result_row_5 = [name]

			# e = 5
			# T
			# SPN, SPNk, SPNg
			e5t5_row = [df.iloc[0]['C'], min(df.iloc[10]['C'], df.iloc[11]['C'], df.iloc[12]['C']), df.iloc[3]['C']]

			# Ta
			# SPN, SPNk, SPNg
			e5ta5_row = [df.iloc[0]['LS-5'], min(df.iloc[10]['LS-5'], df.iloc[11]['LS-5'], df.iloc[12]['LS-5']),
						 df.iloc[3]['LS-5']]

			# Tr
			# SPN, SPNk, SPNg
			e5tr5_row = [df.iloc[0]['RLS-5'], min(df.iloc[10]['RLS-5'], df.iloc[11]['RLS-5'], df.iloc[12]['RLS-5']),
						 df.iloc[3]['RLS-5']]

			result_row.extend(e5t5_row)
			result_row.extend(e5ta5_row)
			result_row.extend(e5tr5_row)

			# result_row_5.extend(e5t5_row)
			# result_row_5.extend(e5ta5_row)
			# result_row_5.extend(e5tr5_row)

			print("Storing model scores {}".format([df.iloc[0]['MODEL'], df.iloc[7]['MODEL'], df.iloc[1]['MODEL'],
													df.iloc[0]['MODEL'], df.iloc[9]['MODEL'], df.iloc[3]['MODEL'],
													df.iloc[0]['MODEL'], df.iloc[11]['MODEL'], df.iloc[5]['MODEL']]))
			csv_writer = csv.writer(f)
			csv_writer.writerow(result_row)
		# csv_writer.writerow(result_row_1)
		# csv_writer.writerow(result_row_3)
		# csv_writer.writerow(result_row_5)


def generate_supplementary_results_code(filename):
	directory = 'results'
	mkdir_p("supplementary_results_final")
	file = os.path.join(directory, filename)
	print("Parsing file {} to generate final csv".format(filename))
	name, description, type = get_details(filename)
	if os.path.isfile(file):
		df = pd.read_csv(file, header=[0])
		with open("supplementary_results_final/{}.csv".format(type), "a") as f:
			result_row = [name]
			# result_row_1 = [name]

			# e = 1
			# T
			# SPN, SPNk, SPNg
			e1t1_row = [df.iloc[0]['C'], min(df.iloc[4]['C'], df.iloc[5]['C'], df.iloc[6]['C']), df.iloc[1]['C']]

			# Ta
			# SPN, SPNk, SPNg
			e1ta1_row = [df.iloc[0]['LS-1'], min(df.iloc[4]['LS-1'], df.iloc[5]['LS-1'], df.iloc[6]['LS-1']),
						 df.iloc[1]['LS-1']]

			# Tr
			# SPN, SPNk, SPNg
			e1tr1_row = [df.iloc[0]['RLS-1'], min(df.iloc[4]['RLS-1'], df.iloc[5]['RLS-1'], df.iloc[6]['RLS-1']),
						 df.iloc[1]['RLS-1']]

			result_row.extend(e1t1_row)
			result_row.extend(e1ta1_row)
			result_row.extend(e1tr1_row)

			# result_row_1.extend(e1t1_row)
			# result_row_1.extend(e1ta1_row)
			# result_row_1.extend(e1tr1_row)
			#
			# result_row_3 = [name]

			# e = 3
			# T
			# SPN, SPNk, SPNg
			e3t3_row = [df.iloc[0]['C'], min(df.iloc[7]['C'], df.iloc[8]['C'], df.iloc[9]['C']), df.iloc[2]['C']]

			# Ta
			# SPN, SPNk, SPNg
			e3ta3_row = [df.iloc[0]['LS-3'], min(df.iloc[7]['LS-3'], df.iloc[8]['LS-3'], df.iloc[9]['LS-3']),
						 df.iloc[2]['LS-3']]

			# Tr
			# SPN, SPNk, SPNg
			e3tr3_row = [df.iloc[0]['RLS-3'], min(df.iloc[7]['RLS-3'], df.iloc[8]['RLS-3'], df.iloc[9]['RLS-3']),
						 df.iloc[2]['RLS-3']]

			result_row.extend(e3t3_row)
			result_row.extend(e3ta3_row)
			result_row.extend(e3tr3_row)

			# result_row_3.extend(e3t3_row)
			# result_row_3.extend(e3ta3_row)
			# result_row_3.extend(e3tr3_row)
			#
			# result_row_5 = [name]

			# e = 5
			# T
			# SPN, SPNk, SPNg
			e5t5_row = [df.iloc[0]['C'], min(df.iloc[10]['C'], df.iloc[11]['C'], df.iloc[12]['C']), df.iloc[3]['C']]

			# Ta
			# SPN, SPNk, SPNg
			e5ta5_row = [df.iloc[0]['LS-5'], min(df.iloc[10]['LS-5'], df.iloc[11]['LS-5'], df.iloc[12]['LS-5']),
						 df.iloc[3]['LS-5']]

			# Tr
			# SPN, SPNk, SPNg
			e5tr5_row = [df.iloc[0]['RLS-5'], min(df.iloc[10]['RLS-5'], df.iloc[11]['RLS-5'], df.iloc[12]['RLS-5']),
						 df.iloc[3]['RLS-5']]

			result_row.extend(e5t5_row)
			result_row.extend(e5ta5_row)
			result_row.extend(e5tr5_row)

			# result_row_5.extend(e5t5_row)
			# result_row_5.extend(e5ta5_row)
			# result_row_5.extend(e5tr5_row)

			print("Storing model scores {}".format([df.iloc[0]['MODEL'], df.iloc[7]['MODEL'], df.iloc[1]['MODEL'],
													df.iloc[0]['MODEL'], df.iloc[9]['MODEL'], df.iloc[3]['MODEL'],
													df.iloc[0]['MODEL'], df.iloc[11]['MODEL'], df.iloc[5]['MODEL']]))
			csv_writer = csv.writer(f)
			csv_writer.writerow(result_row)
		# csv_writer.writerow(result_row_1)
		# csv_writer.writerow(result_row_3)
		# csv_writer.writerow(result_row_5)


def generate_paper_results():
	for type in ["LL", "CLL"]:
		for dataset_name in ['plants', 'bnetflix', 'dna', 'tmovie', 'bbc']:
			if type == "LL":
				generate_paper_results_code("{}_{}.csv".format(dataset_name, type))
			elif type == "CLL":
				for evidence_percentage in EVIDENCE_PERCENTAGES:
					generate_paper_results_code("{}_{}_{}.csv".format(dataset_name, type, evidence_percentage))


def generate_supplementary_results():
	for type in ["LL", "CLL"]:
		for dataset_name in DEBD_DATASETS:
			if type == "LL":
				generate_supplementary_results_code("{}_{}.csv".format(dataset_name, type))
			elif type == "CLL":
				for evidence_percentage in EVIDENCE_PERCENTAGES:
					generate_supplementary_results_code("{}_{}_{}.csv".format(dataset_name, type, evidence_percentage))


# def generate_paper_results_latex_code(filename, type):
# 	directory = 'paper_results_final'
# 	mkdir_p("{}/tex/".format(directory))
# 	file = os.path.join(directory, filename)
# 	print("Converting to file {} to latex".format(filename))
# 	if os.path.isfile(file):
# 		df = pd.read_csv(file, header=None)
# 		with open("results_final/tex/{}_table.tex".format(type), "a") as f:
# 			f.write("\n")
# 			f.write("\\begin{table*}[t]\n")
# 			f.write("\\begin{center} \n")
# 			f.write("\\begin{tabular}{ |" + " | ".join(["c"] * len(df.columns)) + " |}\n")
# 			f.write("\\hline \n")
# 			# f.write(" & \\multicolumn{9}{|c|}{D=1} & \\multicolumn{9}{|c|}{D=3} \\\\\n")
# 			# f.write("\\hline \n")
# 			# f.write(
# 			# 	" & \\multicolumn{3}{|c|}{SPN}  & \\multicolumn{3}{|c|}{SPN\'} & \\multicolumn{3}{|c|}{SPN\"}  & \\multicolumn{3}{|c|}{SPN}  & \\multicolumn{3}{|c|}{SPN\'} & \\multicolumn{3}{|c|}{SPN\"}  \\\\\n")
# 			# f.write("\\hline \n")
# 			# f.write(
# 			# 	" \\multicolumn{1}{|c|}{dataset} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} \\\\\n")
# 			# f.write("\\hline \n")
# 			f.write("\\hline \n")
# 			# f.write(" & \\multicolumn{9}{|c|}{$h$=3} \\\\\n")
# 			# f.write("\\hline \n")
# 			f.write(
# 				" & \\multicolumn{3}{|c|}{\\test}  & \\multicolumn{3}{|c|}{\\testa} & \\multicolumn{3}{|c|}{\\testr}  \\\\\n")
# 			f.write("\\hline \n")
# 			f.write(
# 				" \\multicolumn{1}{|c|}{dataset} & \\multicolumn{1}{|c|}{\\spn} & \\multicolumn{1}{|c|}{\\spnk} & \\multicolumn{1}{|c|}{\\spng} & \\multicolumn{1}{|c|}{\\spn} & \\multicolumn{1}{|c|}{\\spnk} & \\multicolumn{1}{|c|}{\\spng} & \\multicolumn{1}{|c|}{\\spn} & \\multicolumn{1}{|c|}{\\spnk} & \\multicolumn{1}{|c|}{\\spng} \\\\\n")
# 			f.write("\\hline \n")
# 			for i, row in df.iterrows():
# 				f.write(" & ".join([str(x) for x in row.values]) + " \\\\\n")
# 				f.write("\\hline \n")
# 			f.write("\\hline \n")
# 			f.write("\\end{tabular}\n")
# 			f.write("\\end{center} \n")
# 			f.write("\\end{table*}\n")


# def generate_results_latex_code(filename, type):
# 	directory = 'results_final'
# 	mkdir_p("results_final/tex/")
# 	file = os.path.join(directory, filename)
# 	print("Converting to file {} to latex".format(filename))
# 	if os.path.isfile(file):
# 		df = pd.read_csv(file, header=None)
# 		with open("results_final/tex/{}_table.tex".format(type), "a") as f:
# 			f.write("\n")
# 			f.write("\\begin{table*}[t]\n")
# 			f.write("\\begin{center} \n")
# 			f.write("\\begin{tabular}{ |" + " | ".join(["c"] * len(df.columns)) + " |}\n")
# 			f.write("\\hline \n")
# 			# f.write(" & \\multicolumn{9}{|c|}{D=1} & \\multicolumn{9}{|c|}{D=3} \\\\\n")
# 			# f.write("\\hline \n")
# 			# f.write(
# 			# 	" & \\multicolumn{3}{|c|}{SPN}  & \\multicolumn{3}{|c|}{SPN\'} & \\multicolumn{3}{|c|}{SPN\"}  & \\multicolumn{3}{|c|}{SPN}  & \\multicolumn{3}{|c|}{SPN\'} & \\multicolumn{3}{|c|}{SPN\"}  \\\\\n")
# 			# f.write("\\hline \n")
# 			# f.write(
# 			# 	" \\multicolumn{1}{|c|}{dataset} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} \\\\\n")
# 			# f.write("\\hline \n")
# 			f.write("\\hline \n")
# 			# f.write(" & \\multicolumn{9}{|c|}{$h$=3} \\\\\n")
# 			# f.write("\\hline \n")
# 			f.write(
# 				" & \\multicolumn{3}{|c|}{W-1}  & \\multicolumn{3}{|c|}{W-3} & \\multicolumn{3}{|c|}{W-5}  \\\\\n")
# 			f.write("\\hline \n")
# 			f.write(
# 				" \\multicolumn{1}{|c|}{dataset} & \\multicolumn{1}{|c|}{\\spn} & \\multicolumn{1}{|c|}{\\spna} & \\multicolumn{1}{|c|}{\\spnr} & \\multicolumn{1}{|c|}{\\spn} & \\multicolumn{1}{|c|}{\\spna} & \\multicolumn{1}{|c|}{\\spnr} & \\multicolumn{1}{|c|}{\\spn} & \\multicolumn{1}{|c|}{\\spna} & \\multicolumn{1}{|c|}{\\spnr} \\\\\n")
# 			f.write("\\hline \n")
# 			for i, row in df.iterrows():
# 				f.write(" & ".join([str(x) for x in row.values]) + " \\\\\n")
# 				f.write("\\hline \n")
# 			f.write("\\hline \n")
# 			f.write("\\end{tabular}\n")
# 			f.write("\\end{center} \n")
# 			f.write("\\end{table*}\n")


# def generate_supplementary_results_latex_code(filename, type):
# 	directory = 'results_final'
# 	mkdir_p("results_final/tex/")
# 	file = os.path.join(directory, filename)
# 	print("Converting to file {} to latex".format(filename))
# 	if os.path.isfile(file):
# 		df = pd.read_csv(file, header=None)
# 		with open("results_final/tex/{}_table.tex".format(type), "a") as f:
# 			f.write("\n")
# 			f.write("\\begin{table*}[t]\n")
# 			f.write("\\begin{center} \n")
# 			f.write("\\begin{tabular}{ |" + " | ".join(["c"] * len(df.columns)) + " |}\n")
# 			f.write("\\hline \n")
# 			# f.write(" & \\multicolumn{9}{|c|}{D=1} & \\multicolumn{9}{|c|}{D=3} \\\\\n")
# 			# f.write("\\hline \n")
# 			# f.write(
# 			# 	" & \\multicolumn{3}{|c|}{SPN}  & \\multicolumn{3}{|c|}{SPN\'} & \\multicolumn{3}{|c|}{SPN\"}  & \\multicolumn{3}{|c|}{SPN}  & \\multicolumn{3}{|c|}{SPN\'} & \\multicolumn{3}{|c|}{SPN\"}  \\\\\n")
# 			# f.write("\\hline \n")
# 			# f.write(
# 			# 	" \\multicolumn{1}{|c|}{dataset} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} \\\\\n")
# 			# f.write("\\hline \n")
# 			f.write("\\hline \n")
# 			f.write(" & \\multicolumn{9}{|c|}{$h$=3} \\\\\n")
# 			f.write("\\hline \n")
# 			f.write(" & \\multicolumn{3}{|c|}{\\test}  & \\multicolumn{3}{|c|}{\\testa} & \\multicolumn{3}{|c|}{\\testr}  \\\\\n")
# 			f.write("\\hline \n")
# 			f.write(" \\multicolumn{1}{|c|}{dataset} & \\multicolumn{1}{|c|}{\\spn} & \\multicolumn{1}{|c|}{\\spna} & \\multicolumn{1}{|c|}{\\spnr} & \\multicolumn{1}{|c|}{\\spn} & \\multicolumn{1}{|c|}{\\spna} & \\multicolumn{1}{|c|}{\\spnr} & \\multicolumn{1}{|c|}{\\spn} & \\multicolumn{1}{|c|}{\\spna} & \\multicolumn{1}{|c|}{\\spnr} \\\\\n")
# 			f.write("\\hline \n")
# 			for i, row in df.iterrows():
# 				f.write(" & ".join([str(x) for x in row.values]) + " \\\\\n")
# 				f.write("\\hline \n")
# 			f.write("\\hline \n")
# 			f.write("\\end{tabular}\n")
# 			f.write("\\end{center} \n")
# 			f.write("\\end{table*}\n")


def generate_paper_results_latex_code(filename, type):
	directory = 'paper_results_final'
	mkdir_p("{}/tex/".format(directory))
	file = os.path.join(directory, filename)
	print("Converting to file {} to latex".format(filename))
	if os.path.isfile(file):
		df = pd.read_csv(file, header=None)
		with open("paper_results_final/tex/{}_table.tex".format(type), "a") as f:
			# f.write("\n")
			# f.write("\\clearpage \n")
			# f.write("\\onecolumn \n")
			f.write("\\begin{table*}[t]\n")
			f.write("\\begin{center} \n")
			# f.write("\\begin{tabular}{ |" + " | ".join(["c"] * len(df.columns)) + " |}\n")
			f.write("\\begin{tabular}{ |cc|ccc|ccc|ccc| }\n")
			# f.write("\\begin{longtable}{ |cc|ccc|ccc|ccc| }\n")
			f.write("\\hline \n")
			f.write(
				"\\multicolumn{1}{|c|}{ \\multirow{2}{*}{DATASET} } & \\multicolumn{1}{|c|}{ \\multirow{2}{*}{$\\epsilon$} } & \\multicolumn{3}{c|}{\\test} & \\multicolumn{3}{c|}{\\testa} & \\multicolumn{3}{c|}{\\testr} \\\\\n")
			f.write("\\cline{3-11} \n")
			# f.write("\\hline \n")
			f.write(
				" & \\multicolumn{1}{|c|}{}  & \\spn  & \\spnk & \\spng  & \\spn  & \\spnk & \\spng & \\spn & \\spnk & \\spng \\\\\n")
			f.write("\\hline \n")
			for i, row in df.iterrows():
				row_list = list(row.values)
				dataset_name = row_list[0]
				values_list = row_list[1:]

				avg_list = []
				for avgid in range(9):
					avg_list.append(
						round(np.mean([values_list[avgid], values_list[9 + avgid], values_list[18 + avgid]]), 2))

				# row-1
				row1 = "\\multicolumn{1}{|c|}{ \\multirow{3}{*}{" + str(
					dataset_name) + "}} & 1 & \\multicolumn{1}{|c|}{ \\multirow{3}{*}{ " + str(values_list[0]) + "}}"
				for row1id in range(1, 9):
					row1 += " & " + " \\multicolumn{1}{c|}{ " + str(values_list[row1id]) + "} "
				row1 += "\\\\\n"
				f.write(row1)
				f.write("\\cline{2-2} \\cline{4-11} \n")
				# f.write("\\hline \n")

				# row-2
				row2 = " \\multicolumn{1}{|c|}{} & 3  & \\multicolumn{1}{c|}{}  "
				for row2id in range(1, 9):
					row2 += " & " + " \\multicolumn{1}{c|}{ " + str(values_list[9 + row2id]) + "} "
				row2 += "\\\\\n"
				f.write(row2)
				f.write("\\cline{2-2} \\cline{4-11} \n")
				# f.write("\\hline \n")

				# row-3
				row3 = " \\multicolumn{1}{|c|}{} & 5  & \\multicolumn{1}{c|}{}  "
				for row3id in range(1, 9):
					row3 += " & " + " \\multicolumn{1}{c|}{ " + str(values_list[18 + row3id]) + "} "
				row3 += "\\\\\n"
				f.write(row3)
				f.write("\\hline \n")

				# avg-row
				avgrow = "\\multicolumn{2}{|c|}{Avg.} "
				for avgid in range(9):
					avgrow += " & " + " \\multicolumn{1}{c|}{ " + str(avg_list[avgid]) + "} "
				avgrow += "\\\\\n"
				f.write(avgrow)
				f.write("\\hline \n")

			# f.write("\\hline \n")
			# f.write("\\end{longtable}")
			f.write("\\end{tabular}\n")
			f.write("\\end{center} \n")
			f.write("\\end{table*}\n")
		# f.write("\\clearpage \n")
		# f.write("\\twocolumn \n")


def generate_supplementary_results_latex_code(filename, type):
	directory = 'supplementary_results_final'
	mkdir_p("{}/tex/".format(directory))
	file = os.path.join(directory, filename)
	print("Converting to file {} to latex".format(filename))
	if os.path.isfile(file):
		df = pd.read_csv(file, header=None)
		with open("supplementary_results_final/tex/{}_table.tex".format(type), "a") as f:
			f.write("\n")
			f.write("\\clearpage \n")
			f.write("\\onecolumn \n")
			# f.write("\\begin{table*}[t]\n")
			# f.write("\\begin{center} \n")
			# f.write("\\begin{tabular}{ |" + " | ".join(["c"] * len(df.columns)) + " |}\n")
			# f.write("\\begin{tabular}{ |cc|ccc|ccc|ccc| }\n")
			f.write("\\begin{longtable}{ |cc|ccc|ccc|ccc| }\n")
			f.write("\\hline \n")
			f.write(
				"\\multicolumn{1}{|c|}{ \\multirow{2}{*}{DATASET} } & \\multicolumn{1}{|c|}{ \\multirow{2}{*}{$\\epsilon$} } & \\multicolumn{3}{c|}{\\test} & \\multicolumn{3}{c|}{\\testa} & \\multicolumn{3}{c|}{\\testr} \\\\\n")
			f.write("\\cline{3-11} \n")
			# f.write("\\hline \n")
			f.write(
				" & \\multicolumn{1}{|c|}{}  & \\spn  & \\spnk & \\spng  & \\spn  & \\spnk & \\spng & \\spn & \\spnk & \\spng \\\\\n")
			f.write("\\hline \n")
			for i, row in df.iterrows():
				row_list = list(row.values)
				dataset_name = row_list[0]
				values_list = row_list[1:]

				avg_list = []
				for avgid in range(9):
					avg_list.append(
						round(np.mean([values_list[avgid], values_list[9 + avgid], values_list[18 + avgid]]), 2))

				# row-1
				row1 = "\\multicolumn{1}{|c|}{ \\multirow{3}{*}{" + str(
					dataset_name) + "}} & 1 & \\multicolumn{1}{|c|}{ \\multirow{3}{*}{ " + str(values_list[0]) + "}}"
				for row1id in range(1, 9):
					row1 += " & " + " \\multicolumn{1}{c|}{ " + str(values_list[row1id]) + "} "
				row1 += "\\\\\n"
				f.write(row1)
				f.write("\\cline{2-2} \\cline{4-11} \n")
				# f.write("\\hline \n")

				# row-2
				row2 = " \\multicolumn{1}{|c|}{} & 3  & \\multicolumn{1}{c|}{}  "
				for row2id in range(1, 9):
					row2 += " & " + " \\multicolumn{1}{c|}{ " + str(values_list[9 + row2id]) + "} "
				row2 += "\\\\\n"
				f.write(row2)
				f.write("\\cline{2-2} \\cline{4-11} \n")
				# f.write("\\hline \n")

				# row-3
				row3 = " \\multicolumn{1}{|c|}{} & 5  & \\multicolumn{1}{c|}{}  "
				for row3id in range(1, 9):
					row3 += " & " + " \\multicolumn{1}{c|}{ " + str(values_list[18 + row3id]) + "} "
				row3 += "\\\\\n"
				f.write(row3)
				f.write("\\hline \n")

				# avg-row
				avgrow = "\\multicolumn{2}{|c|}{Avg.} "
				for avgid in range(9):
					avgrow += " & " + " \\multicolumn{1}{c|}{ " + str(avg_list[avgid]) + "} "
				avgrow += "\\\\\n"
				f.write(avgrow)
				f.write("\\hline \n")

			# f.write("\\hline \n")
			f.write("\\end{longtable}")
			# f.write("\\end{tabular}\n")
			# f.write("\\end{center} \n")
			# f.write("\\end{table*}\n")
			f.write("\\clearpage \n")
			f.write("\\twocolumn \n")


def generate_paper_results_latex():
	for type in ["LL", "CLL"]:
		if type == "LL":
			generate_paper_results_latex_code("{}.csv".format(type), type)
		elif type == "CLL":
			for evidence_percentage in EVIDENCE_PERCENTAGES:
				generate_paper_results_latex_code("{}-{}.csv".format(type, evidence_percentage),
												  type="{}-{}".format(type, evidence_percentage))


def generate_supplementary_results_latex():
	for type in ["LL", "CLL"]:
		if type == "LL":
			generate_supplementary_results_latex_code("{}.csv".format(type), type)
		elif type == "CLL":
			for evidence_percentage in EVIDENCE_PERCENTAGES:
				generate_supplementary_results_latex_code("{}-{}.csv".format(type, evidence_percentage),
												  type="{}-{}".format(type, evidence_percentage))


# def generate_results_latex():
# 	for type in ["LL", "CLL"]:
# 		if type == "LL":
# 			generate_results_latex_code("{}.csv".format(type), type)
# 		elif type == "CLL":
# 			for evidence_percentage in EVIDENCE_PERCENTAGES:
# 				generate_results_latex_code("{}-{}.csv".format(type, evidence_percentage),
# 											type="{}-{}".format(type, evidence_percentage))


if __name__ == '__main__':
	# generate_csv()
	# generate_latex()
	# generate_results_csv()
	# generate_paper_results()
	# generate_paper_results_latex()
	generate_supplementary_results()
	generate_supplementary_results_latex()
# generate_results_latex()
