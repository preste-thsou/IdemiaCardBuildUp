from jmetal.algorithm.multiobjective import NSGAII, OMOPSO
from jmetal.operator import SBXCrossover, PolynomialMutation, IntegerPolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.archive import NonDominatedSolutionsArchive
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.evaluator import MultiprocessEvaluator

from jmetalPresteEdition.algorithm.smpso import SMPSO
from jmetalPresteEdition.util.archive import CrowdingDistanceArchive
from jmetalPresteEdition.util.observer import WriteFrontToFileObserver
from jmetalPresteEdition.lab.visualization.plotting import Plot
from jmetalPresteEdition.util import constraint_handling
from problem.main_problem import MainProblem
from problem import problem_variables
from problem import utils
from visualization import plot_ERT, plot_fronts, WritePlan, GetListOfSolutionsVar


import logging
import argparse
import os
import metrics
import numpy as np
import time
import sys
import random
from shutil import copyfile

LOGGER = logging.getLogger('jmetal')


def run(problem, algorithm, result_folder, max_evaluations=1000, visualization = False, print_to_file = False ):

	LOGGER.info('{} is solving {} '.format(algorithm.get_name(), problem.get_name()))
	algorithm.observable.register(ProgressBarObserver(max=max_evaluations))
	if visualization:
		algorithm.observable.register(VisualizerObserver())
	algorithm.observable.register(WriteFrontToFileObserver(output_directory = result_folder, print_to_file = print_to_file))

	start_time = time.time()
	algorithm.run()
	end_time = time.time()

	solutions = algorithm.get_result()
	feasible_solutions = utils.GetFeasibleSolution(solutions)
	front = get_non_dominated_solutions(solutions)
	#Run time empirical cumulative density function
	ert_data = problem.ert_data
	utils.WriteERTDataToFile(ert_data, result_folder+'ERT_data.{}.{}'.format(algorithm.get_name(), problem.get_name()))

	drop_rate_const1 = 100*(problem.total_rejected_const1/problem.total_eval)
	drop_rate_const2 = 100*(problem.total_rejected_const2/problem.total_eval)
	drop_rate_const4 = 100*(problem.total_rejected_const4/problem.total_eval)

	LOGGER.info('[{}] - # Total evaluations : {:10d} '.format(algorithm.get_name(), problem.total_eval))
	LOGGER.info('[{}] - # Drop rate const1       : {:8.2f} %'.format(algorithm.get_name(), drop_rate_const1))
	LOGGER.info('[{}] - # Drop rate const2       : {:8.2f} %'.format(algorithm.get_name(), drop_rate_const2))
	LOGGER.info('[{}] - # Drop rate const4       : {:8.2f} %'.format(algorithm.get_name(), drop_rate_const4))
	LOGGER.info('[{}] - # DM                : {:8.2f}  '.format(algorithm.get_name(), metrics.DiversificationMetric(front)))
	LOGGER.info('[{}] - # MID               : {:8.2f}  '.format(algorithm.get_name(), metrics.MeanIdealDistance(front)))
	# save to files



	print_function_values_to_file(feasible_solutions, result_folder+'FUN_population.{}.{}'.format(algorithm.get_name(), problem.get_name()))
	print_function_values_to_file(front, result_folder+'FUN_front.{}.{}'.format(algorithm.get_name(), problem.get_name()))

	print_variables_to_file(feasible_solutions, result_folder+'VAR_population.{}.{}'.format(algorithm.get_name(), problem.get_name()))
	print_variables_to_file(front, result_folder+'VAR_front.{}.{}'.format(algorithm.get_name(), problem.get_name()))

	listofsolution = GetListOfSolutionsVar(result_folder+'VAR_front.{}.{}'.format(algorithm.get_name(), problem.get_name()))
	for i, s in enumerate(listofsolution):
		WritePlan(result_folder+'Plan_front.{}.{}.{}'.format(algorithm.get_name(), problem.get_name(),i),s)

	plot_front = Plot(title='Pareto front approximation', axis_labels=['Symmetry', 'Nb of layers'])
	plot_front.plot(feasible_solutions, front, label='{}-{}'.format(algorithm.get_name(), problem.get_name()), filename=result_folder+'{}-{}'.format(algorithm.get_name(), problem.get_name()), format='png')

	return feasible_solutions, front, ert_data, drop_rate_const1, drop_rate_const2, drop_rate_const4, end_time - start_time


def experiment(fileplan_NSGAII, fileplan_SMPSO,folder = '', seed = 1, visualization = False, print_to_file = False):

	NSGAII_exp_plan, SMPSO_exp_plan, result_folder = None, None, None
	# Save Settings
	np.save(folder+'SETTINGS.npy', problem_variables.SETTINGS)
	no_NSGAII = False
	no_SMPSO = False
	
	try:
		NSGAII_exp_plan = np.genfromtxt(fileplan_NSGAII, delimiter=',', skip_header=1)
		SMPSO_exp_plan = np.genfromtxt(fileplan_SMPSO, delimiter=',',  skip_header=1)

		if len(NSGAII_exp_plan.shape)== 1:
			if NSGAII_exp_plan.shape[0] != 0:
				NSGAII_exp_plan = NSGAII_exp_plan.reshape((1,8))
			else:
				no_NSGAII = True

		if len(SMPSO_exp_plan.shape) == 1:
			if SMPSO_exp_plan.shape[0] != 0:
				SMPSO_exp_plan = SMPSO_exp_plan.reshape((1,19))
			else:
				no_SMPSO = True


	except Exception as e:
		print('[experiment] {}'.format(e))


	all_front_list = []
	all_ert_data_list = []
	all_algo_name_list = []
	all_drop_rate_const1_list = []
	all_drop_rate_const2_list = []
	all_drop_rate_const4_list = []
	all_elapsed_time = []
	all_seed_list = []

	# NSGAII plan
	if not no_NSGAII:
		for e in range(NSGAII_exp_plan.shape[0]):
			N, multiple, strategy, max_evaluations, population_size, offspring_population_size, mutation_p, crossover_p, distribution_index = NSGAII_exp_plan[e]
			
			if population_size%2 == 1:
				population_size+=1

			if offspring_population_size%2 == 1:
				offspring_population_size+=1


			if seed > 0:
				seed_e = int(N)*seed		
			else:
				seed_e = np.random.randint(2**32-1)

			LOGGER.info('Current Random Seed : {}'.format(seed_e))
			utils.SetupRandomSeed(seed_e)
			
			problem = MainProblem(multiple = multiple, strategy = strategy)
			algorithm = NSGAII(
										problem=problem,
										population_size=int(population_size),
										offspring_population_size= int(offspring_population_size),
										mutation=PolynomialMutation(probability= mutation_p, distribution_index=int(distribution_index)),
										crossover=SBXCrossover(probability = crossover_p, distribution_index=int(distribution_index)),
										termination_criterion=StoppingByEvaluations(max_evaluations=int(max_evaluations))
									)
			try:
				result_folder = folder + algorithm.get_name() + '_' + str(int(N)) + '/'
				if not os.path.exists(result_folder):
					os.mkdir(result_folder)
			except OSError:
				LOGGER.info("Creation of the directory {} failed".format(result_folder))

			_, front, ert_data, drop_rate_const1, drop_rate_const2, drop_rate_const4, elapsed_time= run(problem, algorithm, max_evaluations=int(max_evaluations), result_folder = result_folder, visualization = visualization, print_to_file = print_to_file)

			# Collect data
			all_front_list.append(front)
			all_ert_data_list.append(ert_data)
			all_algo_name_list.append( algorithm.get_name() + '_' + str(int(N)))
			all_drop_rate_const1_list.append(drop_rate_const1)
			all_drop_rate_const2_list.append(drop_rate_const2)
			all_drop_rate_const4_list.append(drop_rate_const4)
			all_elapsed_time.append(elapsed_time)
			all_seed_list.append(seed_e)

			# Calculate metrics
			metrics.CalculateAllMetrics(all_front_list, all_algo_name_list, all_drop_rate_const1_list, all_drop_rate_const2_list, all_drop_rate_const4_list, all_elapsed_time, all_seed_list, folder + 'metrics.csv')
			# Save logs
			copyfile('jmetalpy.log', folder+'jmetalpy.log')

	# SMPSO plan
	if not no_SMPSO:
		for e in range(SMPSO_exp_plan.shape[0]):
			N , multiple, strategy, max_evaluations, swarm_size, mutation_p, distribution_index, crowding_distance_archive_n, c1_min, c1_max, c2_min, c2_max, r1_min, r1_max, r2_min, r2_max, min_weight, max_weight, change_velocity1, change_velocity2 = SMPSO_exp_plan[e]
			
			if seed > 0:
				seed_e = int(N)*seed		
			else:
				seed_e = np.random.randint(2**32-1)

			LOGGER.info('Current Random Seed : {}'.format(seed_e))
			utils.SetupRandomSeed(seed_e)
			
			problem = MainProblem(multiple = multiple, strategy = strategy)
			algorithm = SMPSO(
										problem=problem,
										swarm_size=int(swarm_size),
										mutation=PolynomialMutation(probability=mutation_p, distribution_index=int(distribution_index)),
										leaders=CrowdingDistanceArchive(int(crowding_distance_archive_n)),
										termination_criterion=StoppingByEvaluations(max_evaluations=int(max_evaluations)),
										c1_min = c1_min, # 1.5  
										c1_max = c1_max, # 2.5  
	  									c2_min = c2_min, # 1.5
										c2_max = c2_max, # 2.5
										r1_min = r1_min, # 0
										r1_max = r1_max,  # 1
										r2_min = r2_min, # 0
										r2_max = r2_max,  # 1
										min_weight = min_weight, # 0.1
										max_weight = max_weight, # 0.1
										change_velocity1 = change_velocity1, # -1 
										change_velocity2 = change_velocity2  # -1 
									)

			try:
				result_folder = folder + algorithm.get_name() + '_' + str(int(N)) + '/'
				if not os.path.exists(result_folder):
					os.mkdir(result_folder)
			except OSError:
				LOGGER.info ("Creation of the directory {} failed".format(result_folder))

			_, front, ert_data, drop_rate_const1, drop_rate_const2, drop_rate_const4, elapsed_time  = run(problem, algorithm, max_evaluations=int(max_evaluations), result_folder = result_folder, visualization = visualization, print_to_file = print_to_file)

			# Collect data
			all_front_list.append(front)
			all_ert_data_list.append(ert_data)
			all_algo_name_list.append( algorithm.get_name() + '_' + str(int(N)) )
			all_drop_rate_const1_list.append(drop_rate_const1)
			all_drop_rate_const2_list.append(drop_rate_const2)
			all_drop_rate_const4_list.append(drop_rate_const4)
			all_elapsed_time.append(elapsed_time)
			all_seed_list.append(seed_e)

			# Calculate metrics
			metrics.CalculateAllMetrics(all_front_list, all_algo_name_list, all_drop_rate_const1_list, all_drop_rate_const2_list, all_drop_rate_const4_list, all_elapsed_time, all_seed_list, folder + 'metrics.csv')
			# Save logs
			copyfile('jmetalpy.log', folder+'jmetalpy.log')

	# Plot and save fronts and ERTs comparison.
	plot_ERT(all_ert_data_list, all_algo_name_list, folder)
	plot_fronts(all_front_list, all_algo_name_list, folder)

	# Try to remove logfile
	try:
		os.remove("jmetalpy.log") 
	except:
		LOGGER.info("Can't delete file jmetalpy.log")


def parse_args():
	""" Parse input arguments """
	parser = argparse.ArgumentParser(description='MetaHeuristic arguments for experiment')

	parser.add_argument('-d', '--debug', dest='debug', help='run un debug mode - True/False', type=bool, default=False)
	parser.add_argument('-v', '--vis', dest='visualization', help='Visualization of the process - True/False', type=bool, default=False)
	parser.add_argument('-s', '--seed', dest='seed', help='Random seed ', type=int, default=-1)
	parser.add_argument('-r', '--result_folder', dest='result_folder', help='Folder with result ', type=str, default='test')
	parser.add_argument('--fileplan_NSGAII', dest='fileplan_NSGAII', help='The file with NSGAII experiment plan', type=str, default='NSGAII_basic_exp_plan.csv')
	parser.add_argument('--fileplan_SMPSO', dest='fileplan_SMPSO', help='The file with SMPSO experiment plan ', type=str, default='SMPSO_basic_exp_plan.csv')
	parser.add_argument('--intermediate_results', dest='intermediate_results', help='Save intermediate results - True/False', type=bool, default=True)

	return parser.parse_args()



if __name__ == "__main__":

	args = parse_args()
	if args.debug:
		LOGGER.setLevel(logging.DEBUG)

	result_folder = 'result/'
	try:
		result_folder = 'result/' + args.result_folder 
		if not os.path.exists(result_folder):
			os.mkdir(result_folder)
	except OSError:
		LOGGER.info("Creation of the directory {} failed".format(result_folder))

	# start experiment
	experiment(fileplan_NSGAII = args.fileplan_NSGAII, fileplan_SMPSO = args.fileplan_SMPSO, folder = result_folder + '/', seed = args.seed, visualization = args.visualization, print_to_file = args.intermediate_results)
