from jmetal.util.solution import get_non_dominated_solutions
from problem import utils

import numpy as np
import csv 


def GetMaxMinObjectives(data):
	"""
	Calculate min and max values for all objectives functions.
	return 2 np.array with min and max values for all objectives functions: [0] symmetry; [1] nb of layer
	
	"""
	min_data = np.min(data, axis = 0)
	max_data = np.max(data, axis = 0)
	
	return min_data, max_data


def QualityMetric(fronts):
	"""
	Quality Metric (QM)

	The quality criterion is one of the most critical metrics to compare multi-objective
	meta-heuristic algorithms’ answers. Pareto set of solutions for all algorithms are
	compared to determine the quality metric, and those dominated by solutions of the other
	algorithm are removed. Then, each algorithm's quality metric is determined by dividing
	the cardinal of the remaining set of Pareto-optimal solutions to the cardinal of the
	original set of Pareto optimal solutions. 

	The higher the quality metric, the better the
	performance of the algorithm (Nemati et al., 2019).
	"""

	front_np_list = []
	total_front = []

	for i, front in enumerate(fronts):
		front_np_list.append(utils.TransformSolutionToArray(front))
		for solution in front:
			total_front.append(solution)

	best_front_np = utils.TransformSolutionToArray(get_non_dominated_solutions(total_front))

	QM = list(map(lambda x: sum( [x[i] in best_front_np for i,v in enumerate(x)] )/ len(x), front_np_list))
	return QM
	

def MeanIdealDistance(front):
	"""
	Mean Ideal Distance (MID)

	The mean ideal distance metric measures the relative distance of Pareto-optimal
	solutions from the ideal solution. The ideal solution is explained as the solution that
	individually optimizes all objective functions of the problem. 

	The lower the MID, the better the
	performance of the algorithm in terms of MID.
	"""
	front_np = utils.TransformSolutionToArray(front)
	min_data, max_data = GetMaxMinObjectives(front_np)
	f_symmetry_min, f_layer_min, f_interact_min = min_data
	f_symmetry_max, f_layer_max, f_interact_max = max_data
	f_symmetry_best, f_layer_best, f_interact_best = f_symmetry_min, f_layer_min,f_interact_min

	n = front_np.shape[0]

	symmetry_denominator = f_symmetry_max - f_symmetry_min
	layer_denominator = f_layer_max - f_layer_min
	interact_denominator = f_interact_max - f_interact_min

	if symmetry_denominator == 0 and layer_denominator == 0 and interact_denominator == 0:
		rerunt = -1

	if symmetry_denominator == 0:
		tardiness_denominator = 1


	if layer_denominator == 0:
		layer_denominator = 1

	if interact_denominator == 0:
		interact_denominator = 1
	
	MID = np.sum( np.sqrt( ( (front_np[:,0] - f_symmetry_best)/(symmetry_denominator) )**2 + ( (front_np[:,1] - f_layer_best)/(layer_denominator) )**2 + ( (front_np[:,1] - f_interact_best)/(interact_denominator) )**2 ) ) / n
	return MID


def DiversificationMetric(front):
	"""
	Diversification Metric (DM)

	The diversification metric shows the diversity of solutions obtained by metaheuristic
	algorithms. Equation (2) is used to calculate this performance metric for a biobjective
	optimization problem.

	DM’s higher values indicate the algorithm's better performance in terms of this
	performance metric (Nemati et al., 2019).

	"""
	front_np = utils.TransformSolutionToArray(front)
	min_data, max_data = GetMaxMinObjectives(front_np)
	
	DM = np.sqrt( np.sum( (max_data - min_data)**2 ) )
	return DM


def NumberOfParetoOptimalSolutions(front):
	"""
	Number of Pareto-optimal Solutions (NPS). This metric is defined as the cardinal of the set of Paretooptimal solutions
	"""
	front_np = utils.TransformSolutionToArray(front)
	return front_np.shape[0]


def MeanStd(front):
	"""
	Calculate mean and std values for Pareto front (solutions)
	return 2 np.array with mean and std values for all objectives functions: [0] tardiness; [1] overallcost.
	"""

	front_np = utils.TransformSolutionToArray(front)

	front_std = np.std(front_np, axis=0) 
	front_mean = np.mean(front_np, axis=0)

	return front_mean, front_std


def CalculateAllMetrics(fronts, algorithms_name, drop_rate_S_list, drop_rate_O_list, drop_rate_const4_list, elapsed_times, seeds, filename):
	"""
	Calculate and save all available metrics.
	"""
	QM = QualityMetric(fronts)
	MID = []
	DM = []
	NPS = []

	# field names  
	fields = ['Name', 'Seed', 'Time_s', 'Drop_rate_const1', 'Drop_rate_const2', 'drop_rate_const4_list','Symmetry_deviation_front_mean', 'Symmetry_deviation_front_std', 'nb_layer_front_mean', 'nb_layer_front_std', 'QM_max', 'MID_min', 'DM_max', 'NPS_max']

	# data rows of csv file  
	result = []
	for i,front in enumerate(fronts):

		front_mean, front_std = MeanStd(front)
		result.append( [ algorithms_name[i], seeds[i], round(elapsed_times[i]), drop_rate_S_list[i],  drop_rate_O_list[i], drop_rate_const4_list[i],front_mean[0], front_std[0], front_mean[1], front_std[1], QM[i],MeanIdealDistance(front), DiversificationMetric(front), NumberOfParetoOptimalSolutions(front)])
		
	with open(filename, 'w') as f: 
		# using csv.writer method from CSV package 
		write = csv.writer(f)  
		write.writerow(fields) 
		write.writerows(result) 

	return result