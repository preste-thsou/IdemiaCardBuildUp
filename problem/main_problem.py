from math import sqrt, pow, sin, pi, cos

from jmetal.core.problem import FloatProblem, IntegerProblem
from jmetal.core.solution import FloatSolution, IntegerSolution
from problem import utils
from problem import problem_variables

import copy
import numpy as np
import json


class MainProblem(FloatProblem):
    """ 
    MainProblem.
        
    """

    def __init__(self, multiple = 2, strategy = 1):
        """ 
            :param number_of_variables: Number of decision variables of the problem.
            :param multiple: factor of multiplication for the size of exact sequence.
            :param strategy: strategy for generating solution and building Timeplan. 1 - Deterministic approach. 2 - Sequential approach  
        """
        super(MainProblem, self).__init__()

        self.multiple = multiple

        # Get and check initial Settings   
        self.Settings = copy.deepcopy(problem_variables.SETTINGS)
        utils.CheckSettings(self.Settings)
             
        self.number_of_objectives = 3
        self.number_of_constraints = 3


        #self.number_of_variables = 2*round(self.Settings['jmax'] * self.Settings['omax']*self.multiple)
        #self.number_of_variables = self.Settings['max_nl']*self.Settings['laser_pc']*self.Settings['list_u'].shape[0]*self.Settings['max_x']*self.Settings['max_y']
        self.number_of_variables = self.Settings['len_e']*5
        self.lower_bound = self.Settings['len_e']*[0,0,0,self.Settings['min_x'], self.Settings['min_y']]
        self.upper_bound = self.Settings['len_e']*[self.Settings['max_nl']-1,self.Settings['laser_pc']-1, self.Settings['list_u'].shape[0]-1, self.Settings['max_x'],self.Settings['max_y']]

        # Additional variables

        self.total_eval = 0
        self.total_rejected_const1 = 0
        self.total_rejected_const2 = 0
        self.total_rejected_const4 = 0
        self.min_l = np.inf
        self.min_symmetry = np.inf
        self.min_interact = np.inf
        self.ert_data = []
        
        print('--Main Problem--\n')


    def create_solution(self) -> FloatSolution:
        # print("-create_solution-")
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)

        new_solution.variables = utils.GenerateRandomInitialValidSequencev2(self.Settings)

        return new_solution


    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        
        # print("-evaluate-")
        self.total_eval += 1
        show = False
        Plan = copy.deepcopy(problem_variables.PLAN)
        

        utils.buildPlan( self.Settings, Plan, solution.variables,)


        if self.__evaluate_constraints(solution, Plan):
            # Calculate objectives functions
            solution.objectives[0] = utils.obj_evaluateSymmetry(Plan)
            solution.objectives[1] = utils.obj_evaluteMinStruct(self.Settings, Plan)
            solution.objectives[2] = utils.obj_minimizeEinteraction(self.Settings, Plan)

            #if Plan['AL_e'][0] == Plan['AL_e'][1] and Plan['nl']==4:
            #    print('there is one valid solution with offset and silkscreen with 4 layers where interaction value is {}'.format(Plan['interaction']))

            new_opt = False
            if solution.objectives[0] < self.min_symmetry:
                self.min_symmetry = solution.objectives[0]
                new_opt = True

            if solution.objectives[1] < self.min_l:
                self.min_l = solution.objectives[1]
                new_opt = True

            if solution.objectives[2] < self.min_interact:
                self.min_interact = solution.objectives[2]
                new_opt = True

            if new_opt:
                self.ert_data.append((self.total_eval, self.min_symmetry, self.min_l, self.min_interact))
        
        return solution


    def __evaluate_constraints(self, solution: FloatSolution, Plan) -> bool:
        #print("-__evaluate_constraints-")
        constraints = True
        constraint1 = utils.const_evaluateISOthickness(self.Settings,Plan)
        constraint2 = utils.const_evaluateMinStruct(self.Settings, Plan)
        constraint4 = utils.const_evaluateCEcomp(self.Settings, Plan)

        if constraint1 == 1:
            self.total_rejected_const1 += 1
            solution.constraints[0] = -1
            constraints = False

        if constraint2 == 2:
            self.total_rejected_const2 += 1
            solution.constraints[1] = -1
            constraints = False

        if constraint4 == 4:
            self.total_rejected_const4 += 1
            solution.constraints[2] = -1
            constraints = False

        if constraints:
            # Constraint not violated
            solution.constraints[0] = 1
            solution.constraints[1] = 1
            solution.constraints[2] = 1
        
        return constraints

    def get_name(self):
        return 'main problem'