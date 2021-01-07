import unittest
import numpy as np
from problem import utils
from problem import problem_variables
import copy
import logging

logger = logging.getLogger('utils')
logger.setLevel(logging.DEBUG)

# create a logging format
formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class TestProblemVariablesInit(unittest.TestCase):
    def test_1(self):
        self.Settings = copy.deepcopy(problem_variables.SETTINGS)
        utils.CheckSettings(self.Settings)


class TestInitialSequence(unittest.TestCase):
    def test_1(self):
        self.Settings = copy.deepcopy(problem_variables.SETTINGS)
        self.Plan = copy.deepcopy(problem_variables.PLAN)
        sequence = utils.GenerateRandomInitialSequence(self.Settings)
        print(sequence)
        utils.buildPlan(self.Settings, self.Plan, sequence)
        print(self.Plan['nl'])
        print(self.Plan['c'])
        print(self.Plan['l'])
        print(self.Plan['total_u'])
        print(self.Plan['half1_u'])
        print(self.Plan['half2_u'])
        print( utils.obj_evaluateSymmetry(self.Plan))
        print(utils.const_evaluateISOthickness(self.Settings, self.Plan))
        print(utils.const_evaluateMinStruct(self.Settings, self.Plan))

if __name__ == '__main__':
    unittest.main()
