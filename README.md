# Idemia Card Buildup
Project_IDEMIA Card Buildup

Project hierarchy

```
├── problem                         # 
│   ├── main_problem.py             # Our problem representation for JMetal library
│   ├── problem_variables.py        # The file contains all problem's variables. 
│   └── utils.py                    # The file contains functions to process PLAN: BuildTotalPlan, OverallCost, Tardiness ...
├── README.md                       # This file
├── result                          # **need to be created** Folder containing the results of the algorithm 
├── run_metaheuristics.py           # Script for running algorithms.
└── unit_tests.py                   # The file contains unit tests for implemented functions.
```

1. Install **[JMetalPy](https://jmetal.github.io/jMetalPy/index.html)** library:

```sh
   $ pip install jmetalpy
```

2. Create **result** folder.

3. Run experiment:

**Note** : Before running the script, you need to specify/create experiment plans according to the samples [NSGAII_basic_exp_plan.csv, SMPSO_basic_exp_plan.csv].


Parameters for NSGAII

```sh

   N        - number of experiment (id)
   multiple - multiple
   strategy - strategy [RFU, not used]
   max_evaluations
   population_size
   offspring_population_size
   mutation_p
   crossover_p
   distribution_index

```


Parameters for SMPSO

```sh

   N        - number of experiment (id)
   multiple - multiple
   strategy - strategy [RFU - not used]
   max_evaluations
   swarm_size
   mutation_p
   distribution_index
   crowding_distance_archive_n
   c1_min
   c1_max
   c2_min
   c2_max
   r1_min
   r1_max
   r2_min
   r2_max
   min_weight
   max_weight
   change_velocity1
   change_velocity2

```


Parameters for `run_experiment.py`
  - **d** - Debug mode  [ True/False ] Default = False
  - **v** - Visualization  [ True/False ] Default = False
  - **s** - Random seed [if < 0 random seed otherwise setted by value]. Default = -1
  - **r** - Folder to save results [string]. Default 'test'
  - **fileplan_NSGAII** -  The file with NSGAII experiment plan [string]. Default = 'NSGAII_basic_exp_plan.csv'
  - **fileplan_SMPSO** - The file with SMPSO experiment plan [string]. Default = 'SMPSO_basic_exp_plan.csv'
  - **intermediate_results** - Save intermediate results - [True/False]. Default = True

```sh
   $ python3 run_experiment.py -r result_folder -s -1 --fileplan_NSGAII NSGAII_basic_exp_plan.csv --fileplan_SMPSO SMPSO_basic_exp_plan.csv --intermediate_results True
```

