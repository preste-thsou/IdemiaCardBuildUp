import numpy as np
import random
import logging
import copy
from problem import problem_variables
from math import *
import os


LOGGER = logging.getLogger('jmetal')

# Plan - Set of all decisions variables 
# Solution - special type for JMetal - lits[FloatSolutions] 


def SetupRandomSeed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)


def CheckSettings(Settings):
    """
    Check initial Settings
    """
    assert Settings['len_e'] == Settings['list_e'].shape[0]
    assert Settings['C_e_pc_lc_u'].shape == (Settings['list_type_e'],Settings['type_pc'], Settings['laser_pc'], Settings['list_u'].shape[0])
    assert Settings['IP_e'].shape == (Settings['len_e'], 4)
    assert Settings['r_e'].shape[0] == Settings['list_type_e']
    assert Settings['dir_e'].shape[0] == Settings['list_type_e']
    assert Settings['EX_ee'].shape == (Settings['list_type_e'], Settings['list_type_e'])
    for e in Settings['list_e']:
        assert e <= Settings['list_type_e'] - 1

def GetDataFromFileFUN(filename):
    """
    Get OBJECTIVES FUNCTIONS values from file
    return list
    """
    file = open(filename, 'r') 
    data = []
    for line in file.readlines() :
        data.append([float(x) for x in line.split()])
    return data


def GetFeasibleSolution(solutions):
    """
    Get feasible solution from population.
    return list[solution]
    """
    from jmetalPresteEdition.util import constraint_handling

    if type(solutions) is not list:
        solutions = [solutions]

    data = []
    for solution in solutions:
        if constraint_handling.is_feasible(solution):
            data.append(solution)
    return data


def TransformSolutionToArray(solutions):
    """
    Transform solution to list
    """

    if type(solutions) is not list:
        solutions = [solutions]

    data = []
    for solution in solutions:
        e = [function_value for function_value in solution.objectives ]
        data.append(e)

    return np.array(data)


def WriteERTDataToFile(ert_data, filename: str):
    """
    Write ERT_data to file
    """
 
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        LOGGER.info("Creation of the directory {} failed".format(filename))

    with open(filename, 'w') as of:
        for point in ert_data:
            for p_value in point:
                of.write(str(p_value) + ' ')
            of.write('\n')


def GenerateRandomInitialSequence(Settings):
    '''
    DEPRECIATED
    Generate initial random sequence of elements information : (l, type, thickness, x, y)
    l within nl_max, thickness within list_u, x and y


    :param Settings:
    :return:
    '''

    random_sequence_e = []
    seq = [(random.randrange(Settings['max_nl']), random.randrange(Settings['laser_pc']), random.randrange(Settings['list_u'].shape[0]), random.randrange(Settings['max_x']), random.randrange(Settings['max_y'])) for
           i in range(Settings['len_e'])]
    for quint in seq :
        random_sequence_e.append(quint[0])
        random_sequence_e.append(quint[1])
        random_sequence_e.append(quint[2])
        random_sequence_e.append(quint[3])
        random_sequence_e.append(quint[4])
    return random_sequence_e

def GenerateRandomInitialValidSequencev1(Settings):
    '''
    DEPRECIATED - missing constraint 4, and min boundaries of x,y
    :param Settings:
    :return:
    '''
    counter = 0
    while True:
        Plan = copy.deepcopy(problem_variables.PLAN)
        random_sequence_e = []
        counter +=1
        seq = [(random.randrange(Settings['max_nl']), random.randrange(Settings['laser_pc']),
                random.randrange(Settings['list_u'].shape[0]), random.randrange(Settings['max_x']),
                random.randrange(Settings['max_y'])) for
               i in range(Settings['len_e'])]
        for quint in seq:
            random_sequence_e.append(quint[0])
            random_sequence_e.append(quint[1])
            random_sequence_e.append(quint[2])
            random_sequence_e.append(quint[3])
            random_sequence_e.append(quint[4])
        buildPlan(Settings, Plan, random_sequence_e)
        if const_evaluateMinStruct(Settings, Plan)==0 and const_evaluateISOthickness(Settings, Plan)==0:
            return random_sequence_e
        else:
            if counter >= 1000:
                raise RuntimeError("not possible to generate one valid solution")

def GenerateRandomInitialValidSequencev2(Settings):
    counter = 0
    while True:
        Plan = copy.deepcopy(problem_variables.PLAN)
        random_sequence_e = []
        counter +=1
        for e in range(Settings['len_e']):
            type_e = Settings['list_e'][e]
            random_sequence_e.append(random.randrange(Settings['max_nl'])) #pure random for layer nb for element e
            if np.sum(Settings['C_e_pc_lc_u'][type_e,:,0,:]) > 0 and np.sum(Settings['C_e_pc_lc_u'][type_e,:,1,:]) > 0:
                laser_pc = random.randrange(Settings['laser_pc'])
            elif np.sum(Settings['C_e_pc_lc_u'][type_e,:,0,:]) > 0:
                laser_pc = 0
            else:
                laser_pc = 1
            random_sequence_e.append(laser_pc)
            _ , u = np.asarray(Settings['C_e_pc_lc_u'][type_e,:,laser_pc,:]== 1).nonzero()
            random_sequence_e.append(random.choice(u))
            random_sequence_e.append(random.randrange(Settings['min_x'], Settings['max_x']-Settings['IP_e'][e][2]))
            random_sequence_e.append(random.randrange(Settings['min_y'], Settings['max_y']-Settings['IP_e'][e][3]))

        buildPlan(Settings, Plan, random_sequence_e)
        if const_evaluateMinStruct(Settings, Plan) == 0 and const_evaluateISOthickness(Settings, Plan) == 0 and const_evaluateCEcomp(Settings, Plan) == 0:
            print(counter)
            return random_sequence_e
        else:
            if counter >= 100000:
                raise RuntimeError("not possible to generate one valid solution")


def SeqtoQuint(sequence):
    sequence2 = [(int(round(sequence[i-4])),int(round(sequence[i-3])),int(round(sequence[i-2])),int(round(sequence[i-1])), int(round(sequence[i]))) for i in range(4, len(sequence), 5)]
    return sequence2


def computeL(Settings, Plan):
    sorted_l = np.sort(Plan['AL_e'])
    rank = 0
    Plan['l'][sorted_l[0]][2] = rank
    Plan['l'][sorted_l[0]][3] = 1 #first layer is an overlay
    rank +=1
    for i in range(1, sorted_l.shape[0]):
        if sorted_l[i] != sorted_l[i-1]:
            Plan['l'][sorted_l[i]][2] = rank
            rank +=1
    for i in range(0, Settings['max_nl']):
        if Plan['l'][i][2] == Plan['nl'] - 1:
             Plan['l'][i][3] = 1 #last layer is an overlay

def computeTotalThickness(Settings, Plan):
    Plan['total_u_min'] = np.sum(Plan['l'][:,0])*Settings['factor_comp_min']
    Plan['total_u_max'] = np.sum(Plan['l'][:, 0]) * Settings['factor_comp_max']

def const_evaluateISOthickness(Settings, Plan):
    '''
    Validate constraint 1 : total doc thickness is within ISO
    :param Settings:
    :param Plan:
    :return:
    '''
    if Plan['total_u_max'] >= Settings['min_ISO_u'] and Plan['total_u_min'] <= Settings['max_ISO_u']:
        return 0
    else:
        return 1

def const_evaluateCEcomp(Settings, Plan):
    '''
    validate constraint 4 : all e are positioned in compatible layers
    :param Settings:
    :param Plan:
    :return:
    '''
    check = True
    for e in range(Settings['len_e']):
        l_e = Plan['l'][Plan['AL_e'][e]]
        index_u = np.nonzero(Settings['list_u'] == l_e[0])[0][0]
        if Settings['C_e_pc_lc_u'][Settings['list_e'][e]][l_e[3]][l_e[1]][index_u] == 0:
            check = False
            break

    if check:
        return 0
    else:
        return 4

def const_evaluateMinStruct(Settings, Plan):
    '''
    validate constraint 2 + 3 : sufficient nb of layers of PC + no e out of boundaries
    :param Settings:
    :param Plan:
    :return:
    '''
    const = 0
    if Plan['nl'] < 3:
        const = 2
    for e in range(Settings['len_e']):
        if Plan['P_e'][e][0] + Plan['P_e'][e][2] > Settings['max_x'] or Plan['P_e'][e][1]+ Plan['P_e'][e][3] > Settings['max_y']:
            const = 2
            break
    return const

def buildPlan(Settings, Plan, sequence):
    '''
    Build decision variables from problem variables and population.
    Only thickness of the first ocurrence of a given layer l will be considered
    Non-used layers l will be disregarded as far as the doc structure is concerned
    x,y will be overwritten depending on IP_e
    :param Settings:
    :param Plan:
    :param sequence:
    :return:
    '''

    for e, (l, lc, thickness, x, y) in enumerate(SeqtoQuint(sequence)):
        Plan['AL_e'][e] = l
        if Plan['l'][l][0] == 0:
            Plan['nl'] += 1
            Plan['l'][l][0] = Settings['list_u'][thickness]
            Plan['l'][l][1] = lc
        if Settings['IP_e'][e][0] == 0 and Settings['IP_e'][e][1] == 0:
            Plan['P_e'][e][0] = x
            Plan['P_e'][e][1] = y
    Plan['c'] = Plan['nl']/2
    computeL(Settings, Plan)
    computeTotalThickness(Settings, Plan)
    computeSymmetry(Settings, Plan)

def computeSymmetry(Settings, Plan):
    for l in range(Settings['max_nl']):
        if Plan['l'][l][2] < floor(Plan['c']):
            Plan['half1_u'] += Plan['l'][l][0]
        elif Plan['l'][l][2] >= ceil(Plan['c']):
            Plan['half2_u'] += Plan['l'][l][0]


def obj_evaluateSymmetry(Plan):
    return abs( Plan['half2_u']-Plan['half1_u'])

def obj_evaluteMinStruct(Settings, Plan):
    return Plan['nl']


def computeIntersectArea(Settings, Plan, e, ee):
    min_ex = Plan['P_e'][e][0] - Settings['r_e'][Settings['list_e'][e]]
    min_eex =Plan['P_e'][ee][0] - Settings['r_e'][Settings['list_e'][ee]]
    max_ex = Plan['P_e'][e][0]+ Plan['P_e'][e][2] + Settings['r_e'][Settings['list_e'][e]]
    max_eex = Plan['P_e'][ee][0]+ Plan['P_e'][ee][2] + Settings['r_e'][Settings['list_e'][ee]]
    min_ey = Plan['P_e'][e][1] - Settings['r_e'][Settings['list_e'][e]]
    min_eey = Plan['P_e'][ee][1] - Settings['r_e'][Settings['list_e'][ee]]
    max_ey = Plan['P_e'][e][1] + Plan['P_e'][e][3] + Settings['r_e'][Settings['list_e'][e]]
    max_eey = Plan['P_e'][ee][1] + Plan['P_e'][ee][3] + Settings['r_e'][Settings['list_e'][ee]]

    delta_x = 0
    delta_y = 0
    list_x = [min_ex, max_ex, min_eex, max_eex]
    list_y = [min_ey, max_ey, min_eey, max_eey]
    sortedlist_x = sorted(list_x)
    sortedlist_y = sorted(list_y)

    if sortedlist_x == list_x or sortedlist_x == [min_eex, max_eex, min_ex, max_ex] or sortedlist_y == list_y or sortedlist_y == [min_eey, max_eey, min_ey, max_ey]:
        return 0
    else:
        if sortedlist_x == [min_ex, min_eex, max_ex, max_eex]:
            delta_x = max_ex - min_eex
        if sortedlist_x == [min_eex, min_ex, max_eex, max_ex]:
            delta_x = max_eex - min_ex
        if sortedlist_x == [min_ex,min_eex, max_eex, max_ex]:
            delta_x = max_eex - min_eex
        if sortedlist_x == [min_eex, min_ex, max_ex, max_eex]:
            delta_x = max_ex - min_ex
        if sortedlist_y == [min_ey, min_eey, max_ey, max_eey]:
            delta_y = max_ey - min_eey
        if sortedlist_y == [min_eey, min_ey, max_eey, max_ey]:
            delta_y = max_eey - min_ey
        if sortedlist_y == [min_ey,min_eey, max_eey, max_ey]:
            delta_y = max_eey - min_eey
        if sortedlist_y == [min_eey, min_ey, max_ey, max_eey]:
            delta_y = max_ey - min_ey
        return delta_x * delta_y


def obj_minimizeEinteraction(Settings, Plan):
    interact_value = 0
    doc_surface = (Settings['max_x'] - Settings['min_x']) * (Settings['max_y'] - Settings['min_y'])
    for e in range(Settings['len_e']):
        for ee in range(Settings['len_e']):
            if ee == e:
                pass
            elif Settings['EX_ee'][Settings['list_e'][e]][Settings['list_e'][ee]] == 1:
                if Settings['IP_e'][e][0] == -1:
                    if Plan['l'][Plan['AL_e'][e]][2] == Plan['l'][Plan['AL_e'][ee]][2]:
                        interact_value += doc_surface
                elif Settings['dir_e'][Settings['list_e'][e]] == 2:
                    interact_value += computeIntersectArea(Settings, Plan, e, ee)
                elif Settings['dir_e'][Settings['list_e'][e]] == 0 and (
                        Plan['l'][Plan['AL_e'][ee]][2] >= Plan['l'][Plan['AL_e'][e]][2] >= Plan['c'] or
                        Plan['c'] >= Plan['l'][Plan['AL_e'][e]][2] >= Plan['l'][Plan['AL_e'][ee]][2]):
                    interact_value += computeIntersectArea(Settings, Plan, e, ee)
                elif Settings['dir_e'][Settings['list_e'][e]] == 1 and Plan['l'][Plan['AL_e'][e]][2] >= Plan['c'] and \
                        Plan['l'][Plan['AL_e'][e]][2] >= Plan['l'][Plan['AL_e'][ee]][2]:
                    interact_value += computeIntersectArea(Settings, Plan, e, ee)
                elif Settings['dir_e'][Settings['list_e'][e]] == 1 and Plan['l'][Plan['AL_e'][e]][2] <= Plan['c'] and \
                        Plan['l'][Plan['AL_e'][e]][2] <= Plan['l'][Plan['AL_e'][ee]][2]:
                    interact_value += computeIntersectArea(Settings, Plan, e, ee)
    Plan['interaction'] = interact_value
    if 185599 < interact_value < 185601 :
        print("I'm here")
        pass
    return interact_value
