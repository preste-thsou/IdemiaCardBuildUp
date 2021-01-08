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
    assert Settings['C_e_pc_lc_u'].shape == (Settings['len_e'],Settings['type_pc'], Settings['laser_pc'], Settings['list_u'].shape[0])


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
            random_sequence_e.append(random.randrange(Settings['max_nl'])) #pure random for layer nb for element e
            if np.sum(Settings['C_e_pc_lc_u'][e,:,0,:]) > 0 and np.sum(Settings['C_e_pc_lc_u'][e,:,1,:]) > 0:
                laser_pc = random.randrange(Settings['laser_pc'])
            elif np.sum(Settings['C_e_pc_lc_u'][e,:,0,:]) > 0:
                laser_pc = 0
            else:
                laser_pc = 1
            random_sequence_e.append(laser_pc)
            _ , u = np.asarray(Settings['C_e_pc_lc_u'][e,:,laser_pc,:]== 1).nonzero()
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
        if Settings['C_e_pc_lc_u'][e][l_e[3]][l_e[1]][index_u] == 0:
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
    min_ex = Plan['P_e'][e][0] - Settings['r_e'][e]
    min_eex =Plan['P_e'][ee][0] - Settings['r_e'][ee]
    max_ex = Plan['P_e'][e][0]+ Plan['P_e'][e][2] + Settings['r_e'][e]
    max_eex = Plan['P_e'][ee][0]+ Plan['P_e'][ee][2] + Settings['r_e'][ee]
    min_ey = Plan['P_e'][e][1] - Settings['r_e'][e]
    min_eey = Plan['P_e'][ee][1] - Settings['r_e'][ee]
    max_ey = Plan['P_e'][e][1] + Plan['P_e'][e][3] + Settings['r_e'][e]
    max_eey = Plan['P_e'][ee][1] + Plan['P_e'][ee][3] + Settings['r_e'][ee]

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
            if Settings['EX_ee'][e][ee] == 1:
                if Settings['IP_e'][e][0] == -1:
                    if Plan['l'][Plan['AL_e'][e]][2] == Plan['l'][Plan['AL_e'][ee]][2]:
                        interact_value += doc_surface
                elif Settings['dir_e'][e] == 2:
                    interact_value += computeIntersectArea(Settings, Plan, e, ee)
                elif Settings['dir_e'][e] == 0 and (
                        Plan['l'][Plan['AL_e'][ee]][2] >= Plan['l'][Plan['AL_e'][e]][2] >= Plan['c'] or
                        Plan['c'] >= Plan['l'][Plan['AL_e'][e]][2] >= Plan['l'][Plan['AL_e'][ee]][2]):
                    interact_value += computeIntersectArea(Settings, Plan, e, ee)
                elif Settings['dir_e'][e] == 1 and Plan['l'][Plan['AL_e'][e]][2] >= Plan['c'] and \
                        Plan['l'][Plan['AL_e'][e]][2] >= Plan['l'][Plan['AL_e'][ee]][2]:
                    interact_value += computeIntersectArea(Settings, Plan, e, ee)
                elif Settings['dir_e'][e] == 1 and Plan['l'][Plan['AL_e'][e]][2] <= Plan['c'] and \
                        Plan['l'][Plan['AL_e'][e]][2] <= Plan['l'][Plan['AL_e'][ee]][2]:
                    interact_value += computeIntersectArea(Settings, Plan, e, ee)
    Plan['interaction'] = interact_value
    return interact_value

def SumAByT(Plan,  Settings):
    """
    Sum A_tjio by t and devided by (ST_jio + P_jio)
    """

    # Sum A_tjio by t
    A_jio = np.sum(Plan['A_tjio'], axis = 0)
    #preinit
    A = np.zeros(Plan['A_tjio'].shape[1:] )
    ST_plus_P_jio = Settings['ST_jio'] + Settings['P_jio']
    #only divide nonzeros else 0
    np.divide(A_jio, ST_plus_P_jio, out=A, where=ST_plus_P_jio!=0)

    return A


def CalculateCompletionTime(Plan,  Settings):
    """
    Calculate completion time of operation o of job j on cell i
    C_jio

    """

    # Sum A_tjio by t
    A = SumAByT(Plan,  Settings)

    return  A * Settings['O_jio'] * (Plan['S_jio'] + Settings['ST_jio'] + Settings['P_jio'])


def CalculateTardiness(Plan,  Settings):
    """

    Calculate tardiness - objective function
    SUM_n_j (T_j × w_j)
    

    """
    T = -1
    # C_jio =  CalculateCompletionTime(Plan,  Settings)
    C_jio = Plan['C_jio']
    #Cj = max C_jio   
    #     io
    Cj = np.max(np.max(C_jio, axis = 1, keepdims=True) , axis = 2, keepdims=True)
    C_j = Cj[:,0,0]
    # Ad_j = C_j + θ_j + D_j
    Ad_j = C_j + Settings['Sh_j'] + Settings['D_j']
    
    # Tj = max (Adj – dj, 0)
    T_j = Ad_j - Settings['d_j']
    T_j[T_j < 0] = 0

    # SUM_n_j (T_j × w_j)
    T = np.sum(T_j*Settings['w_j'])
    return T


def CalculateOverallCost(Plan,  Settings):
    """

    Calculate OverallCost - objective function
    OverallCost =    cost of processing time 
                   + cost of setup time 
                   + cost of transfer within cells 
                   + cost of using material handling equipment

    """
    OverallCost = -1
    # Sum A_tjio by t
    A = SumAByT(Plan,  Settings)
    cost_of_processing_time = Settings['P_jio']*Settings['lamda']
    cost_of_setup_time =  Settings['ST_jio']*Settings['ro']
    cost_of_transfer_within_cells = Settings['TRT_jio']*Settings['tau']
    cost_of_using_material_handling_equipment = np.sum(np.sum(Plan['MH_lt'][:,:,0]>0, axis = (1))*Settings['delta'])
    OverallCost = np.sum( A*(cost_of_processing_time             \
                             + cost_of_setup_time                 \
                             + cost_of_transfer_within_cells
                            ) 
                        )                                         \
                    + cost_of_using_material_handling_equipment
    return OverallCost


def AreAllOperationsRealized(Plan, Settings):
    """
    Constraint 6
    All operations should be realized.
    Return TRUE if all operations was realized otherwise FALSE
    """
    A = SumAByT(Plan, Settings)
    all_realized_operations = np.sum(A)
    all_operations = np.sum( np.sum(Settings['ST_jio'] + Settings['P_jio'], axis = (1) ) > 0)
    return all_realized_operations == all_operations


def IsMHDistributedCorrectly(Plan, Settings):
    """
    Constraint 7

    Constraint on material handling equipment, check if material handling is distributed correctly
    Return TRUE and data for visualization if MH is distributed correctly otherwise FALSE
    """
    iprev = None
    iprime = None
    busyness = None
    data =  [[] for l in range(Settings['Beta'])] 
    
    for l in range(Settings['Beta']):
        t = 0

        if Plan['MH_lt'][l,t,0] > 0: #there is a movement at T0, from the initialized position of MH (which is a lost information in this case) to MH(l,0,1)
            iprime = Plan['MH_lt'][l,t,1]
            while Plan['MH_lt'][l,t,1] == iprime and Plan['MH_lt'][l,t,0] > 0:
                t += 1    

        while( t < np.max(Plan['C_jio'])):
            
            while Plan['MH_lt'][l,t,0] == 0 and t < np.max(Plan['C_jio']):
                t += 1
             
            if t == np.max(Plan['C_jio']):
                break  
            
            iprev = Plan['MH_lt'][l, t-1, 1]
            iprime = Plan['MH_lt'][l, t, 1]
            busyness = Plan['MH_lt'][l,t,0]

            if iprev == iprime:
                LOGGER.info("A transfer from a cell i to the same cell exists in MH - t : {},l: {}, iprev : {}, MH {}".format(t, l, iprev, Plan['MH_lt'][l, t-10:t+10,:]))
                return False, -1    
            
            for tt in range(t, t + Settings['TR_ii'][iprev, iprime]):
                if Plan['MH_lt'][l,tt,1] != iprime or Plan['MH_lt'][l,tt,0] != busyness:
                    LOGGER.info("A transfer has been interrupted before the time needed for its completion - t : {},l: {}, iprev : {}, MH {}".format(t, l, iprev, Plan['MH_lt'][l, t-10:t+10,:]))
                    return False, -1
            data[l].append( [busyness, (t, iprev), (t+Settings['TR_ii'][iprev, iprime]-1, iprime)])
            t += Settings['TR_ii'][iprev, iprime]
    
    return True, data


def areMHandSandCcoherent(Plan, Settings):
    '''
    we check that for each job, when 2 operations are made on different cells, at least one equipment l is allocated to the transfer
    To do this, we check that for each (j,o,i) where o> 0, if S_jio(o,i,j) != tmax and S_jio(o-1,i,j) == tmax, one of the equipment l should be going at i between Cj?o-1 and Sjio
    versus operation start.
    If IsMHDistributedCorrectly is OK, then this condition is sufficient to check coherence between A and MH 
    '''

    for j in range(Settings['jmax']):
        for o in range(1, Settings['omax']):
            if Plan['AL'][j, o] != Settings['imax'] and Plan['AL'][j, o] != Plan['AL'][j, o-1]:
                check = False
                prev_i = Plan['AL'][j,o-1]
                i = Plan['AL'][j,o]
                for l in range(Settings['Beta']):
                    for t in range(Plan['C_jio'][j,prev_i,o-1], Plan['S_jio'][j,i,o]):
                        if Plan['MH_lt'][l,t,1] == i and Plan['MH_lt'][l,t,0] == 1:
                            check = True
                if not(check):
                    LOGGER.info("No equipement is taking care of operation j:{}, o:{}, i: {}".format(j,o,i))
                    return False
    return True 


def GenerateSequenceJobOperation(Settings):
    """
    Generate sequence job-operation respecting order of operations: (j,o1), (j,o2), (j,o3), ... 
    """

    Gj= [[] for j in range(Settings['jmax'])]
    for j in range(Settings['jmax']):
        nb_o = np.sum(np.sum(Settings['O_jio'][:,:,:] , axis = (1))[j,:] > 0)
        Git = [i for i in range(nb_o)]
        Gj[j] = iter(Git)
    return Gj


def GenerateRandomSequenceJobCell(Settings, multiple = 2):
    """
    Genetare random sequence of potential solution: (j1,i1), (j2,i1), ...
    return list
    """
    random_sequence_ji = []
    s_ji = [(random.randrange(Settings['jmax']), random.randrange(Settings['imax'])) for i in range( round(Settings['jmax']*Settings['omax']*multiple) )]
    for pair in s_ji:
        random_sequence_ji.append(pair[0])
        random_sequence_ji.append(pair[1])

    return random_sequence_ji


def GenerateRandomSequenceJobCellMH(Settings, multiple = 2):
    """
    Genetare random sequence of potential solution: (j1,r1,l1), (j2,r2,l2), ...
    return list
    """

    random_sequence_jil = []
    s_jil = [(random.randrange(Settings['jmax']), random.randrange(Settings['imax']), random.randrange(Settings['Beta'])) for i in range( round(Settings['jmax']*Settings['omax']*multiple) )]
    for triplet in s_jil:
        random_sequence_jil.append(triplet[0])
        random_sequence_jil.append(triplet[1])
        random_sequence_jil.append(triplet[2])

    return random_sequence_jil


def ExtractFirstSeqFromRandomSequenceJobCellv2(sequence, Settings):
    """
    Generate (j,o,i), (j,o,i), (j,o,i) , ... from a sequence of (j,r) (where r is the rank of cell i to chose among authorized cells for (j,o)
    """

    sequence = [(int(round(sequence[i-1])), int(round(sequence[i]))) for i in range(1,len(sequence), 2)]
    Gj = GenerateSequenceJobOperation(Settings)


    result = []
    i= 0
    ii = 0
    nb_of_all_operations = np.sum(np.sum(Settings['O_jio'] , axis = (1)) > 0)
    while i < nb_of_all_operations:
        try:
            j = sequence[ii][0]
            r = sequence[ii][1]
            o = next(Gj[j])
            
            possible_i =  np.where(Settings['O_jio'][j,:,o] == 1)[0]
            index = r%len(possible_i)
            
            #               j  o          i
            result.append( (j, o , possible_i[index] ))
            i = i+1
            ii = ii+1
        except StopIteration:
            ii = ii+1
            pass
        except ZeroDivisionError:
            LOGGER.info("[ExtractFirstSeqFromRandomSequenceJobCellv2] ZeroDivisionError")
            pass
        except Exception as e:
            LOGGER.debug("ExtractFirstSeqFromRandomSequenceJobCellv2: Not enough elements in the sequence : increase multiple\nError: {}\n".format(e))
            return result, False
    return result, True


def ExtractFirstSeqFromRandomSequenceJobCellMH(sequence, Settings):
    """
    Generate (j,o,i,l), (j,o,i,l), (j,o,i,l) , ... from a sequence of (j,r,l) (where r is the rank of cell i to chose among authorized cells for (j,o) and MH l allocated if a transportation needed 
    """

    sequence = [(int(round(sequence[i-2])), int(round(sequence[i-1])), int(round(sequence[i]))) for i in range(2,len(sequence), 3)]
    
    Gj = GenerateSequenceJobOperation(Settings)

    result = []
    i= 0
    ii = 0
    nb_of_all_operations = np.sum(np.sum(Settings['O_jio'] , axis = (1)) > 0)
    while i < nb_of_all_operations:
        try:
            j = sequence[ii][0]
            r = sequence[ii][1]
            l = sequence[ii][2]
            o = next(Gj[j])
            
            possible_i =  np.where(Settings['O_jio'][j,:,o] == 1)[0]
            index = r%len(possible_i)
            
            #               j  o          i           l
            result.append( (j, o , possible_i[index], l ))
            i = i+1
            ii = ii+1
        except StopIteration:
            ii = ii+1
            pass
        except ZeroDivisionError:
            LOGGER.info("[ExtractFirstSeqFromRandomSequenceJobCellMH] ZeroDivisionError")
            pass
        except Exception as e:
            LOGGER.debug("ExtractFirstSeqFromRandomSequenceJobCellMH: Not enough elements in the sequence : increase multiple\nError: {}\n".format(e))
            return result, False
    return result, True


def IsSequenceFeasible(sequence, Settings):
    """
    Check if a generated sequence can be totaly processed according to O_oji
    """
    for o in sequence:
        #                      j    i   o
        if Settings['O_jio'][o[0],o[2],o[1]] == 0:
            return False
    return True


def IsSequenceFeasible2Strategy1(sequence, Settings, i):
    """
    Check if from a generated sequence we can build valid time plan
    """
    Plan = copy.deepcopy(problem_variables.PLAN)
    code = BuildTotalPlanWithMHd(sequence, Plan, Settings)
    if code != 0:
        LOGGER.info('\n\n\n[IsSequenceFeasible2]  i : {} -- code : {}\n\n'.format(i, code))
        return False
    return True


def IsSequenceFeasible2Strategy2(sequence, Settings, i):
    """
    Check if from a generated sequence we can build valid time plan
    """
    Plan = copy.deepcopy(problem_variables.PLAN)
    code = BuildTotalPlanWithMHs(sequence, Plan, Settings)
    if code != 0:
        LOGGER.info('[IsSequenceFeasibleMH2]  i : {} -- code : {}'.format(i, code))
        return False
    return True


def GenerateSolutionStrategyV0(Settings, multiple = 2):
    """
    Return random generated sequence (j,i)
    """
    return GenerateRandomSequenceJobCell(Settings, multiple = multiple)


def GenerateSolutionStrategyV1(Settings, multiple = 2):
    """
    Return a sequence which has enough operations for each job.
    """
    i = 0
    while True:
        i+=1
        if i > 10000:
            exit('[GenerateSolutionStrategyV1]: Check the solution generation strategy v1')

        random_sequence_ji = GenerateRandomSequenceJobCell(Settings, multiple = multiple)
        sequence_joi, f = ExtractFirstSeqFromRandomSequenceJobCellv2(random_sequence_ji, Settings)
        if f:
            if IsSequenceFeasible(sequence_joi, Settings) and IsSequenceFeasible2Strategy1(random_sequence_ji, Settings, i):
                return random_sequence_ji
            else:
                continue


def GenerateSolutionStrategyV2(Settings, multiple = 2):
    """
    Return a sequence which has enough operations for each job.
    """
    i = 0
    while True:
        i+=1
        if i > 10000:
            exit('[GenerateSolutionStrategyV2]: Check the solution generation strategy v2')

        random_sequence_jil = GenerateRandomSequenceJobCellMH(Settings, multiple = multiple)
        sequence_joil, f = ExtractFirstSeqFromRandomSequenceJobCellMH(random_sequence_jil, Settings)
        if f:
            if IsSequenceFeasible(sequence_joil, Settings) and IsSequenceFeasible2Strategy2(random_sequence_jil, Settings, i):
                return random_sequence_jil
            else:
                continue


def BuildAllocationMatrixFromSequence(sequence, Settings):
    """
    Build Allocation Matrix From random generated sequence. Need to apply filter to check if operation exist for fob j.
    """

    AL = [[Settings['imax'] for o in range(Settings['omax'])] for j in range(Settings['jmax'])]
    for item in sequence:
        try:
            AL[item[0]][item[1]] = item[2]
        except:
            break
    return np.array(AL, dtype = np.int)


def ApplyMaskOfExistingOperationToAllocationMatrix(Plan, Settings):
    """
    If the operation for a job does not exist, put imax (as agreed)
    """

    ST_P_jio = (Settings['ST_jio'] + Settings['P_jio'])
    mask_for_o = (np.sum(ST_P_jio, axis=1) == 0).astype(np.int) # 1  - non exist , 0 - exist
    idx = np.where(mask_for_o == 1)
    Plan['AL'][idx] = Settings['imax']


def CellCapacityIsExceeded(o,j,i,t, Plan, Settings):
    """
    Constraint 3

    The number of assigned operations to start on cell i at time t cannot exceed its total number of machines

    SUM_oj [A_tjio] ≤ N_i;

    True: if doesn't satisfy the conditions 
    False: If satisfy the conditions

    """
    condition = None
    t_end = t + Settings['ST_jio'][j,i,o] + Settings['P_jio'][j,i,o] 

    CellCapacity_on_each_time_unit = np.sum(Plan['A_tjio'][t:t_end], axis = (1,3))

    # If the condition is violated, we obtain a value greater than 0
    condition = np.sum(CellCapacity_on_each_time_unit[:,i] >= Settings['N_i'][i])

    if condition == 0:
        return False
    else:
        return True


def UpdateTimePLAN(o,j,i,t, Plan, Settings):
    """
    Update TimePLAN
    """
    # S
    Plan['S_jio'][j,i,o] = t

    t_end = t + Settings['ST_jio'][j,i,o] + Settings['P_jio'][j,i,o] 
    # C
    Plan['C_jio'][j,i,o] = t_end
    # A
    if t_end >  Settings['t_max']:
        raise Exception('t_max is too small')

    Plan['A_tjio'][t:t_end,j,i,o] += 1


def BuildTotalPlan(sequence, Plan, Settings):
    """
    Build total plan based on constraints 
    !!! WITHOUT MH !!!
    """
    
    sequence, _ = ExtractFirstSeqFromRandomSequenceJobCellv2(sequence, Settings)
    if not _ :
        # drop because of a sequence element
        return 1 
    Plan['AL'] = BuildAllocationMatrixFromSequence(sequence, Settings)
    ApplyMaskOfExistingOperationToAllocationMatrix(Plan, Settings)


    for e, (j, o, i) in enumerate(sequence):
        if Settings['O_jio'][j, i, o] == 0:
            #drop because of O_jio
            return 2

        if o == 0:
            t = Settings['R_j'][j] + Settings['DT_j'][j]             #mininimum starting time at t = Rj + DTj
            while CellCapacityIsExceeded(o,j,i,t, Plan, Settings):   #we start o of job j on cell i as soon as cell i is not saturated
                t += 1
            UpdateTimePLAN(o,j,i,t, Plan, Settings)                  #once we know the earliest time t for (o,j) on i, we update S and compute C & A

        if o > 0:
            if Plan['AL'][j,o] == Settings['imax']:
                return 3     #test if there is an operation to be performed for job j /  #test if an operation o of job j can be performed at i 
            
            if Plan['AL'][j, o-1] == Plan['AL'][j, o]: #operation o continues after operation o-1 on cell i
                t = Plan['C_jio'][j,i, o-1] + Settings['TRT_jio'][j , i, o-1]

                while CellCapacityIsExceeded(o,j,i,t, Plan, Settings):  #we start o of job j on cell i as soon as cell i is not saturated
                    t+= 1
                UpdateTimePLAN(o,j,i,t, Plan, Settings)                 #once we know the earliest time t for (o,j) on i, we update S and compute C & A
            else: #operation o moves to another cell iprime
                iprime = Plan['AL'][j,o]
                i =  Plan['AL'][j, o-1]
                t = Plan['C_jio'][(j, i, o-1)] + Settings['TR_ii'][i,iprime] 
                while CellCapacityIsExceeded(o,j,iprime,t, Plan, Settings): #we start o of job j on cell iprime as soon as cell iprime is not saturated
                    t+=1
                UpdateTimePLAN(o,j,iprime,t, Plan, Settings)
    return 0


def replaceSeqMH(Plan, Settings,  l, initial_t, end_t):
    ''' 
                            initial_t                                    end_t
    Replace a sequence of   (1,i)        (2,ii)(2,ii)* (0,ii)* (2,iii)* (2,iii)(1,iiii)

    by :                    initial_t                        end_t           
                            (1,i)  (2,iii)(2,iii)* (0,iii)* (0,iii)(1,iiii) with how much (2,iii) needed to move MH_l from i to iii
                            
    we also check the transitivity of TR : moving from i to ii and then from ii to iii should take longer than moving from i to iii, otherwise it's better not to change the seq                        
    '''    
    i = Plan['MH_lt'][l,initial_t,1]
    ii = Plan['MH_lt'][l,initial_t+1,1]
    iii = Plan['MH_lt'][l,end_t,1]
    
    if( Settings['TR_ii'][i,ii] + Settings['TR_ii'][ii,iii] > Settings['TR_ii'][i,iii]):
        for t in range(Settings['TR_ii'][i,iii]):
            Plan['MH_lt'][l,initial_t + 1 + t,0] = 2
            Plan['MH_lt'][l,initial_t + 1 + t,1] = iii
        for t in range(initial_t+1 + Settings['TR_ii'][i,iii], end_t+1):
            Plan['MH_lt'][l,t,0] = 0
            Plan['MH_lt'][l,t,1] = iii


def optimizeMH(Plan, Settings):
    '''
    Remove unecessary movements of equipmments l on a finalized MH plan for each l independantly
    '''
    for l in range(Settings['Beta']):
        #                                                               initial_t                                  end_t        
        # loop on MH_lt(l) over t to find a sequence in the form of ...(1,i)        (2,ii)(2,ii)* (0,ii)* (2,iii)* (2,iii)(1,iiii)... where:
        #    i!= ii (but no need to check this first condition, it's always true if we move from (1,i) to (2,n)), ii!=iii, iii!=iiii
        t = 0

        tt_max = np.max(Plan['C_jio'])
        while(t < tt_max):
            while(Plan['MH_lt'][l,t,0] != 1 and Plan['MH_lt'][l,t+1,0] != 2 and t < (tt_max-2)):
                t+= 1
            
            if t == tt_max-1:
                break 
            
            ii = Plan['MH_lt'][l,t+1,1]  # we are at first position of (2,ii)  in the potential sequence
            initial_t = t
            t = t+1
            while( Plan['MH_lt'][l,t,1] == ii and Plan['MH_lt'][l,t,0] != 1 and t < (tt_max-1)): #there can be as many (2,ii) or (0,ii) in the potential sequence
                t = t+1
            
            if t == tt_max-1:
                break 
                
            if( Plan['MH_lt'][l,t,0] != 2 or Plan['MH_lt'][l,t,1]== ii ):  #the next item should be a (2,iii), otherwise we are not in the right sequence, restart the search starting from new t
                continue
            
            else:
                iii = Plan['MH_lt'][l,t,1]
                while( Plan['MH_lt'][l,t,1] == iii and Plan['MH_lt'][l,t,0] == 2 and t < (tt_max-1)):
                    t =t+1
                
                if t == tt_max-1:
                    break
                
                if( Plan['MH_lt'][l,t,1] != iii and Plan['MH_lt'][l,t,0] == 1): # we have found a sequence, between initial_t and t
                    replaceSeqMH(Plan, Settings, l, initial_t, end_t = t-1)
                else:
                    continue
    return 0


def allocateMH(Plan, Settings, l, i, iprime, initial_t, cas, relevant_t):
    """
    Put information to MH matrix based on case appeared
    """
    initial_i = Plan['MH_lt'][l, initial_t, 1]

    Plan['MH_lt'][l, initial_t : initial_t + Settings['TR_ii'][initial_i,i]] = (2, i) #equipment is busy but empty
    Plan['MH_lt'][l, initial_t + Settings['TR_ii'][initial_i,i] : initial_t + Settings['TR_ii'][initial_i,i]+ Settings['TR_ii'][i, iprime]] = (1, iprime) #equipment is busy and not empty
        
    if cas == 0: # in this case, at relevant_t, l will be busy carrying stuff from iprime to a new location. Need to feel the plan with "free in iprime" before this happens
        Plan['MH_lt'][l,initial_t + Settings['TR_ii'][initial_i,i]+ Settings['TR_ii'][i, iprime]: relevant_t] = (0, iprime) 
        
    elif cas == 1: # in this case, at relevant_t, l will be moved empty to a new location next_i, so we need to move it to next_i as soon as we can and fill the rest with "free in next_i"
        next_i = Plan['MH_lt'][l,relevant_t,1]
        t = initial_t + Settings['TR_ii'][initial_i,i]+ Settings['TR_ii'][i, iprime] + Settings['TR_ii'][iprime, next_i] 
        Plan['MH_lt'][l, initial_t + Settings['TR_ii'][initial_i,i]+ Settings['TR_ii'][i, iprime] : t] = (2, next_i)
        
        while t < relevant_t:  #if t smaller than relevant_t, we know that MH is filled with zeros until relevant_t, but with a location which may not be the proper one, so we set it to next_i
            Plan['MH_lt'][l,t] = (0, next_i)
            t +=1
        while Plan['MH_lt'][l,t,0] == 2: #if t equals or bigger than relevant_t, we need to remove the remaining (2, next_i) and feed with zeros
            Plan['MH_lt'][l,t] = (0, next_i)
            t +=1
    else:
        check_neverusedafter = np.sum( Plan['MH_lt'][l, initial_t + Settings['TR_ii'][initial_i,i] + Settings['TR_ii'][i, iprime]:Settings['t_max'], 0] )        
        
        if check_neverusedafter == 0:
            Plan['MH_lt'][l, initial_t + Settings['TR_ii'][initial_i,i]+ Settings['TR_ii'][i, iprime] : Settings['t_max']] = (0, iprime)
            #the new place of equipment l untill the end of the processing period is iprime
        else:
            final_i = Plan['MH_lt'][l, initial_t + Settings['TR_ii'][initial_i,i]+ Settings['TR_ii'][i, iprime]][1]
            Plan['MH_lt'][l, initial_t + Settings['TR_ii'][initial_i,i]+ Settings['TR_ii'][i, iprime] : initial_t + Settings['TR_ii'][initial_i,i]+ Settings['TR_ii'][i, iprime] + Settings['TR_ii'][iprime, final_i] ] = (2, final_i) #equipment is busy but empty
            #we bring back MH_l to it's original location because it's expected to be there in future


def computeMHv5s(i, iprime, t, l, Plan, Settings):
    ''' 
    Sequence based approach.
    Move material from i to iprime at the best available time for equipment l, considering earliest prod start at t
    Update MH_l using tuple (busyness, cell_value) where busyness can be 0 (not busy), 1 (busy and full) or 2 (busy and empty) and cell_value is the cell where MH_l is located or heading to
    '''
    
    #maximum transfer for any cell to any other one
    max_transfer_time = np.max(Settings['TR_ii'])
    tt = max(0,t - max_transfer_time)
    slot_allocated = False
    initial_i = None

    while not(slot_allocated) and tt < Settings['t_max']:      
        while Plan['MH_lt'][l,tt,0] > 0 and tt < Settings['t_max']: #we look for the next time where material handling l is free
            tt += 1

        if tt == Settings['t_max']:
            #no slot could be allocated before tmax was reached
            break

        needed_time_slot = Settings['TR_ii'][Plan['MH_lt'][l,tt,1], i] + Settings['TR_ii'][i, iprime] #the minimum time windows of availibility of l needed is the time to transport from is current point Settings['MH_lt'][l][tt][1] to i, then from i to iprimme

        is_free = True
        for ttt in range(tt, max( tt+needed_time_slot, t+Settings['TR_ii'][i,iprime] )): # max is needed to check that we do not get l too early
            if(Plan['MH_lt'][l,ttt, 0] != 0):
                is_free = False #the equipment l cannot be used for the required transfer because it's in use at least once during the needed time for the transfer
                tt = ttt        #tt is updated with the last non free timeslot


        if is_free == True:     #the equipment l is available for the transfer from i to iprime at earliest time tt. It will bring l in cell iprime, we need to check that it's compatible with the next timesteps of Settings['MH_lt'][l]
            ttt = max( tt+needed_time_slot, t+Settings['TR_ii'][i,iprime] )
            next_i = Plan['MH_lt'][l,ttt,1]

            for tttt in range( ttt, ttt+ max(Settings['TR_ii'][iprime, next_i], 1)):        #max needed if iprime == next_i, we still need to check the equipemnt is not busy moving from previous_i to next_i  ! 
                if Plan['MH_lt'][l,tttt,0] == 1 and Plan['MH_lt'][l,tttt-1, 1] == iprime:   #the equipment is used shortly after the needed_time_slot, but it's used with the assumption that the equipemnt origin is the location where we will bring it, so it' s ok.
                    best_t = max(tt, t-Settings['TR_ii'][Plan['MH_lt'][l,tt,1],i])          #earliest time at which we can start, without starting too soon vis-a-vis t (production start)
                    cas = 0
                    initial_i = Plan['MH_lt'][l, best_t, 1]
                    allocateMH(Plan,Settings, l, i, iprime, best_t, cas, relevant_t = tttt)
                    slot_allocated = True
                    break
                elif Plan['MH_lt'][l,tttt,0] == 1:
                    tt = tttt
                    is_free = False
                    break   #needed to exit previous for loop, slot cannot be used, equipment is needed from next_i before we have time to bring it there
                elif Plan['MH_lt'][l,tttt,0] == 2:  #there is an empty movement in the near future of the needed time slot (ie before we have time to bring l to next_i), so we need to check whether it can be compatible
                    next_next_i = Plan['MH_lt'][l,tttt,1]
                    compatible = True
                    for ttttt in range(tttt, max(tttt, ttt+ Settings['TR_ii'][iprime,next_next_i])): #next_next_i could be equal to iprime, in which case this is automatically compatible
                        if Plan['MH_lt'][l,ttttt,0] == 1:
                            compatible = False
                    
                    if compatible == False:
                        is_free = False
                        tt = tttt
                        break # we exit the previous for loop, slot cannot be used
                    else: # there is an empty movement shortly after needed_time_slot, we don't have time to bring l back to it's original place but we can change the future allocation to sthg compatible
                        best_t = max(tt, t-Settings['TR_ii'][Plan['MH_lt'][l,tt,1],i]) #earliest time at which we can start, without starting too soon vis-a-vis t (production start)
                        cas = 1 
                        initial_i = Plan['MH_lt'][l, best_t, 1]
                        allocateMH(Plan,Settings,l, i, iprime, best_t, cas, relevant_t = tttt)
                        slot_allocated = True
                        break
            if not slot_allocated and is_free: #slot is still potentially ok, and no allocation was made yet, so it means near future is filled with zeros, we can indeed use the slot
                best_t = max(tt, t-Settings['TR_ii'][Plan['MH_lt'][l,tt,1],i]) #earliest time at which we can start, without starting too soon vis-a-vis t (production start)
                cas = 2
                initial_i = Plan['MH_lt'][l, best_t, 1] 
                allocateMH(Plan,Settings,l, i, iprime, best_t, cas, relevant_t = best_t)
                slot_allocated = True

    if slot_allocated:
        if initial_i != None:
            return best_t + Settings['TR_ii'][initial_i,i] + Settings['TR_ii'][i, iprime]
        else:
            exit('[computeMHv5s] No initial_i')
    else:
        #no slot could be allocated before tmax was reached
        exit('[computeMHv5s] No slot could be allocated before tmax was reached') 


def computeMHv5d(i, iprime, t, Plan, Settings):
    ''' 
    Deterministic approach.
    Move material from i to iprime at the best available time for equipment l, considering earliest prod start at t
    Update MH_l using tuple (busyness, cell_value) where busyness can be 0 (not busy), 1 (busy and full) or 2 (busy and empty) and cell_value is the cell where MH_l is located or heading to
    '''
    #we init our best starting time matrix with 0 
    best_t = [0 for l in range(Settings['Beta'])] 
    #we init our best prod starting time matrix with t_max
    best_prod_start_time_t  = [Settings['t_max'] for l in range(Settings['Beta'])]
    # intialization
    cas = [int for l  in range(Settings['Beta'])]
    relevant_t = [int for l in range(Settings['Beta'])]
    slot_allocated = [bool for l in range(Settings['Beta'])]
    max_transfer_time = np.max(Settings['TR_ii']) #maximum transfer for any cell to any other one
    
    for l in range(Settings['Beta']):
        tt = max(0,t - max_transfer_time)
        slot_allocated[l] = False
        
        while not(slot_allocated[l]) and tt < Settings['t_max']:
            
            while Plan['MH_lt'][l,tt,0] > 0 and tt < Settings['t_max']: #we look for the next time where material handling l is free
                tt += 1
            
            if tt == Settings['t_max']:
                break

            needed_time_slot = Settings['TR_ii'][Plan['MH_lt'][l,tt,1], i] + Settings['TR_ii'][i, iprime] #the minimum time windows of availibility of l needed is the time to transport from is current point Settings['MH_lt'][l][tt][1] to i, then from i to iprimme
           
            is_free = True
            for ttt in range(tt, max( tt + needed_time_slot, t + Settings['TR_ii'][i,iprime] )): # max is needed to check that we do not get l too early
                if(Plan['MH_lt'][l,ttt, 0] != 0):
                    is_free = False  # the equipment l cannot be used for the required transfer because it's in use at least once during the needed time for the transfer
                    tt = ttt         # tt is updated with the last non free timeslot
            
            if is_free:              # the equipment l is available for the transfer from i to iprime at earliest time tt. It will bring l in cell iprime, we need to check that it's compatible with the next timesteps of Settings['MH_lt'][l]
                ttt = max( tt+needed_time_slot, t+Settings['TR_ii'][i,iprime] )

                next_i = Plan['MH_lt'][l,ttt,1]
                for tttt in range( ttt, ttt + max(Settings['TR_ii'][iprime, next_i], 1)):        #max needed if iprime == next_i, we still need to check the equipemnt is not busy moving from previous_i to next_i  ! 
                    if Plan['MH_lt'][l,tttt,0] == 1 and Plan['MH_lt'][l, tttt-1, 1] == iprime:   #the equipment is used shortly after the needed_time_slot, but it's used with the assumption that the equipemnt origin is the location where we will bring it, so it' s ok.
                        best_t[l] = max(tt, t-Settings['TR_ii'][Plan['MH_lt'][l,tt,1],i])        #earliest time at which we can start, without starting too soon vis-a-vis t (production start)
                        best_prod_start_time_t[l] = best_t[l] + Settings['TR_ii'][Plan['MH_lt'][l,tt,1],i]+ Settings['TR_ii'][i, iprime]
                        cas[l] = 0
                        relevant_t[l] = tttt
                        slot_allocated[l]= True
                        break
                    elif Plan['MH_lt'][l,tttt,0] == 1:
                        tt = tttt
                        is_free = False
                        #needed to exit previous for loop, slot cannot be used, equipment is needed from next_i before we have time to bring it there
                        break   
                    elif Plan['MH_lt'][l,tttt,0] == 2:   #there is an empty movement in the near future of the needed time slot (ie before we have time to bring l to next_i), so we need to check whether it can be compatible
                        next_next_i = Plan['MH_lt'][l,tttt,1]
                        compatible = True
                        for ttttt in range(tttt, max(tttt, ttt+ Settings['TR_ii'][iprime,next_next_i])): #next_next_i could be equal to iprime, in which case this is automatically compatible
                            if Plan['MH_lt'][l,ttttt,0] == 1:
                                compatible = False
                        if compatible == False:
                            is_free = False
                            tt = tttt
                            # we exit the previous for loop, slot cannot be used
                            break 
                        else: # there is an empty movement shortly after needed_time_slot, we don't have time to bring l back to it's original place but we can change the future allocation to sthg compatible
                            best_t[l] = max(tt, t-Settings['TR_ii'][Plan['MH_lt'][l,tt,1],i]) #earliest time at which we can start, without starting too soon vis-a-vis t (production start)
                            best_prod_start_time_t[l] = best_t[l] + Settings['TR_ii'][Plan['MH_lt'][l,tt,1],i]+ Settings['TR_ii'][i, iprime]
                            cas[l] = 1 
                            relevant_t[l] = tttt
                            slot_allocated[l] = True
                            break
                if is_free and not(slot_allocated[l]): #slot is still potentially ok, and no allocation was made yet, so it means near future is filled with zeros, we can indeed use the slot
                    best_t[l] = max(tt, t-Settings['TR_ii'][Plan['MH_lt'][l,tt,1],i]) #earliest time at which we can start, without starting too soon vis-a-vis t (production start)
                    best_prod_start_time_t[l] = best_t[l] + Settings['TR_ii'][Plan['MH_lt'][l,tt,1],i]+ Settings['TR_ii'][i, iprime]
                    cas[l] = 2 
                    relevant_t[l] = best_t[l]
                    slot_allocated[l] = True
    
    solution_exist = False
    for l in range(Settings['Beta']):
        solution_exist = slot_allocated[l] or solution_exist
    
    if not solution_exist:
        exit("[computeMHv5d] No equipment MH could be allocated to a job/operation within the current t_max")
        
    #we choose the first l which minimizes best_prod_start_time THEN minimizes cost_of_MH THEN maximizes best_starting_time
    indices_prod_time = [idx for idx, x in enumerate(best_prod_start_time_t) if x == min(best_prod_start_time_t)]
    cost_of_mh = [Settings['delta'][idx] * (Settings['TR_ii'][Plan['MH_lt'][idx, best_t[idx], 1],i] + Settings['TR_ii'][i, iprime]) for idx in indices_prod_time]
    indices_cost = [idx for idx, x in enumerate(cost_of_mh) if x == min(cost_of_mh)]
    value = [int for idx in indices_cost]
    for idx,l in enumerate(indices_cost):
        value[idx] = best_t[indices_prod_time[l]]   
    best_l = value.index(max(value))
    best_l = indices_prod_time[indices_cost[best_l]]
    best_tt = best_t[best_l]

    #we allocate the transfer to the chosen l
    initial_i =  Plan['MH_lt'][best_l, best_tt, 1]
    allocateMH(Plan, Settings, best_l, i, iprime, best_tt, cas[best_l], relevant_t[best_l])
    return best_tt + Settings['TR_ii'][initial_i,i]+ Settings['TR_ii'][i, iprime]


def BuildTotalPlanWithMHd(sequence, Plan, Settings):
    """
    Deterministic approach.
    Build total plan based on constraints.
    """
    sequence_f = copy.deepcopy(sequence)
    sequence, _ = ExtractFirstSeqFromRandomSequenceJobCellv2(sequence, Settings)
    if not _ :
        # drop because of a sequence element
        return 1 
    Plan['AL'] = BuildAllocationMatrixFromSequence(sequence, Settings)
    ApplyMaskOfExistingOperationToAllocationMatrix(Plan, Settings)


    for e, (j, o, i) in enumerate(sequence):
        if Settings['O_jio'][j, i, o] == 0:
            #drop because of O_jio
            return 2

        if o == 0:
            t = Settings['R_j'][j] + Settings['DT_j'][j]            # mininimum starting time at t = Rj + DTj
            while CellCapacityIsExceeded(o,j,i,t, Plan, Settings):  # we start o of job j on cell i as soon as cell i is not saturated
                t += 1
            UpdateTimePLAN(o,j,i,t, Plan, Settings)                # once we know the earliest time t for (o,j) on i, we update S and compute C & A

        if o > 0:
            if Plan['AL'][j,o] == Settings['imax']:
                return 3                                            # test if there is an operation to be performed for job j /  #test if an operation o of job j can be performed at i 
            
            if Plan['AL'][j, o-1] == Plan['AL'][j, o]:              # operation o continues after operation o-1 on cell i
                t = Plan['C_jio'][j,i, o-1] + Settings['TRT_jio'][j , i, o-1]

                while CellCapacityIsExceeded(o,j,i,t, Plan, Settings):  #we start o of job j on cell i as soon as cell i is not saturated
                    t+= 1
                UpdateTimePLAN(o,j,i,t, Plan, Settings)                #once we know the earliest time t for (o,j) on i, we update S and compute C & A
            else:                                                       #operation o moves to another cell iprime

                iprime = Plan['AL'][j,o]
                i =  Plan['AL'][j, o-1]

                t = Plan['C_jio'][(j, i, o-1)] 
                t = computeMHv5d(i, iprime, t, Plan, Settings)
                while CellCapacityIsExceeded(o, j, iprime, t, Plan, Settings): #we start o of job j on cell iprime as soon as cell iprime is not saturated
                    t+=1
                UpdateTimePLAN(o,j,iprime,t, Plan, Settings)
    
    # Remove necessary MH movements
    optimizeMH(Plan, Settings)

    if not IsMHDistributedCorrectly(Plan, Settings)[0]:
        exit('[BuildTotalPlanWithMHd] : MH distributed uncorrectly!')

    if not areMHandSandCcoherent(Plan, Settings):
        exit('[BuildTotalPlanWithMHd] : distributution of MH is uncoherent with total plan A_tjio -> (S_jio and C_jio)')

    return 0


def BuildTotalPlanWithMHs(sequence, Plan, Settings):
    """
    Sequence base approach.
    Build total plan based on constraints
    """

    sequence, _ = ExtractFirstSeqFromRandomSequenceJobCellMH(sequence, Settings)
    if not _ :
        # drop because of a sequence element
        return 1 

    Plan['AL'] = BuildAllocationMatrixFromSequence(sequence, Settings)
    ApplyMaskOfExistingOperationToAllocationMatrix(Plan, Settings)


    for e, (j, o, i, l) in enumerate(sequence):
        if Settings['O_jio'][j, i, o] == 0:
            #drop because of O_jio
            return 2

        if o == 0:
            t = Settings['R_j'][j] + Settings['DT_j'][j]                # mininimum starting time at t = Rj + DTj
            while CellCapacityIsExceeded(o,j,i,t, Plan, Settings):      #we start o of job j on cell i as soon as cell i is not saturated
                t += 1
            UpdateTimePLAN(o,j,i,t, Plan, Settings)                    #once we know the earliest time t for (o,j) on i, we update S and compute C & A

        if o > 0:
            if Plan['AL'][j,o] == Settings['imax']:
                return 3                                                #test if there is an operation to be performed for job j /  #test if an operation o of job j can be performed at i 
            
            if Plan['AL'][j, o-1] == Plan['AL'][j, o]:                  #operation o continues after operation o-1 on cell i
                t = Plan['C_jio'][j,i, o-1] + Settings['TRT_jio'][j , i, o-1]

                while CellCapacityIsExceeded(o,j,i,t, Plan, Settings):  #we start o of job j on cell i as soon as cell i is not saturated
                    t+= 1
                UpdateTimePLAN(o,j,i,t, Plan, Settings)                #once we know the earliest time t for (o,j) on i, we update S and compute C & A
            else:                                                       #operation o moves to another cell iprime
                iprime = Plan['AL'][j,o]
                i =  Plan['AL'][j, o-1]
                t = Plan['C_jio'][(j, i, o-1)] 
                
                t = computeMHv5s(i, iprime, t, l, Plan, Settings)
                while CellCapacityIsExceeded(o,j,iprime,t, Plan, Settings): #we start o of job j on cell iprime as soon as cell iprime is not saturated
                    t+=1
                UpdateTimePLAN(o,j,iprime,t, Plan, Settings)

    # Remove necessary MH movements
    optimizeMH(Plan, Settings)
    
    if not IsMHDistributedCorrectly(Plan, Settings)[0]:
        exit('[BuildTotalPlanWithMHs] : MH distributed uncorrectly!')

    if not areMHandSandCcoherent(Plan, Settings):
        exit('[BuildTotalPlanWithMHs] : distributution of MH is uncoherent with total plan A_tjio -> (S_jio and C_jio)')

    return 0


######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
# # Note: not used. 
def CellCapacityIsNotExceeded(Plan,  Settings):
    """
    Constraint 3

    The number of assigned operations to start on cell i at time t cannot exceed its total number of machines
    
    SUM_oj [A_tjio] ≤ N_i;

    True: if satisfies the conditions
    False: If doesn't satisfie the conditions

    """
    condition = None
    CellCapacity_on_each_time_unit = np.sum(Plan['A_tjio'], axis = (1,3))

    # If the condition is violated, we obtain a value greater than 0
    condition = np.sum(CellCapacity_on_each_time_unit > Settings['N_i'])
    
    if condition == 0:
        return True
    else:
        return False


# # Note: not used. 
def OneOperationIsProcessedOnOnlyOneCell(Plan):
    """
    Constraint 4
    One operation cannot be processed on more than one cell
    SUM_i [A_tjio] ≤ 1
    """
    condition = None
    # If the condition is violated, we obtain a value greater than 0
    condition = np.sum( np.sum(Plan['A_tjio'], axis = (2)) > 1 )  
    
    if condition == 0:
        return True
    else:
        return False


# # Note: not used. 
def AreAllOperationCompleted(Plan, Settings):
    """
    Constraint 5
    When we start on a machine, we use 
    the machine till the end of Processing time + setup time.
    Return TRUE if all operations was completed otherwise FALSE
    """

    S_jio_bin = (Plan['S_jio'] != Settings['t_max']).astype(np.int)
    Total_needed_time_jio = S_jio_bin * (Settings['ST_jio'] + Settings['P_jio'])
    
    for j in range(Settings['jmax']):
        for i in range(Settings['imax']):
            for o in range(Settings['omax']):

                needed_time = Total_needed_time_jio[j,i,o]
                start_t = Plan['S_jio'][j,i,o]
                scheduled_time = np.sum(Plan['A_tjio'][start_t: start_t + needed_time, j, i, o] , axis = (0))
                
                if needed_time != scheduled_time:
                    return False
    return True


# # Note: not used. 
def GenerateIndicesOfOperationForSetupAndProcessingTime(time_start, time_end, idx):
    """
    Generate indices for setup and processing time of a given operation idx (where we should put 1)
    """

    new_idx = np.concatenate(([time_start], idx))
    for t in range(time_start+1, time_end):
        new_idx = np.vstack((new_idx, np.concatenate(([t], idx))))
    return new_idx.T


# Note: not used. 
def GenerateAtjioFromStjio(S_jio, Settings):
    """
    Generation of A_tjio from matrix S_jio
    """

    # preinit
    A_tjio = np.zeros(( Settings['t_max'], 
                        Settings['jmax'], 
                        Settings['imax'], 
                        Settings['omax']))
    # iterate by units of time
    for t in range(0, A_tjio.shape[0]):

        # get all operations starting at time t
        if t == Settings['t_max']:
            break
        idx_operations_at_t = np.where(S_jio == t)
        # to numpy array
        idx_operations_at_t_np = np.array(idx_operations_at_t)

        # idx_operations_at_t_np.shape[1] - number of matches S_jio == t,
        # in other words the number of operations starting at time t
        for operation in range(idx_operations_at_t_np.shape[1]):
            idx_of_operation = idx_operations_at_t_np[:,operation]
            time_to_end_operation = Settings['ST_jio'][tuple(idx_of_operation)] + Settings['P_jio'][tuple(idx_of_operation)]

            # - we put 1 where operations are performed at the moment
            # - we use (+= 1) instead of  (= 1) to check the overlap
            A_tjio[tuple(GenerateIndicesOfOperationForSetupAndProcessingTime(t, t + time_to_end_operation, idx_of_operation))] += 1
    
    return A_tjio
