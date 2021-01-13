import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.colors as pltcolor
from problem import utils, problem_variables
import copy

def plotLayerScheme(filename, solution):
    localSettings = copy.deepcopy(problem_variables.SETTINGS)
    localPlan = copy.deepcopy(problem_variables.PLAN)
    utils.buildPlan(localSettings, localPlan, solution)
    utils.obj_minimizeEinteraction(localSettings, localPlan)
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 1, 1)

    colorNames = ['blue', 'red', 'purple', 'pink', 'limegreen', 'brown', 'yellow', 'orange', 'darkgreen', 'darkcyan',
                  'dodgerblue', 'royalblue', 'maroon']

    ref_h = 0
    index = 0
    for l in localPlan['l']:
        if l[0] != 0:
            index +=1
            doc_limits = [[(0,ref_h), (100, ref_h)], [(0, ref_h),(0, ref_h+l[0])],[(100,ref_h),(100, ref_h + l[0])], [(0, ref_h+l[0]),(100, ref_h+l[0])]]
            ref_h += l[0]
            lc = mc.LineCollection(doc_limits, color = colorNames[index % len(colorNames)], linewidths=2)
            lc.set_label('layer {} of thickness {}'.format(index, l[0]))
            ax.add_collection(lc)

    ax.legend(loc=1, fontsize=11)
    plt.title('- Doc layer schemes -', fontsize=20)
    plt.xticks(np.arange(-10, 150, 50))
    plt.yticks(np.arange(-50,ref_h+75, 50))


    plt.grid(False)
    plt.savefig(filename)

def plotHalfsDocScheme(filename, solution):
    localSettings = copy.deepcopy(problem_variables.SETTINGS)
    localPlan = copy.deepcopy(problem_variables.PLAN)
    utils.buildPlan(localSettings, localPlan, solution)
    utils.obj_minimizeEinteraction(localSettings, localPlan)

    # Create plot
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # min_x = localSettings['min_x']
    min_x = 0
    # min_y = localSettings['min_y']
    min_y = 0
    max_x = localSettings['max_x'] + 10
    max_y = localSettings['max_y'] + 10

    doc_limits = [[(min_x, min_y), (max_x, min_y)], [(min_x, min_y), (min_x, max_y)], [(min_x, max_y), (max_x, max_y)],
                  [(max_x, min_y), (max_x, max_y)]]
    lc = mc.LineCollection(doc_limits, color='grey', linewidths=2)
    lc2 = mc.LineCollection(doc_limits, color='grey', linewidths=2)
    ax1.add_collection(lc)
    ax2.add_collection(lc2)

    index = [1 for _ in range(localSettings['list_type_e'])]

    colorNames = ['blue', 'red', 'purple', 'pink', 'limegreen', 'brown', 'yellow', 'orange', 'darkgreen', 'darkcyan',
                  'dodgerblue', 'royalblue', 'maroon']
    for e in range(localSettings['len_e']):
        min_x = localPlan['P_e'][e][0]
        min_y = localPlan['P_e'][e][1]
        max_x = min_x + localPlan['P_e'][e][2]
        max_y = min_y + localPlan['P_e'][e][3]
        el_limits = [[(min_x, min_y), (max_x, min_y)], [(min_x, min_y), (min_x, max_y)],
                     [(min_x, max_y), (max_x, max_y)], [(max_x, min_y), (max_x, max_y)]]
        if localPlan['l'][localPlan['AL_e'][e]][2] > localPlan['c']:
            lc = mc.LineCollection(el_limits, color=colorNames[e % len(colorNames)], linewidths=2)
            if index[localSettings['list_e'][e]] == 1:
                lc.set_label('{}'.format(localSettings['Name_type_e'][localSettings['list_e'][e]]))
                index[localSettings['list_e'][e]] += 1
            else:
                lc.set_label('{} {}'.format(localSettings['Name_type_e'][localSettings['list_e'][e]],
                                            index[localSettings['list_e'][e]]))
                index[localSettings['list_e'][e]] += 1
            ax1.add_collection(lc)
        elif localPlan['l'][localPlan['AL_e'][e]][2] < localPlan['c']:
            lc = mc.LineCollection(el_limits, color=colorNames[e % len(colorNames)], linewidths=2)
            if index[localSettings['list_e'][e]] == 1:
                lc.set_label('{}'.format(localSettings['Name_type_e'][localSettings['list_e'][e]]))
                index[localSettings['list_e'][e]] += 1
            else:
                lc.set_label('{} {}'.format(localSettings['Name_type_e'][localSettings['list_e'][e]],
                                            index[localSettings['list_e'][e]]))
                index[localSettings['list_e'][e]] += 1
            ax2.add_collection(lc)
        elif localPlan['l'][localPlan['AL_e'][e]][2] == localPlan['c']:
            lc = mc.LineCollection(el_limits, color=colorNames[e % len(colorNames)], linewidths=2)
            lc2 = mc.LineCollection(el_limits, color=colorNames[e % len(colorNames)], linewidths=2)
            if index[localSettings['list_e'][e]] == 1:
                lc.set_label('{}'.format(localSettings['Name_type_e'][localSettings['list_e'][e]]))
                lc2.set_label('{}'.format(localSettings['Name_type_e'][localSettings['list_e'][e]]))
                index[localSettings['list_e'][e]] += 1
            else:
                lc.set_label('{} {}'.format(localSettings['Name_type_e'][localSettings['list_e'][e]],
                                            index[localSettings['list_e'][e]]))
                lc2.set_label('{} {}'.format(localSettings['Name_type_e'][localSettings['list_e'][e]],
                                            index[localSettings['list_e'][e]]))
                index[localSettings['list_e'][e]] += 1
            ax1.add_collection(lc)
            ax2.add_collection(lc2)

    ax1.axis('equal')
    ax1.margins(0.4)
    ax2.axis('equal')
    ax2.invert_xaxis()
    ax2.margins(0.4)

    ax1.legend(loc=2, fontsize = 11)
    ax2.legend(loc =2, fontsize = 11)
    plt.title('- Doc half schemes -', fontsize=20)
    plt.grid(False)
    plt.savefig(filename)

def plotDocScheme(filename, solution):
    localSettings = copy.deepcopy(problem_variables.SETTINGS)
    localPlan = copy.deepcopy(problem_variables.PLAN)
    utils.buildPlan(localSettings, localPlan, solution)
    utils.obj_minimizeEinteraction(localSettings, localPlan)

    # Create plot
    fig = plt.figure(figsize = (17,10))
    ax = fig.add_subplot(1, 1, 1)

    #min_x = localSettings['min_x']
    min_x = 0
    #min_y = localSettings['min_y']
    min_y = 0
    max_x = localSettings['max_x']+10
    max_y = localSettings['max_y']+10

    doc_limits = [[(min_x, min_y),(max_x, min_y)],[(min_x, min_y), (min_x, max_y)],[(min_x, max_y),(max_x, max_y)],[(max_x, min_y),(max_x,max_y)]]
    lc = mc.LineCollection(doc_limits, color = 'grey', linewidths = 2)
    ax.add_collection(lc)

    index = [1 for e in range(localSettings['list_type_e'])]
    colorNames = ['blue', 'red', 'purple', 'pink', 'limegreen', 'brown', 'yellow', 'orange', 'darkgreen', 'darkcyan', 'dodgerblue', 'royalblue','maroon']
    for e in range(localSettings['len_e']):
        min_x = localPlan['P_e'][e][0]
        min_y = localPlan['P_e'][e][1]
        max_x = min_x + localPlan['P_e'][e][2]
        max_y = min_y + localPlan['P_e'][e][3]
        el_limits = [[(min_x, min_y),(max_x, min_y)],[(min_x, min_y), (min_x, max_y)],[(min_x, max_y),(max_x, max_y)],[(max_x, min_y),(max_x,max_y)]]
        lc = mc.LineCollection(el_limits, color = colorNames[e % len(colorNames)], linewidths= 2)
        if index[localSettings['list_e'][e]] == 1:
            lc.set_label('{}'.format(localSettings['Name_type_e'][localSettings['list_e'][e]]))
            index[localSettings['list_e'][e]] += 1
        else:
            lc.set_label('{} {}'.format(localSettings['Name_type_e'][localSettings['list_e'][e]], index[localSettings['list_e'][e]]  ))
            index[localSettings['list_e'][e]] += 1
        ax.add_collection(lc)

    ax.axis('equal')
    ax.margins(0.4)

    plt.title('- Doc scheme -', fontsize=20)
    plt.grid(False)
    plt.legend(loc=2, fontsize=15)
    plt.savefig(filename)

def PlotTimeplan(Plan, Settings):
    """
    Time plan visualization 2D.
    """

    cell_cap = np.zeros((SETTINGS['imax'], np.max(PLAN['C_jio'])))
    k_i = 1/(SETTINGS['N_i'] + 2)
    
    # Create plot
    fig = plt.figure(figsize = (15,7))
    ax = fig.add_subplot(1, 1, 1)

    for j in range(SETTINGS['jmax']):
        data = np.where(PLAN['A_tjio'][:,j,:,:] == 1)
                
        ax.scatter(data[0], 1 + data[1] + cell_cap[(data[1], data[0])]*k_i[(data[1])]  , alpha=0.5, edgecolors='none', s=300, label= 'job: {}'.format(j+1))
        
        for x,y,o in zip(data[0],1+data[1] + cell_cap[(data[1], data[0])]*k_i[(data[1])], 1+data[2]):

            label = "{:d}".format(o)

            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,-3), # distance from text to points (x,y)
                         ha='center',
                         fontsize = 11,
                         fontweight ='bold') # horizontal alignment can be left, right or center
        
        ax.plot(data[0], 1+ data[1] + cell_cap[(data[1], data[0])]*k_i[(data[1])], alpha=0.9)
        cell_cap[(data[1], data[0])] +=1
    
    plt.yticks(np.arange(0,SETTINGS['imax']+2, 1))
    plt.xticks(np.arange(0,np.max(PLAN['C_jio'])+1, 1))
    plt.title('- Time Plan -', fontsize = 20)
    plt.xlabel("Time ", fontsize = 15)
    plt.ylabel("Cell", fontsize = 15)
    plt.legend(loc=2,fontsize = 15)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # PlotTimeplan(Plan, Settings)
    pass