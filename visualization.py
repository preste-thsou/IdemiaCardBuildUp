import numpy as np
import copy
import matplotlib.pyplot as plt
import json

from problem import utils, problem_variables


def plot_fronts(data_list, algos, folder, show = False):
    
    def UpadeMax(data, max_x_T, max_y_C):

        if np.max(data[:,0]) > max_x_T:
            max_x_T = np.max(data[:,0])

        if np.max(data[:,1]) > max_y_C:
            max_y_C = np.max(data[:,1])

        return max_x_T, max_y_C


    fig = plt.figure(figsize = (15,7))
    ax = fig.add_subplot(1, 1, 1)
    
    max_x_T = 0
    max_y_C = 0

    for data_t, algo in zip(data_list, algos):
        data = utils.TransformSolutionToArray(data_t)
        data_T = data[:,0]
        data_C = data[:,1]

        ax.scatter(data_T, data_C,  s = 200, alpha = 0.7, label = "{}".format(algo))

        max_x_T, max_y_C = UpadeMax(data, max_x_T, max_y_C)



    ax.grid(True)
    ax.set_ylabel('Nb of layers', fontsize=20)
    ax.set_xlabel('Symmetry deviation', fontsize=20)
    ax.legend(fontsize = 15)
   

    st_x_T = int( (max_x_T + 5) / 10 )
    st_y_C = int( (max_y_C + 5) / 10 )
    

    ax.xaxis.set_ticks(np.arange(0, max_x_T + st_x_T + 5, max(1,st_x_T)))
    ax.yaxis.set_ticks(np.arange(0, max_y_C + st_y_C + 5, max(1,st_y_C)))

    if show:
        plt.show()
    else:
        plt.savefig(folder+'fronts_comparison.png')




def plot_ERT(data_list, algos, folder, show = False):
    """
    Run time empirical cumulative density function plot.
    """
    def GeneratePoints(data_T, data_C):
    
        x = []
        y_T = []
        y_C = []

        for e in range(data_T.shape[0]-1):
            x.append(data_T[e,0])
            x.append(data_T[e+1,0])
            x.append(data_T[e+1,0])

            y_T.append(data_T[e,1])
            y_T.append(data_T[e,1])
            y_T.append(data_T[e+1,1])

            y_C.append(data_C[e,1])
            y_C.append(data_C[e,1])
            y_C.append(data_C[e+1,1])


        x.append(data_T[-1,0]+5)
        y_T.append(data_T[-1,1]) 
        y_C.append(data_C[-1,1])

        return x, y_T, y_C

    def UpadeMax(data, max_x, max_y_T, max_y_C):

        if np.max(data[:,0]) > max_x:
            max_x = np.max(data[:,0])

        if np.max(data[:,1]) > max_y_T:
            max_y_T = np.max(data[:,1])

        if np.max(data[:,2]) > max_y_C:
            max_y_C = np.max(data[:,2])

        return max_x, max_y_T, max_y_C


    fig = plt.figure(figsize = (15,7))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    max_x = 0
    max_y_T = 0
    max_y_C = 0

    for data_t, algo in zip(data_list, algos):

        data = np.array(data_t)
        data_T = data[:,:2]
        data_C = data[:,[0,2]]

        ax1.scatter(data_T[:,0], data_T[:,1])
        ax2.scatter(data_C[:,0], data_C[:,1])

        max_x, max_y_T, max_y_C = UpadeMax(data, max_x, max_y_T, max_y_C)
        x, y_T, y_C =  GeneratePoints(data_T, data_C)

        ax1.plot(x, y_T, label = "{}".format(algo))
        ax2.plot(x, y_C, label = "{}".format(algo))

    ax1.grid(True)
    ax1.set_ylabel('Symmetry deviation', fontsize=20)
    ax1.set_xlabel('Evaluation', fontsize=20)
    ax1.legend(fontsize = 15)
    ax2.set_ylabel('Nb of layers', fontsize=20)
    ax2.set_xlabel('Evaluation', fontsize=20)
    ax2.grid(True)
    ax2.legend(fontsize = 15)



    st_x = int( (max_x+ 5) / 10 )
    st_y_T = int( (max_y_T + 5) / 10 )
    st_y_C = int( (max_y_C + 5) / 10 )
    if max_x != np.inf:
        ax1.xaxis.set_ticks(np.arange(0, max_x + 5, max(1,st_x)))
        ax1.yaxis.set_ticks(np.arange(0, max_y_T + st_y_T + 5, max(1,st_y_T)))

        ax2.xaxis.set_ticks(np.arange(0, max_x + 5, max(1,st_x)))
        ax2.yaxis.set_ticks(np.arange(0, max_y_C + st_y_C + 5, max(1,st_y_C)))

    if show:
        plt.show()
    else:
        plt.savefig(folder+'metaheuristic_ert_comparison.png')


def GetListOfSolutionsVar(filename):
    """

    """
    data = open(filename, 'r') 

    vars_list = []
    for line in data.readlines() :
        vars_list.append([float(x) for x in line.split()])
    
    return vars_list

def WritePlan(filename, solution):
    localSettings = copy.deepcopy(problem_variables.SETTINGS)
    localPlan = copy.deepcopy(problem_variables.PLAN)
    utils.buildPlan(localSettings,localPlan, solution)
    with open(filename, "w") as of:
        of.write(str(localPlan))



def PlotTimeplan(Plan, Settings):
    """
    Time plan visualization 2D.
    """

    cell_cap = np.zeros((Settings['imax'], np.max(Plan['C_jio'])))
    k_i = 1/(Settings['N_i'] + 2)
    # Create plot
    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot(1, 1, 1)

    for j in range(Settings['jmax']):
        data = np.where(Plan['A_tjio'][:,j,:,:] == 1)        
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
        
        #ax.plot(data[0], 1+ data[1] + cell_cap[(data[1], data[0])]*k_i[(data[1])], alpha=0.9)
        cell_cap[(data[1], data[0])] +=1
    plt.yticks(np.arange(0,Settings['imax']+4, 1))
    plt.xticks(np.arange(0,np.max(Plan['C_jio'])+2, 1))

    plt.title('- Time Plan -', fontsize = 20)

    plt.xlabel("Time ", fontsize = 15)
    plt.ylabel("Cell", fontsize = 15)
    plt.legend(loc=2,fontsize = 15, ncol=Settings['jmax'])
    plt.grid(True)
    plt.show()


def PlotTimeplanMH(Plan, Settings):
    """
    Time plan with MH visualization 2D.
    """
    
    def transform_list_for_plot(data):
        b = data[0]
        x = [data[1][0], data[2][0]]
        y = [data[1][1]+1, data[2][1]+1]
        return b,x,y

    cell_cap = np.zeros((Settings['imax'], np.max(Plan['C_jio'])))
    k_i = 1/(Settings['N_i'] + 2)
    # Create plot
    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot(1, 1, 1)

    for j in range(Settings['jmax']):
        data = np.where(Plan['A_tjio'][:,j,:,:] == 1)        
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
        
        #ax.plot(data[0], 1+ data[1] + cell_cap[(data[1], data[0])]*k_i[(data[1])], alpha=0.9)
        cell_cap[(data[1], data[0])] +=1
    
    f, transportations = utils.IsMHDistributedCorrectly(Plan, Settings)
    if f:
        for l, data_l in enumerate(transportations):
            if len(data_l) == 0:
                continue
            
            labeled_b1 = False
            labeled_b2 = False
            
            for transp in data_l:
                b,x,y = transform_list_for_plot(transp)
                
                if not labeled_b1 and b==1:
                    labeled_b1 = (b==1)
                    ax.plot(x,y, ls = 'solid' if b==1 else 'dashed' , c=(l/Settings['Beta'],0,0,1), label = 'MH: {} empty'.format(l) if b == 2 else 'MH: {} loaded'.format(l)   ) 
                
                elif not labeled_b2 and b==2:
                    labeled_b2 = (b==2)
                    ax.plot(x,y, ls = 'solid' if b==1 else 'dashed' , c=(l/Settings['Beta'],0,0,1), label = 'MH: {} empty'.format(l) if b == 2 else 'MH: {} loaded'.format(l)   ) 
                else:
                    ax.plot(x,y, ls = 'solid' if b==1 else 'dashed' , c=(l/Settings['Beta'],0,0,1))     
    else:
        print("ERROR: MH isn't distributed correctly!")
        
    plt.yticks(np.arange(0,Settings['imax']+3, 1))
    plt.xticks(np.arange(0,np.max(Plan['C_jio'])+1, 1))

    
    
    plt.title('- Time Plan -', fontsize = 20)
    plt.xlabel("Time ", fontsize = 15)
    plt.ylabel("Cell", fontsize = 15)
    plt.legend(loc=2,fontsize = 15, ncol=Settings['jmax'])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print('Visualization tools')