import numpy as np
import matplotlib.pyplot as plt


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