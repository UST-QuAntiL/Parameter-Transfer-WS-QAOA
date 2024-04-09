import matplotlib.pyplot as plt
import os

from plotting import *


# Load files
grid_files = sorted(os.listdir('./energies'))
files = [[None, None]]*4
figsize = (7.166, 7.166)

# Reordering of the graphs
ordering = [[(0,1,2,4,3,5),(0,1,2)],
            [(0,1,2,3,4,5,8,7,6,11,10,9),(0,1,2,3,7,5,6,4)],
            [(0,1,2,3,4,5,6,7,11,10,9,8,15,14,13,12),(0,1,2,3,4,5,8,7,6,11,10,9)],
            [(0,1,2,3,4,5,6,7,8,9,14,13,12,11,10,19,18,17,16,15),(0,1,2,3,4,5,6,7,11,10,9,8,15,14,13,12)]]

# Row and Column dimensions for the plot
dimensions = [[(2,3), (1,3)],
              [(3,4), (2,4)],
              [(4,4), (3,4)],
              [(4,5), (4,4)]]

for j in range(0,4):
    for k in range(0,2):
        # (2-j)-k graphs
        name_list = sorted(['./energies/' + name for name in grid_files if f'(2-{j+2})-{k}' in name])
        files[j][k]= name_list
        # reorder the files
        files[j][k] = [name_list[i] for i in ordering[j][k]]
        # Plot
        draw_multiple_landscapes_and_graphs(files[j][k], rows=dimensions[j][k][0], cols=dimensions[j][k][1], figsize=figsize)
        # plt.savefig(f'(2-{j+2})-{k}.pdf')

# show
plt.show()