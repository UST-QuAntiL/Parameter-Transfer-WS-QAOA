import matplotlib.pyplot as plt
import os

from plotting import *


# Load files
grid_files = sorted(os.listdir('./energies'))
files = [[None, None, None]]*3
# figsize = (7.166, 7.166)
figsize = (10, 10)

# Reordering of the graphs
ordering = [[(0,1,2,3,4,5,8,7,6,10,9,11),(0,1,2,3,5,7,6,8,9), range(5)],
            [tuple(range(12))+(15,14,13,12)+(19,18,17,16)+(23,22,21,20),
             (0,1,2,6,3,4,5,9,10,11,14,13,12,17,16,15,18,23,22,21),
             (0,1,2,3,4,5,7,6,9,8,11,10)],
            [tuple(range(15))+(19,18,17,16,15)+(24,23,22,21,20)+(29,28,27,26,25),
             (0,1,2,3,8,9,10,11,7,12,13,14,15)+(19,18,17,16)+(23,22,21,20)+(24,31,30,29,28),
             tuple(range(8))+(11,10,9)+(14,13,12)+(17,16,15)]]

# Row and Column dimensions for the plot
dimensions = [[(3,4), (2,5), (1,5)],
              [(4,6), (4,5), (3,4)],
              [(5,6), (4,8), (3,6)]]

for j in range(0,3):
    for k in range(0,3):
        # (3-j)-k graphs
        name_list = sorted(['./energies/' + name for name in grid_files if f'(3-{j+3})-{k}' in name])
        files[j][k]= name_list
        # reorder the files
        files[j][k] = [name_list[i] for i in ordering[j][k]]
        # Plot
        draw_multiple_landscapes_and_graphs(files[j][k], rows=dimensions[j][k][0], cols=dimensions[j][k][1], figsize=figsize)
        # plt.savefig(f'(3-{j+3})-{k}.pdf')

# show
plt.show()