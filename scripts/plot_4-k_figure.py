import matplotlib.pyplot as plt
import os

from plotting import *


# Load files
grid_files = sorted(os.listdir('./energies'))
files = [[None, None, None, None]]*2
# figsize = (7.166, 7.166)
figsize = (10, 10)

# Reordering of the graphs
ordering = [[tuple(range(10))+(13,12,11,10)+(16,15,14)+(18,17,19),
             (0,1,2,3,4,5,8,9,11)+(14,13,12)+(16,15)+(17,19,20),
             (0,1,2,3,4,5,8)+(10,9)+(12,11)+(14,15),
             range(6)],
            [tuple(range(20))+(24,23,22,21,20)+(29,28,27,26,25)+(34,33,32,31,30)+(39,38,37,36,35),
             (0,1,2,3)+(8,4,5,6)+(7,16,17,18,19,15)+(20,21,22,23) + (27,26,25,24)+(31,30,29,28)+(32,39,38,37,36)+(40,47,46,45,44),
            #  (0,1,2,3)+(8,9,10,11)+(7,16,12,13,14,15)+(20,21,22,23) + (27,26,25,24)+(31,35,34,33)+(32,43,42,41)+(40,47,46,45,44),  # alternative ordering
             (0,1,2,9,3,4,5,12,6,7,8,15,16,17)+(20,19,18)+(23,22,21)+(27,26,25,24)+(30,35,34,33),
             tuple(range(8))+(9,8,11,10,13,12,15,14)]]

# Row and Column dimensions for the plot
dimensions = [[(4,5), (4,5), (4,4), (2,3)],
              [(5,8), (6,6), (4,7), (4,4)]]

for j in range(0,2):
    for k in range(0,4):
        # (4-j)-k graphs
        name_list = sorted(['./energies/' + name for name in grid_files if f'(4-{j+4})-{k}' in name])
        files[j][k]= name_list
        # reorder the files
        files[j][k] = [name_list[i] for i in ordering[j][k]]
        # Plot
        draw_multiple_landscapes_and_graphs(files[j][k], rows=dimensions[j][k][0], cols=dimensions[j][k][1], figsize=figsize)
        # plt.savefig(f'(4-{j+4})-{k}.pdf')

# show
plt.show()