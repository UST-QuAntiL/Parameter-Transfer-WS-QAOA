import matplotlib.pyplot as plt
import os

from plotting import *


# Load files
grid_files = sorted(os.listdir('./energies'))
files = [None, None, None, None, None]
# figsize = (7.166, 7.166)
figsize = (12.45, 13.34)

# Reordering of the graphs
ordering = [tuple(range(15))+(19,18,17,16,15)+(23,22,21,20)+(26,25,24)+(28,27,29),
            (0,1,2,3,4,5,6,7,11,12,13,16,17,19)+(23,22,21,20)+(26,25,24,27)+(31,30,32,34,35),
            (0,1,2,3,4,5,6,7,8,13,14,17)+(20,19,18)+(23,22,21)+(26,24,28,31,32),
            (0,1,2,3,4,5,6,7,11)+(13,12,15,14,18,16,19),
            range(8)]

# Row and Column dimensions for the plot
dimensions = [(5,6), (4,7), (4,6), (4,4), (2,4)]

for k in range(0,5):
    # (4-j)-k graphs
    name_list = sorted(['./energies/' + name for name in grid_files if f'(5-5)-{k}' in name])
    files[k]= name_list
    # reorder the files
    files[k] = [name_list[i] for i in ordering[k]]
    # Plot
    draw_multiple_landscapes_and_graphs(files[k], rows=dimensions[k][0], cols=dimensions[k][1], figsize=figsize)
    # plt.savefig(f'(5-5)-{k}.pdf')

# show
plt.show()