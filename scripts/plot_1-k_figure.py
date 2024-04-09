import matplotlib.pyplot as plt
import os

from plotting import *


# Load files
grid_files = sorted(os.listdir('./energies'))
files = [None]*5
figsize = [(7.166, 7.166/2)]*5
figsize[4] = (7.166, 7.166)

ordering = [(0,1),
            (0,1,3,2),
            (0,1,2,5,4,3),
            (0,1,2,3,7,6,5,4),
            (0,1,2,3,4,9,8,7,6,5)]
dimensions = [(1,2),
              (1,4),
              (2,3),
              (2,4),
              (2,5)]

for k in range(1,5):
    # (1-k)-0 graphs
    name_list = sorted(['./energies/' + name for name in grid_files if f'(1-{k+1})-0' in name])
    files[k]= name_list
    # reorder the files
    files[k] = [name_list[i] for i in ordering[k]]
    # Plot
    draw_multiple_landscapes_and_graphs(files[k], rows=dimensions[k][0], cols=dimensions[k][1], figsize=figsize[k])
    # plt.savefig(f'(1-{k+1})-0.pdf')

# show
plt.show()
