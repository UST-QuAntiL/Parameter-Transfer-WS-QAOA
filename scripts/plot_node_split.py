import matplotlib.pyplot as plt
import os

from plotting import *


# Use Latex font
plt.rcParams.update({
    "font.family": "Helvetica"
})

files = ['(3-3)-0-000101.npy', '(3-3)-1-00010.npy', '(3-3)-1-00101.npy', '(3-3)-2-0001.npy']
node_split_files = ['./energies/' + f for f in files]

# Plot
draw_multiple_landscapes_and_graphs(node_split_files , rows=1, cols=4, figsize=(7.166, 7.166/3))

# show
plt.show()
# plt.savefig('node_split.pdf')