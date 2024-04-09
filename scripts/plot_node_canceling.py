import matplotlib.pyplot as plt

from plotting import *


# Use Latex font
plt.rcParams.update({
    "font.family": "Helvetica"
})

files = ['./energies/ordered/123_(3-3)-0-000000.npy', './energies/ordered/459_(5-5)-0-0000010001.npy', './energies/ordered/206_(3-5)-0-00000001.npy']

# Plot
draw_multiple_landscapes_and_graphs(files , rows=1, cols=3, figsize=(7.166, 7.166*0.4))

# show
# plt.show()
plt.savefig('node_canceling.pdf')