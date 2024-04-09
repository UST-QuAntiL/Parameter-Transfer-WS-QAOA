import numpy as np
import networkx as nx
from networkx import Graph
import matplotlib.pyplot as plt

from calculations import *
from plotting import *
from graph_management import *
import random


np.random.seed(5)
random.seed(40)

# Generate random graph with bounded degree of 5
seq = [random.randint(1,5) for i in range(50)]
G = nx.random_degree_sequence_graph(seq, seed=42)

# Calculate cut with GW algorithm
apx = GW_maxcut(G)

# Break graph into subgraphs
subgraphs = get_ws_subgraphs(G, apx, get_equivalents=False)
# Break graph into subgraphs and transform them into standard form
equivalents = get_ws_subgraphs(G, apx, get_equivalents=True)

# initialize energy arrays
true_energy = np.zeros((30, 30))
apx_energy = np.zeros((30, 30))
apx_energy_thresh = np.zeros((30, 30))

# Calculate the true energy
for sg, edge, occ in subgraphs:
  grid = load_landscape_from_graph(sg)
  true_energy += occ * grid

# Calculate the approximate energy
for sg, edge, occ in equivalents:
  grid = load_landscape_from_graph(sg)
  apx_energy += occ * grid

# Only consider the 3 most frequent graphs
for sg, edge, occ in equivalents:
  if occ < 9:  # third highest occurence is 9
    continue
  grid = load_landscape_from_graph(sg)
  apx_energy_thresh += occ * grid


# Plot the figure
fig = plt.figure(figsize=(7.166, 1.75))  #(7.166, 2.2)
plt.subplots_adjust(left=0.0, right=0.96, top=0.94, bottom=0.09, wspace=0.5)  # page config
ax1 = fig.add_subplot(1, 4, 1)
ax1.set_aspect(0.8)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
ax4 = fig.add_subplot(1, 4, 4)
options = {'ax': ax1, 'node_size': 20}
draw_graph_with_cut(G, apx, show=False, **options)
plot_energy(true_energy, axes=ax2, show=False, title='True Energy')
plot_energy(apx_energy, axes=ax3, show=False, title='Approximate Energy')
plot_energy(apx_energy_thresh, axes=ax4, show=False, title='Energy of Top 3')
  
# plt.savefig('landscape_approximation.pdf')
plt.show()