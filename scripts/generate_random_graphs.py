import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx import Graph
import itertools

from plotting import *
from graph_management import *

maximum_degree = 5
# possible combinations of (number of) neighbors
pairs = [(j,i) for i in range(maximum_degree) for j in range(i+1)]
graphs =[]

for (i,j) in pairs:
    for num_merged_nodes in range(i+1):
        G = Graph()
        edges = [(0,1)]  # central edge
        edges.extend([(0, k+2) for k in range(i)])  # neighbors of node 0
        n = i + 2 - num_merged_nodes  # reuse existing nodes
        edges.extend([(1, k+n) for k in range(j)])  # neighbors of node 1

        G.add_edges_from(edges)
        nx.convert_node_labels_to_integers(G)
        # draw_graph_with_ws(G, show=False)
        # plt.get_current_fig_manager().canvas.set_window_title(f'({i+1}-{j+1})-{num_merged_nodes}')
        graphs.append(G)
        nx.write_adjlist(G, f'./graphs/({i+1}-{j+1})-{num_merged_nodes}.graph')

# plt.show()
print(len(graphs))
print(f'Landscape total: {sum([len(get_relevant_warmstartings(G)) for G in graphs])}')
fig = plt.figure(figsize=(8,8))

for i, G in enumerate(graphs):
    ax = fig.add_subplot(6, 6, i+1)
    draw_graph_with_ws(G, show=False, axes=ax)
fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
plt.show()

# for i, graph in enumerate(graphs):
#     nx.write_adjlist(graph, f'./test-run/graphs/{i}.graph')