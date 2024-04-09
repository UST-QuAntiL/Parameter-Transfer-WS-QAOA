import matplotlib.pyplot as plt
import networkx as nx
from circuit_generation import qaoa_circuit
from plotting import *


G = nx.read_adjlist('./test-run/graphs/(3-3)-0.graph', nodetype=int)

ws = [1,0,0,1,0,0]
qc = qaoa_circuit(G, apx_sol=ws)

# rydata = [d for d in qc.data if d[0].name == 'ry']
# costdata = [d for d in qc.data if d[0].name == 'rzz']
# mixerdata = [d for d in qc.data if d[0].name == 'rx']
# qc.data = rydata[:]
# qc.barrier()
# qc.data += costdata
# qc.barrier()
# qc.data += mixerdata


# draw_graph_with_ws(G, warmstarting=ws)
print(qc.draw('latex_source'))
plt.show()
