import numpy as np
import networkx as nx
from networkx import Graph
from itertools import product



def iso_in_dict(G, apx_sol1, iso_dict):
  """
  Checks if an isomorphic graph exists within a dictionary of graphs.

  Parameters:
    G (nx.Graph): The input graph
    apx_sol1 (list): The approximate solution for the given graph
    iso_dict (dict): A dictionary of graphs, indexed by a hash function.

  Returns:
    bool: True if an isomorphic graph exists, False otherwise.
  """

  # Copy the graph
  G1 = G.copy()
  # Pre-select relevant graphs with ws_hash (same # of 0s, ...)
  ws_hash = ws_hashing(apx_sol1)

  # Check if a candidate is already in the dictionary
  if ws_hash in iso_dict.keys():
    # Add the approximate solution to G1 as 'weight'
    attrs = {i: apx_sol1[i] for i in range(len(apx_sol1))}
    nx.set_node_attributes(G1, attrs, 'weight')
    # Mark the central edge
    attrs = {e: 0 for e in G1.edges}
    attrs[(0,1)] = 1
    nx.set_edge_attributes(G1, attrs, 'weight')
    # Graph to compare to 
    G2 = G.copy()

    # check every candidate for isomorphism
    for apx_sol2 in iso_dict[ws_hash]:
      # Add approximate solution to G2 as 'weight'
      attrs = {i: apx_sol2[i] for i in range(len(apx_sol2))}
      nx.set_node_attributes(G2, attrs, 'weight')
      # Mark the central edge (important for random graphs)
      attrs = {e: 0 for e in G2.edges}
      attrs[(0,1)] = 1
      nx.set_edge_attributes(G2, attrs, 'weight')

      # Isomorphism check
      comp = lambda g1, g2: g1['weight'] == g2['weight']
      if nx.is_isomorphic(G1, G2, node_match=comp, edge_match=comp):
        # found iso
        return True

    # found no iso
    return False
  else:
    # no candidate
    return False


def ws_hashing(apx_sol):
  """ number of 0s and ones in tuple [0, 0, 1] --> (2, 1) """
  ws_hash = (np.count_nonzero(apx_sol == 0.),
             np.count_nonzero(apx_sol == 1.))
  return ws_hash


def number_to_ws(n, l, base=2):
    """
    Gives the warm start corresponding to a certain number.
    Index 0 is warm starting of node 0
    l is the length of the warm start i.e. number of nodes.
    """
    if n == 0:
        return np.array([0]*l)
    digits = []
    while n:
        digits.append(int(n % base))
        n //= base
    # pad with zeros
    digits.extend([0]*(l-len(digits)))
    return np.array(digits) / (base - 1)


def ws_to_number(apx_sol, base=2):
    return int(sum([(base-1)*apx_sol[i] * base**i for i in range(len(apx_sol))]))


def get_ws_from_attributes(G):
    apx = np.array([ws for n, ws in nx.get_node_attributes(G, 'weight').items()])
    return apx


def split_node(G, node):
  """ https://stackoverflow.com/questions/65853641/networkx-how-to-split-nodes-into-two-effectively-disconnecting-edges """
  edges = G.edges(node, data=True)
  ws_value = G.nodes[node]['weight']
  
  new_edges = []
  new_nodes = []

  H = G.__class__()
  H.add_nodes_from(G.subgraph(node))
  
  for i, (s, t, data) in enumerate(edges):
      new_node = '{}_{}'.format(node, i)
      I = nx.relabel_nodes(H, {node:new_node})
      new_nodes += list(I.nodes(data=True))
      new_edges.append((new_node, t, data))
  
  G.remove_node(node)
  G.add_nodes_from(new_nodes, weight=ws_value)
  G.add_edges_from(new_edges)
  
  return nx.convert_node_labels_to_integers(G)


def get_reg0_equivalent(G):
  G_eq = G.copy()
  for node in G.nodes:
    if G_eq.degree(node) == 2:
      G_eq = split_node(G_eq, node)
  return G_eq


def get_equivalent_graph(G, central_edge=(0,1)):
  G_eq = G.copy()
  for node in G.nodes:
    if G_eq.degree(node) == 2:
      # Check if its only neighbors are central nodes
      if central_edge[0] in G_eq.neighbors(node) and central_edge[1] in G_eq.neighbors(node):
        G_eq = split_node(G_eq, node)

  # add nodes if degree < 4
  for i in central_edge:
    while G_eq.degree(i) < 4:
       n = len(G_eq)  # new node number
       G_eq.add_edge(i, n, central=0)
       G_eq.nodes[n]['weight'] = 0
       G_eq.add_edge(i, n+1, central=0)
       G_eq.nodes[n+1]['weight'] = 1

  return G_eq



def get_subgraphs(G):
  subgraphs = []

  for edge in G.edges():
    u, v = edge
    subgraph = Graph()
    subgraph.add_edge(u,v)
    subgraph.add_edges_from([(u,n) for n in G.neighbors(u)])
    subgraph.add_edges_from([(v,n) for n in G.neighbors(v)])
    subgraph = nx.convert_node_labels_to_integers(subgraph)

    # iso check
    for item in subgraphs:
      sg, edge, occurrence = item
      # found iso
      if nx.is_isomorphic(subgraph, sg):
        item[2] += 1  # increase occurrence by one
        break
    else:  # executed if no break
      edge = (0,1)
      subgraphs.append([subgraph, edge, 1])
    
  return subgraphs


def get_ws_subgraphs(G, apx_sol, get_equivalents=False):
  """ returns a tuple of (subgraph, relevant-edge, occurence) """
  subgraphs = []

  for edge in G.edges():
    u, v = edge
    subgraph = Graph()
    subgraph.add_edge(u,v)
    subgraph.add_edges_from([(u,n) for n in G.neighbors(u)])
    subgraph.add_edges_from([(v,n) for n in G.neighbors(v)])
    # Mark the central edge (important for random graphs)
    attrs = {e: 0 for e in subgraph.edges}
    attrs[(u,v)] = 1
    nx.set_edge_attributes(subgraph, attrs, 'central')

    inverted_sg = subgraph.copy()

    # apply apx_sol
    attrs = {i: apx_sol[i] for i in subgraph.nodes}
    nx.set_node_attributes(subgraph, attrs, 'weight')

    # subgraph with inverse apx_sol
    attrs_inv = {i: np.abs(apx_sol[i]-1) for i in inverted_sg.nodes}
    nx.set_node_attributes(inverted_sg, attrs_inv, 'weight')

    # normalize labels
    subgraph = nx.convert_node_labels_to_integers(subgraph)
    inverted_sg = nx.convert_node_labels_to_integers(inverted_sg)

    if get_equivalents:
      subgraph = get_equivalent_graph(subgraph)
      inverted_sg = get_equivalent_graph(inverted_sg)

    # iso check
    comp = lambda g1, g2: g1['weight'] == g2['weight']
    ecomp = lambda g1, g2: g1['central'] == g2['central']
    for item in subgraphs:
      sg, edge, occurrence = item
      if nx.is_isomorphic(subgraph, sg, node_match=comp, edge_match=ecomp) or nx.is_isomorphic(inverted_sg, sg, node_match=comp, edge_match=ecomp):  # found iso
        item[2] += 1  # increase occurrence by one
        break
    else:  # executed if no break
      edge = (0,1)
      subgraphs.append([subgraph, edge, 1])
    
  return subgraphs


def get_relevant_warmstartings(G):
  """ get all warmstartings for graph G up to isomorphism """
  iso_dict = {}  # stores the graphs with in a simple hash table
  warmstartings = []
  candidates = []
  # only check the first half, as all other are isos
  # candidates = candidates[:len(candidates)//2]
  for c in list(product([0, 1], repeat=len(G))):
    if (c[0] == 1 and c[1] == 1) or (c[0] == 0 and c[1] == 1):
      continue
    else:
      candidates.append(c)


  for apx_sol in candidates:
      # ws = np.array(apx_sol[::-1])
      ws = np.array(apx_sol)

      if iso_in_dict(G, ws, iso_dict) or iso_in_dict(G, np.abs(ws-1), iso_dict):
          continue
      else:
          # hash of the warm start
          ws_hash = ws_hashing(ws)

          # check if hash is in dict
          if ws_hash in iso_dict.keys():
              iso_dict[ws_hash].append(ws)
          else:
              iso_dict[ws_hash] = [ws]
          warmstartings.append(ws)
          
  return np.array(warmstartings)


def generate_graph(left_degree: int, right_degree: int, num_merged_nodes: int) -> nx.Graph:
    """
    Generate a graph with a central edge. The central node is
    connected to `left_degree` nodes on one side and `right_degree` nodes
    on the other side. The number `num_merged_nodes` gives the number of
    nodes that will be merged.

    Input:
    ----------
    left_degree: int, the number of nodes connected to the central node
    on the left side of the graph.
    right_degree: int, the number of nodes connected to the central node
    on the right side of the graph.
    num_merged_nodes: int, the number of nodes to be merged.

    Output:
    ----------
    Graph:  A Graph instance representing the generated graph.

    Note: The graph is generated by creating a central edge, and
    then adding `left_degree` nodes on one side and `right_degree` nodes
    on the other side.
    """

    # Account for central edge
    left_degree -= 1
    right_degree -= 1

    # Create a Graph object to represent the graph.
    G = Graph()

    # Create the central edge between the two nodes.
    edges = [(0, 1)]  

    # Add the left neighbors of the central node to the graph.
    edges.extend([(0, k+2) for k in range(left_degree)]) 

    # Reuse existing nodes for edges
    n = left_degree + 2 - num_merged_nodes

    # Add the right neighbors of the central node to the graph.
    edges.extend([(1, k+n) for k in range(right_degree)]) 

    # Add all the edges to the graph.
    G.add_edges_from(edges)
    nx.convert_node_labels_to_integers(G)

    return G


def is_iso(G1, G2, ws1=None, ws2=None):
    # Prepare graphs
    # Check if attributes are already set
    if not G1.nodes[0]:
      G1 = G1.copy()
      attrs = {i: ws1[i] for i in G1.nodes}
      nx.set_node_attributes(G1, attrs, 'weight')
      # mark central edge
      attrs = {e: 0 for e in G1.edges}
      attrs[(0,1)] = 1
      nx.set_edge_attributes(G1, attrs, 'central')

    if not G2.nodes[0]:
      G2 = G2.copy()
      attrs = {i: ws2[i] for i in G2.nodes}
      nx.set_node_attributes(G2, attrs, 'weight')
      # mark central edge
      attrs = {e: 0 for e in G2.edges}
      attrs[(0,1)] = 1
      nx.set_edge_attributes(G2, attrs, 'central')

    G2_inverted = G2.copy()

    # Graph with inverse ws
    attrs_inv = {i: np.abs(ws2[i]-1) for i in G2_inverted.nodes}
    nx.set_node_attributes(G2_inverted, attrs_inv, 'weight')

    # iso check
    comp = lambda g1, g2: g1['weight'] == g2['weight']
    ecomp = lambda g1, g2: g1['central'] == g2['central']

    return nx.is_isomorphic(G1, G2, node_match=comp, edge_match=ecomp) or nx.is_isomorphic(G1, G2_inverted, node_match=comp, edge_match=ecomp)