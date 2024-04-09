import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx

from matplotlib.ticker import FormatStrFormatter
from calculations import *

# Use Latex font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})


def plot_energy(energy_grid, gammaMax=2*np.pi, betaMax=np.pi, title=None, axes=None, filename=None, show=True):
  if axes is None:
    fig, ax = plt.subplots()
  else: 
    ax = axes

  fontsize = 20
  ax.set_title(title, fontsize=fontsize)

  img = ax.imshow(energy_grid, cmap='inferno', origin='lower', extent=[0, betaMax, 0, gammaMax])
  # cbar = plt.colorbar(img, ax=ax, fraction=0.0458, pad=0.04)
  # cbar.ax.tick_params(labelsize=fontsize)

  ax.set_aspect(betaMax/gammaMax)
  # ax.set_xlabel(r'$\beta$')
  # ax.set_ylabel(r'$\gamma$')
  ax.set_xticks(np.linspace(0, betaMax, 3), labels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
  ax.set_yticks(np.linspace(0, gammaMax, 3), labels=[r'$0$', r'$\pi$', r'$2\pi$'], fontsize=fontsize)
  # ax.set_xticks([])
  # ax.set_yticks([])

  if filename is not None:
    plt.savefig(f'{filename}_energy-landscape.pdf')#, dpi=300)
    plt.close()
  else:
    if show:
        plt.show()
    else:
        pass

  return img


def plot_energy_with_marker(energy_grid, gammaMax=2*np.pi, betaMax=np.pi, marker='max', title=None, axes=None, filename=None, show=True, a=1.0):
  thresh= a*energy_grid.max() + (1-a)*energy_grid.min() - 1e-5
  if marker == 'max':
    gam_idx, bet_idx = np.where((energy_grid >= thresh)) 
  elif marker == 'min':
    gam_idx, bet_idx = np.where((energy_grid <= energy_grid.max()+1e-5)) 
  else:
    gam_idx, bet_idx = np.where((energy_grid >= energy_grid.max()-1e-5) | (energy_grid <= energy_grid.min()+1e-5))

  gam = gam_idx + .5  # add half a pixel
  bet = bet_idx + .5
  gam *= 2*np.pi/energy_grid.shape[0]  # adjust scale
  bet *= 1*np.pi/energy_grid.shape[1]

  if axes is None:
    fig, ax = plt.subplots()
  else: 
    ax = axes

  fontsize=10
  ax.set_title(title)
  vmin, vmax = .18, .94
  vmin, vmax = None, None  # relative color scale
  img = ax.imshow(energy_grid, cmap='inferno', vmin=vmin, vmax=vmax, origin='lower', extent=[0, betaMax, 0, gammaMax])
  cbar = plt.colorbar(img, ax=ax, fraction=0.0458, pad=0.04)
  cbar.ax.tick_params(labelsize=fontsize)
  ax.scatter(bet, gam, s=120, linewidths=2.0, facecolors='none', color='deepskyblue')
  

  ax.set_aspect(betaMax/gammaMax)
  ax.set_xlabel(r'$\beta$')
  ax.set_ylabel(r'$\gamma$')
  ax.set_xticks(np.linspace(0, betaMax, 3), labels=[r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=fontsize)
  ax.set_yticks(np.linspace(0, gammaMax, 3), labels=[r'$0$', r'$\pi$', r'$2\pi$'], fontsize=fontsize)
  if filename is not None:
    plt.savefig(f'{filename}_energy-landscape.pdf')#, dpi=300)
    plt.close()
  else:
    if show:
        plt.show()
    else:
        pass


def save_contents(G, qaoa_qc, energy_grid, name, hyperparams=None, folder=None):
  save = input(f'save to /{folder}/{name}? previous data is overwritten! y/n\n')
  if save != 'y':
    return
  #drive_path = '/content/drive/MyDrive/MA/'
  drive_path = './'
  if folder is not None:
    os.makedirs(drive_path + folder, exist_ok=True)
    path = drive_path + f'{folder}/{name}'
  else:
    path = drive_path + name
  nx.write_weighted_edgelist(G, f'{path}.graph')
  nx.draw(G, with_labels=False)
  plt.savefig(f'{path}_graph.png', dpi=150)
  # transpile(qaoa_qc, basis_gates=['h', 'rz', 'rx', 'cx']).draw('mpl', filename=f'{path}_qaoa.png')
  np.save(f'{path}_energy.npy', energy_grid)
  plot_energy(energy_grid, filename=path)


""" 3D-Plot """
def plot_3d(energy_grid):
  samples = 65
  gammas, betas = np.mgrid[0:2*np.pi:samples*1j, 0:np.pi:samples*1j]
  fig, ax = plt.subplots(dpi=150, subplot_kw={"projection": "3d"})
  ax.view_init(elev=30, azim=30)
  surf = ax.plot_surface(gammas, betas, energy_grid, cmap='inferno',
                         linewidth=0, antialiased=False)
  plt.show()


""" graph drawing """
def node_color_mapping(G, cut):
  """ returns a color mapping where the nodes are colored by partition."""
  if isinstance(cut, str):
    cut = cut[::-1]  # reverse cut
  color_map =[]
  for x in G.nodes():
    if cut[x] == '1' or cut[x] == 1:
      # color_map.append('red')
      color_map.append('#ffd500')
    else:
      # color_map.append('#1f78b4')  # default networkx color
      color_map.append('#6b00c2')  # default networkx color
  return color_map


def edge_color_mapping(G, cut):
  """ returns a color mapping where edges belonging to the cut are highlighted."""
  edges = G.edges()
  if isinstance(cut, str):
    cut = cut[::-1]  # reverse cut
  edge_color = []
  for u, v in edges:
    if cut[u] != cut[v]:
      edge_color.append('darkorange')
    else:
      edge_color.append('k')  # default networkx edge color
  return edge_color


def draw_graph_with_cut(G, cut, draw_labels=False, show=True, **kwargs):
  node_colors = node_color_mapping(G, cut)
  # edge_colors = edge_color_mapping(G, cut)
  edge_colors = ['k']*len(G.edges)
  
  nx.draw_kamada_kawai(G, node_color=node_colors, edge_color=edge_colors, width=0.7, with_labels=draw_labels, **kwargs)
  if show: plt.show()


def draw_graph_with_ws(G, warmstarting=None, draw_labels=True, show=True, axes=None, **kwargs):
  """
  Draws a graph with warm-starting and optional attributes.

  Parameters:
  G (nx.Graph): The graph object to draw.
  warmstarting (np.ndarray): An optional numpy array that specifies the warmstarting.
  draw_labels (bool): If True, labels are drawn on nodes.
  show (bool): If True, the resulting plot is shown. Otherwise, the plot is hidden.
  axes (optional) (matplotlib.pyplot.Axes): The matplotlib object to use for plotting. Useful if multiple plots are placed on a figure.
  Additional keyword arguments: pass through any additional keyword arguments to the nx.draw_kamada_kawai function.

  """
  colors = None
  # Check if attributes are already set
  if G.nodes[0]:
    cmap = {0.: '#6b00c2', .5: '#1f78b4', 1.: '#ffd500'}
    colors = [cmap[G.nodes[n]['weight']] for n in G.nodes]

  if warmstarting is not None:
    # Check if warmstarting fits to graph
    if len(warmstarting) != len(G):
      raise ValueError('Invalid warmstarting for the given graph!')
    # Assign warmstarting to the nodes
    apx_sol = np.array(warmstarting)
    attrs = {i: apx_sol[i] for i in range(len(apx_sol))}
    nx.set_node_attributes(G, attrs, name='weight')
    # define colors
    cmap = {0.: '#6b00c2', .5: '#1f78b4', 1.: '#ffd500'}
    colors = [cmap[G.nodes[n]['weight']] for n in G.nodes]

  # Color central edge
  edge_colors = ['k']*len(G.edges)
  edge_colors[0] = 'darkorange'

  # plotting
  if axes is None:
      plt.figure()
  nx.draw_kamada_kawai(G, with_labels=draw_labels, node_color=colors, edge_color=edge_colors, font_color='k', width=1.4, ax=axes, **kwargs)

  # Show plot if specified
  if show: plt.show()


def draw_landscape_and_graph(energy_grid, G, warmstarting=None, title=None, show=True):
    """
    This function visualizes the energy landscape of a problem and the associated graph.
    Parameters:
    energy_grid (array-like): An array, providing the values for the landscape.
    G (nx.Graph): A sparse matrix or a graph-like object.
    warmstarting (array-like): An array with len(G) many entries that are either 0 or 1.
    title (str): An optional title for the plot. If `None`, the function will not include a title in the resulting figure.
    show (bool): An optional boolean variable indicating whether to show the resulting plot or not. Default is `True` (show)
    """
    # Initialize figure
    fig = plt.figure(title, figsize=(12,6))

    # Create two subplots
    left_ax = fig.add_subplot(1, 2, 1)
    right_ax = fig.add_subplot(1, 2, 2)

    # Plot the energy landscape on the left subplot
    plot_energy(energy_grid, axes=left_ax, show=False)

    # Draw the graph with warm-starting on the right subplot
    draw_graph_with_ws(G, warmstarting, axes=right_ax, show=show)


def draw_multiple_landscapes_and_graphs(path_list, rows=1, cols=1, figsize=(7.166, 7.166)):
    """
    Draws multiple energy landscapes and associated graphs from a list of energy grid file paths.
    Parameters:
    path_list: A list of file paths to the energy grid files.
    rows: Set the number of rows to draw. Defaults to 1.
    cols: Set the number of columns to draw. Defaults to 1.
    figsize: Set the size of the figure. Defaults to (7.166, 7.166)

    Returns: None.
    """
    # Space for graph & landscape
    rows *= 2

    # Initialize figure (\textwidth=7.166) (\columnwidth=3.5)
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    for i, path in enumerate(path_list):
      # Skip empty spots
      if path is None:
        continue

      # Draw graphs only every second row
      i += (i // cols) * cols
      # Subplot idx starts at 1
      i += 1
      # Get data
      G, ws, grid = get_graph_ws_and_grid(path)
      # draw_landscape_and_graph(grid, G, warmstarting=ws)

      # Create subplots for graph and energy
      graph_ax = fig.add_subplot(rows, cols, i)
      energy_ax = fig.add_subplot(rows, cols, i+cols)
      graph_ax.set_aspect(0.9)

      # Plot the energy landscape on the left subplot
      img = plot_energy(grid, axes=energy_ax, show=False)

      # Draw the graph with warm-starting on the right subplot
      additional_options = {'font_size': 6, 'node_size': 80}
      draw_graph_with_ws(G, ws, draw_labels=False, axes=graph_ax, show=False, **additional_options)