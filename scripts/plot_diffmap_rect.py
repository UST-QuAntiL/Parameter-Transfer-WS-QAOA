import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from calculations import *
from plotting import *
from graph_management import *


energy_grids = sorted(os.listdir('./energies/ordered'))
n = len(energy_grids)

transfer_map = np.load('./transferability-map.npy')
diff_map = np.load('./diff-map.npy')

# plotting
# fig, ax = plt.subplots(figsize=(7.166, 5.95))
# plt.subplots_adjust(left=0.04, right=1.06, top=0.95, bottom=0.04)
fig, ax = plt.subplots(figsize=(7.166, 6.3))
plt.subplots_adjust(left=0.04, right=1.08, top=0.925, bottom=0.04)
ax.set_xlabel('Donor subgraph', fontsize=10)
ax.set_ylabel('Acceptor subgraph', fontsize=10)
img = ax.imshow(transfer_map, cmap='inferno', interpolation='nearest')  # transferability
# img = ax.imshow(diff_map, cmap='inferno_r', interpolation='nearest')  # avg-diff
cbar = plt.colorbar(img, aspect=40)

# Display x,y coordinated in interactive plot
ax.fmt_ydata = lambda y: f'{y:.0f}'
ax.fmt_xdata = lambda x: f'{x:.0f}'

# line seperators
location = -0.5  # seperator between pixels
label_location = 0
ticks = []
labels = []

# (j-k)-m
for j in range(1, 6):
    for k in range(j, 6):
        for m in range(j):
            landscape_names = sorted([name for name in energy_grids if f'({j}-{k})-{m}' in name])
            location += len(landscape_names)
            label_location += len(landscape_names)
            if location < len(energy_grids) - 1 and m < (j-1):  # skip last lines and degree changes
                # Separate m merged by dotted line
                ax.axvline(x=location, color='white', linestyle=':', linewidth=0.6)
                ax.axhline(y=location, color='white', linestyle=':', linewidth=0.6)

        # Separate k degree by dashed line
        ax.axvline(x=location, color='white', linestyle='--', linewidth=0.7)
        ax.axhline(y=location, color='white', linestyle='--', linewidth=0.7)

        ticks.append(location - label_location/2)
        labels.append(f'({j}-{k})')
        label_location = 0
    # Separate j degrees by solid line
    ax.axvline(x=location, color='white', linestyle='-', linewidth=0.7)
    ax.axhline(y=location, color='white', linestyle='-', linewidth=0.7)

xlabels = ['(1-$*$)','','','','', '(2-$*$)','','',''] + labels[9:]
ylabels = ['(1-$*$)','','','',''] + labels[5:]
plt.xticks(ticks, xlabels)
plt.yticks(ticks, ylabels)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
cbar.set_label('Transferability Coefficient', labelpad=-42)

lw = 2.5
ax.add_patch(Rectangle(xy=(204.5,204.5), width=30, height=30, linewidth=lw, edgecolor='cyan', facecolor='None'))
ax.annotate('A', xy=(219.5, 566), xytext=(219.5, 570), ha='center', va='center', xycoords='data', annotation_clip=False)
ax.add_patch(Rectangle(xy=(277.5,333.5), width=56, height=40, linewidth=lw, edgecolor='cyan', facecolor='None'))
ax.annotate('B', xy=(306.5, 566), xytext=(306.5, 570), ha='center', va='center', xycoords='data', annotation_clip=False)
ax.add_patch(Rectangle(xy=(373.5,333.5), width=18, height=120, linewidth=lw, edgecolor='cyan', facecolor='None'))
ax.annotate('C', xy=(381.5, 566), xytext=(381.5, 570), ha='center', va='center', xycoords='data', annotation_clip=False)

#ax.annotate('A', xy=(353.5, 566), xytext=(353.5, 592),
#            ha='center', va='bottom', xycoords='data', annotation_clip=False,
#            arrowprops=dict(arrowstyle='-[, widthB=1.4, lengthB=.4', lw=1.0))
#ax.annotate('A', xy=(564, 468.5), xytext=(582, 468.5),
#            ha='center', va='center', xycoords='data', annotation_clip=False,
#            arrowprops=dict(arrowstyle='-[, widthB=1.05, lengthB=.4', lw=1.0))
plt.savefig('transferability_map_rect.pdf')
#plt.show()
