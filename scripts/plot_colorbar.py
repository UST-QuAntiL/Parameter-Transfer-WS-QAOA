import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Use Latex font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})

height_adjust = 0.12
fig = plt.figure(figsize=(7.166, 7.166 * height_adjust))
cbar_ax = fig.add_axes([0.12, 0.05 / height_adjust, 0.76, 0.04 / height_adjust])

cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', cmap='inferno')
cbar_ax.set_xticks([0,1],labels=['low', 'high'], fontsize=20)

plt.show()