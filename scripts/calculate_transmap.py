import os
import numpy as np

from calculations import *


energy_grids = sorted(os.listdir('./energies/ordered'))
n = len(energy_grids)

transfer_map = np.empty((n, n))
diff_map = np.empty((n, n))

for i in range(n):
  acceptor = np.load(f'./energies/ordered/{energy_grids[i]}')
  for j in range(n):
    donor = np.load(f'./energies/ordered/{energy_grids[j]}')
    transfer_map[i,j] = transferability_coeff(donor, acceptor)
    diff_map[i,j] = average_difference(donor, acceptor)

np.save('./test-run/transferability-map.npy', transfer_map)
np.save('./test-run/diff-map.npy', diff_map)