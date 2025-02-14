import pytest
import numpy as np
from bonding import Bonding
from molecule import Molecule
import os


symbols = np.array(["H", "O", "H"])
coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

bonding = Bonding(symbols, coordinates)
dist_matrix = bonding.compute_distance_matrix()
print(dist_matrix)
bond_matrix = bonding.compute_bond_matrix()
print(bond_matrix)

bonded_indices = bonding.get_bonded_atoms(1)
print(bonded_indices)  # Output: [0, 2] (H atoms bonded to O)
