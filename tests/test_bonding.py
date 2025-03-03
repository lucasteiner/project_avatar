import pytest
import numpy as np
from bonding import Bonding
from molecule import Molecule
import os


symbols = np.array(["H", "O", "H"])
coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

bonding = Bonding(symbols, coordinates)
print(bonding.distance_matrix)
print(bonding.bond_matrix)

bonded_indices = bonding.get_bonded_atoms(1)
print(bonded_indices)  # Output: [0, 2] (H atoms bonded to O)
