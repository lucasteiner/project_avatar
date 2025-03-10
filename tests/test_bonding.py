import pytest
import numpy as np
from bonding import Bonding
from molecule import Molecule
import os


symbols = np.array(["H", "O", "H"])
coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [1.1, 0.0, 1.1]])

bonding = Bonding(symbols, coordinates)

bonded_indices = bonding.get_bound_indices(1)
