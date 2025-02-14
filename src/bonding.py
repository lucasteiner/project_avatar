import numpy as np
import networkx as nx
from src.config import covalent_radii

class Bonding:
    
    def __init__(self, symbols, coordinates, tolerance=1.2):
        """
        Initialize the MoleculeBonding class.

        Parameters:
        symbols (np.ndarray): Array of atomic symbols.
        coordinates (np.ndarray): Array of atomic coordinates.
        tolerance (float): Multiplicative factor for covalent radii to determine bond lengths.
        """
        self.symbols = symbols
        self.coordinates = coordinates
        self.tolerance = tolerance
        self.bonds = []
        
    def compute_distance_matrix(self):
        """
        Compute the distance matrix for the molecule.

        Returns:
        np.ndarray: A square matrix where element (i, j) represents the distance between atom i and atom j.
        """
        diff = self.coordinates[:, np.newaxis, :] - self.coordinates[np.newaxis, :, :]
        distance_matrix = np.linalg.norm(diff, axis=-1)
        return distance_matrix
    
    def compute_bond_matrix(self):
        """
        Compute the bond matrix based on atomic distances and covalent radii.

        Returns:
        np.ndarray: A square matrix where element (i, j) is 1 if a bond exists, 0 otherwise.
        """
        distance_matrix = self.compute_distance_matrix()
        
        # Get covalent radii for each atom
        radii = np.array([covalent_radii[symbol] for symbol in self.symbols])
        
        # Compute sum of covalent radii for each atom pair
        bond_thresholds = (radii[:, np.newaxis] + radii[np.newaxis, :]) * self.tolerance
        
        # Determine bonding based on distance comparison
        bond_matrix = (distance_matrix <= bond_thresholds) & (distance_matrix > 0)
        
        return bond_matrix.astype(int)  # Convert boolean matrix to integer (1/0)
    
    def get_bonded_atoms(self, atom_index):
        """
        Get the indices, symbols, and bond lengths of all atoms bonded to a given atom.
 
        Parameters:
        atom_index (int): Index of the atom.
 
        Returns:
        list of tuples: Each tuple contains (bonded_atom_index, bonded_atom_symbol, bond_length).
        """
        bond_matrix = self.compute_bond_matrix()  # Compute bond matrix
        distance_matrix = self.compute_distance_matrix()  # Compute distance matrix
 
        # Get indices where bonds exist
        bonded_indices = np.where(bond_matrix[atom_index] == 1)[0]
 
        # Create a list of tuples (index, symbol, bond length)
        bonded_atoms = [
            (i, self.symbols[i], distance_matrix[atom_index, i]) for i in bonded_indices
        ]
 
        return bonded_atoms
