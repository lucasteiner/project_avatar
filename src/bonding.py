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
        self.compute_bond_connectivity()
        
    def compute_bond_connectivity(self):
        """
        Compute bond connectivity based on covalent radii and coordinates.
        """
        num_atoms = len(self.symbols)
        radii = np.array([covalent_radii.get(symbol, 0.0) for symbol in self.symbols])
        coords = self.coordinates
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                distance = np.linalg.norm(coords[i] - coords[j])
                cutoff = (radii[i] + radii[j]) * self.tolerance
                if distance <= cutoff:
                    self.bonds.append((i, j))
                    
    def get_bonds(self):
        """
        Return the list of bonds.

        Returns:
        list of tuples: Each tuple contains indices of bonded atoms.
        """
        return self.bonds

    def to_networkx_graph(self):
        """
        Convert the molecule to a NetworkX graph.

        Returns:
        networkx.Graph: Graph representation of the molecule.
        """
        G = nx.Graph()
        num_atoms = len(self.symbols)
        for i in range(num_atoms):
            G.add_node(i, symbol=self.symbols[i])
        for bond in self.bonds:
            i, j = bond
            G.add_edge(i, j)
        return G

    @staticmethod
    def get_atom_mapping(mol1, mol2):
        """
        Get atom mapping between two molecules.

        Parameters:
        mol1 (MoleculeBonding): First molecule.
        mol2 (MoleculeBonding): Second molecule.

        Returns:
        dict or None: Mapping from mol1 atom indices to mol2 atom indices if isomorphic, else None.
        """
        G1 = mol1.to_networkx_graph()
        G2 = mol2.to_networkx_graph()

        # Include the index in the node_match function
        def node_match(n1, n2):
            return n1['symbol'] == n2['symbol']

        GM = nx.isomorphism.GraphMatcher(G1, G2, node_match=node_match)
        if GM.is_isomorphic():
            mapping = GM.mapping
            return mapping
        else:
            return None


    @staticmethod
    def reorder_coordinates(mapping, coordinates):
        """
        Reorder the coordinates of mol2 to match the atom ordering of mol1.
    
        Parameters:
        mapping (dict): Atom mapping from mol1 indices to mol2 indices.
        coordinates (np.ndarray): Coordinates of mol2.
    
        Returns:
        np.ndarray: Reordered coordinates of mol2 matching mol1's atom ordering.

        Caveate: This function fails for everything which is more complicated than bijections.
        """
        # Create an array of indices corresponding to mol2's atom indices
        indices = np.array([mapping[i] for i in sorted(mapping.keys())])
        # Reorder the coordinates using numpy indexing
        reordered_coords = coordinates[indices]
        return reordered_coords

#    @staticmethod
#    def get_atom_mapping(mol1, mol2):
#        """
#        Get atom mapping between two molecules.
#
#        Parameters:
#        mol1 (MoleculeBonding): First molecule.
#        mol2 (MoleculeBonding): Second molecule.
#
#        Returns:
#        dict or None: Mapping from mol1 atom indices to mol2 atom indices if isomorphic, else None.
#        """
#        G1 = mol1.to_networkx_graph()
#        G2 = mol2.to_networkx_graph()
#        # Define a node_match function to match atoms by symbol
#        def node_match(n1, n2):
#            return n1['symbol'] == n2['symbol']
#        GM = nx.isomorphism.GraphMatcher(G1, G2, node_match=node_match)
#        if GM.is_isomorphic():
#            mapping = GM.mapping  # mapping from G1 nodes to G2 nodes
#            return mapping
#        else:
#            return None

