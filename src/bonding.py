import numpy as np
import networkx as nx

class Bonding:
    covalent_radii = {
        'H': 0.31,
        'He': 0.28,
        'Li': 1.28,
        'Be': 0.96,
        'B': 0.84,
        'C': 0.76,
        'N': 0.71,
        'O': 0.66,
        'F': 0.57,
        'Ne': 0.58,
        'Na': 1.66,
        'Mg': 1.41,
        'Al': 1.21,
        'Si': 1.11,
        'P': 1.07,
        'S': 1.05,
        'Cl': 1.02,
        'Ar': 1.06,
        'K': 2.03,
        'Ca': 1.76,
        'Sc': 1.70,
        'Ti': 1.60,
        'V': 1.53,
        'Cr': 1.39,
        'Mn': 1.39,
        'Fe': 1.32,
        'Co': 1.26,
        'Ni': 1.24,
        'Cu': 1.32,
        'Zn': 1.22,
        'Ga': 1.22,
        'Ge': 1.20,
        'As': 1.19,
        'Se': 1.20,
        'Br': 1.20,
        'Kr': 1.16,
        'Rb': 2.20,
        'Sr': 1.95,
        'Y': 1.90,
        'Zr': 1.75,
        'Nb': 1.64,
        'Mo': 1.54,
        'Tc': 1.47,
        'Ru': 1.46,
        'Rh': 1.42,
        'Pd': 1.39,
        'Ag': 1.45,
        'Cd': 1.44,
        'In': 1.42,
        'Sn': 1.39,
        'Sb': 1.39,
        'Te': 1.38,
        'I': 1.39,
        'Xe': 1.40,
        'Cs': 2.44,
        'Ba': 2.15,
        'La': 2.07,
        'Ce': 2.04,
        'Pr': 2.03,
        'Nd': 2.01,
        'Pm': 1.99,
        'Sm': 1.98,
        'Eu': 1.98,
        'Gd': 1.96,
        'Tb': 1.94,
        'Dy': 1.92,
        'Ho': 1.92,
        'Er': 1.89,
        'Tm': 1.90,
        'Yb': 1.87,
        'Lu': 1.87,
        'Hf': 1.75,
        'Ta': 1.70,
        'W': 1.62,
        'Re': 1.51,
        'Os': 1.44,
        'Ir': 1.41,
        'Pt': 1.36,
        'Au': 1.36,
        'Hg': 1.32,
        'Tl': 1.45,
        'Pb': 1.46,
        'Bi': 1.48,
        'Po': 1.40,
        'At': 1.50,
        'Rn': 1.50,
        'Fr': 2.60,
        'Ra': 2.21,
        'Ac': 2.15,
        'Th': 2.06,
        'Pa': 2.00,
        'U': 1.96,
        'Np': 1.90,
        'Pu': 1.87,
        'Am': 1.80,
        'Cm': 1.69
    }
    
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
        radii = np.array([self.covalent_radii.get(symbol, 0.0) for symbol in self.symbols])
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

