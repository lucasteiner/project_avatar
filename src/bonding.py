import numpy as np
import networkx as nx
from src.config import covalent_radii
from collections import Counter

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
        self.distance_matrix = self.compute_distance_matrix()
        self.bond_matrix = self.compute_bond_matrix()

    def distance(self, atom_index1, atom_index2):
        """
        Calculate the distance between two atoms.

        Parameters:
        atom_index1 (int): Index of the first atom.
        atom_index2 (int): Index of the second atom.

        Returns:
        float: The Euclidean distance between the two atoms.
        """
        if atom_index1 >= len(self.symbols) or atom_index2 >= len(self.symbols):
            raise IndexError("Atom index out of range.")
        coord1 = self.coordinates[atom_index1]
        coord2 = self.coordinates[atom_index2]
        return np.linalg.norm(coord1 - coord2)

    def bond_angle(self, atom_index1, atom_index2, atom_index3):
        """
        Calculate the bond angle formed by three atoms.

        Parameters:
        atom_index1 (int): Index of the first atom.
        atom_index2 (int): Index of the central atom.
        atom_index3 (int): Index of the third atom.

        Returns:
        float: The bond angle in degrees.
        """
        if any(index >= len(self.symbols) for index in [atom_index1, atom_index2, atom_index3]):
            raise IndexError("Atom index out of range.")
        # Vectors from central atom to the two other atoms
        vec1 = self.coordinates[atom_index1] - self.coordinates[atom_index2]
        vec2 = self.coordinates[atom_index3] - self.coordinates[atom_index2]
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        # Compute angle
        cos_theta = np.dot(vec1_norm, vec2_norm)
        # Clamp cos_theta to [-1, 1] to avoid numerical errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        return angle

    def dihedral_angle(self, atom_index1, atom_index2, atom_index3, atom_index4):
        """
        Calculate the dihedral angle defined by four atoms.

        Parameters:
        atom_index1 (int): Index of the first atom.
        atom_index2 (int): Index of the second atom.
        atom_index3 (int): Index of the third atom.
        atom_index4 (int): Index of the fourth atom.

        Returns:
        float: The dihedral angle in degrees.
        """
        if any(index >= len(self.symbols) for index in [atom_index1, atom_index2, atom_index3, atom_index4]):
            raise IndexError("Atom index out of range.")

        p0 = self.coordinates[atom_index1]
        p1 = self.coordinates[atom_index2]
        p2 = self.coordinates[atom_index3]
        p3 = self.coordinates[atom_index4]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # Normalize b1 so that it does not influence magnitude of vector products
        b1 /= np.linalg.norm(b1)

        # Compute vectors normal to the planes
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)

        angle = np.degrees(np.arctan2(y, x))
        return angle

    def get_coordinates_by_symbol(self, element_symbol):
        """
        Return the coordinates of all atoms where the symbol matches the given element symbol.
    
        Parameters:
        element_symbol (str): The element symbol to search for (e.g., 'H', 'O', 'C').
    
        Returns:
        list of np.ndarray: A list of coordinates (np.array) for atoms matching the given symbol.
        """
        # Ensure the element symbol is properly formatted (e.g., 'C', 'O', etc.)
        element_symbol = element_symbol.capitalize()
    
        # Gather all coordinates where the atomic symbol matches the given element symbol
        matching_coords = [
            coord for symbol, coord in zip(self.symbols, self.coordinates) if symbol == element_symbol
        ]

        return matching_coords
 
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
    
    def get_bound_atoms(self, atom_index):
        """
        Get the indices, symbols, and bond lengths of all atoms bonded to a given atom.

        Parameters:
        atom_index (int): Index of the atom.
 
        Returns:
        list of tuples: Each tuple contains (bonded_atom_index, bonded_atom_symbol, bond_length).
        """
        #bond_matrix = self.compute_bond_matrix()  # Compute bond matrix
        #distance_matrix = self.compute_distance_matrix()  # Compute distance matrix
 
        # Get indices where bonds exist
        bonded_indices = np.where(self.bond_matrix[atom_index] == 1)[0]
 
        # Create a list of tuples (index, symbol, bond length)
        return [(i, self.symbols[i], self.distance_matrix[atom_index, i]) for i in bonded_indices]

    def get_bound_indices(self, atom_index, element_symbol=None):
        """_summary_

        Args:
            atom_index (_type_): _description_
            element_symbol (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Do the thing
        bound_indices = np.where(self.bond_matrix[atom_index] == 1)[0]

        # More flexible by element symbol specific filtering 
        if element_symbol:
            symbols = self.get_symbols_by_indices(bound_indices)
            bound_indices = [bound_index for bound_index, symbol in zip(bound_indices, symbols) if symbol == element_symbol]
        # print('get_bound_indices:', type(bound_indices[0]))
        # print('get_bound_indices:', type(bound_indices))
        return bound_indices
 
    def get_common_binding_partners(self, index1: int, index2: int):
        """
        Returns all common binding partners of two atoms given their indices.
    
        Parameters:
        - index1 (int): Index of the first atom.
        - index2 (int): Index of the second atom.
    
        Returns:
        - list: Indices of atoms that are bonded to both index1 and index2.
        """
        # Get bonded atoms for each index
        bonded_to_1 = set(self.get_bound_indices(index1))
        bonded_to_2 = set(self.get_bound_indices(index2))
        #bonded_to_1 = set(np.where(self.bond_matrix[index1] == 1)[0])
        #bonded_to_2 = set(np.where(self.bond_matrix[index2] == 1)[0])
        
        # Find common binding partners
        common_partners = bonded_to_1.intersection(bonded_to_2)
    
        return list(common_partners)

    def get_coordinates_by_indices(self, indices):
        """Gets all symbols corresponding to a list of indices

        Args:
            indices (list): list of atomic indices

        Returns:
            coordinates (list): list of element symbols
        """
        return np.array([self.coordinates[index] for index in indices], dtype=object)

    def get_symbols_by_indices(self, indices: list):
        """Gets all symbols corresponding to a list of indices

        Args:
            indices (list): list of atomic indices

        Returns:
            symbols (list): list of element symbols
        """
        return [self.symbols[index] for index in indices]

    def get_indices_by_symbol(self, element_symbol):
        """
        Get the indices of all atoms with a given element symbol.
 
        Parameters:
        element_symbol (str): The atomic symbol to match.
 
        Returns:
        list: Indices of atoms that match the given symbol.
        """
        #print('get_indices_by_symbol:', list(np.where(self.symbols == element_symbol)[0]))
        return list(np.where(self.symbols == element_symbol)[0])
   
    def get_terminated_fragment(self, start_index: int, terminators: int or list):
        """Helping to call define_terminated_fragment

        Args:
            start_index (int): Chose a starting index from which bound indices get collected until terminators are found
            terminators (intorlist): Index or indices of atoms which terminate a certain fragment

        Returns:
            list: indices of the fragment
        """
        return list(self._define_terminated_fragment(start_index, {terminators} if isinstance(terminators, (int, np.int64)) else set(terminators)))

    def _define_terminated_fragment(self, start_index, fragment):
        """
        Recursively collects all connected atoms into a fragment.
    
        This function expands the given fragment set by adding all atoms 
        that are directly or indirectly bonded to the specified start_index.
    
        Parameters:
        - start_index (int): The index of the atom from which the search starts.
        - fragment (set): A set of atom indices that define the fragment.
          - Indices already in the set act as termination points.
          - The function modifies this set in place by adding all connected atoms.
    
        Returns:
        - set: The updated set containing all bonded atoms connected to start_index.
        """
        fragment.add(start_index)
        for index in list(self.get_bound_indices(start_index)):
            if index not in fragment:
                self._define_terminated_fragment(index, fragment)
        return fragment

    def get_coc_patterns(self):
        """Identifies ether groups
 
        Returns:
            ether_oxygens(list): all oxygen atoms which bind exactly two carbons
        """
        oxygens = self.get_indices_by_symbol('O')
        ether_oxygens = []
        for oxygen in oxygens:
            c_indices = self.get_bound_indices(oxygen, 'C')
            if len(c_indices) == 2:
                ether_oxygens.append(oxygen)
        return ether_oxygens

    def get_co_patterns(self):
        """Identifies alcohols, carbonyls, enols
        Rejects ethers
 
        Returns:
            carbon_binding_oxygens (list): Contains all oxygens which bind exactly one carbon atom
            carbons (list): Index of the corresponding carbon atoms
        """
        oxygens = self.get_indices_by_symbol('O')
        #print('co, o:', oxygens[0], type(oxygens[0]))
        carbon_binding_oxygens = []
        carbons = []
        for oxygen in oxygens:
            bound_carbons = self.get_bound_indices(oxygen, 'C')
            if len(bound_carbons) == 1:
                carbon_binding_oxygens.append(oxygen)
                carbons.append(bound_carbons[0])
                #print('co, c:', carbons[-1], type(carbons[-1]))
        return carbon_binding_oxygens, carbons

    @staticmethod
    def get_indices_of_symbol_in_list_of_symbols(indices, symbols, symbol):
        """Checks a list of symbols for matches and returns corresponding indices

        Args:
            indices (list of ints): List of indices
            symbols (list of strings): List of element symbols
            symbol (string): Element symbol to match

        Returns:
            list of ints: Indices of matching symbols
        """
        symbol.capitalize()
        return [i for i, x in zip(indices, symbols) if x == symbol]

    def count_bound_atoms(self, index, element_symbol=None):
        """counts all hydrogen atoms bound to the indexed atom

        Args:
            index (int): atom index

        Returns:
            count (int): the number of bound hydrogen atoms
        """
        return len(self.get_bound_indices(index, element_symbol=element_symbol))

    def get_acetone_fragments(self):
        """Finds patterns corresponding to acetone (propan-2-one)

        Returns:
            fragments (list of lists): Every entry corresponds to all indices of one acetone fragment 
        """
        fragments = []
        oxygens, carbons = self.get_co_patterns()
        #print('o, c:', oxygens, carbons)
        for oxygen, carbon in zip(oxygens, carbons):
            bound_atoms = self.get_bound_indices(carbon)
            bound_carbons = self.get_bound_indices(carbon, 'C')
            #print('bound carbons:', bound_carbons)
            # includes oxygen
            h_count = [self.count_bound_atoms(carbon, 'H') == 3 for carbon in bound_carbons]
            #print('h count:', h_count)

            # true if number of binding carbons is 2 and both bind 3 hydrogens
            #print('topology_list:', [len(bound_carbons) == 2, len(bound_atoms) == 3] + h_count)
            topology = all([len(bound_carbons) == 2, len(bound_atoms) == 3] + h_count)
            #print('topology:', topology)

            # Returns only OC(-R)x but not X-OC(-R)x 
            #print('Int64?:,', carbon, oxygen)
            #print('Int64?:,', type(carbon), type(oxygen))
            acetone_indices = self.get_terminated_fragment(carbon, oxygen)
            acetone_symbols = self.get_symbols_by_indices(acetone_indices)
            counts = Counter(acetone_symbols)
            sum_formula = all([counts['O'] == 1, counts['C'] == 3, counts['H'] == 6])
            #print('sum_formula:', sum_formula)

            # Excludes wrong sum formula and isomers like propan-1-one
            if sum_formula and topology:
                fragments.append(acetone_indices)
        #print('fragments:', fragments)
        return fragments

    def get_thf_fragments(self):
        """Finds patterns corresponding to tetrahydrofuran (THF)

        Returns:
            fragments (list of lists): Every entry corresponds to a list of all indices of one THF fragment 
        """
        fragments = []
        ether_oxygens = self.get_coc_patterns()
        for oxygen in zip(ether_oxygens):
            carbon_indices = self.get_bound_indices(oxygen, 'C')

            # Returns only OC(-R)x but not X-OC(-R)x 
            thf_indices = self.get_terminated_fragment(carbon_indices[0], oxygen)
            thf_symbols = self.get_symbols_by_indices(thf_indices)
            carbon_indices = self.get_indices_of_symbol_in_list_of_symbols(thf_indices, thf_symbols, 'C')
            # checks if all carbons bind exactly 2 hydrogen atoms
            check_hydrogens = all([self.count_bound_atoms(index, 'H') == 2 for index in carbon_indices])
            counts = Counter(thf_symbols)
            # Excludes by wrong sum formula only
            if counts['O'] == 1 and counts['C'] == 4 and counts['H'] == 8 and check_hydrogens:
                fragments.append(thf_indices)
        return fragments
