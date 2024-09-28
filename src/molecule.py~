#!/bin/env python3
import numpy as np
import re
from molmass import Formula, ELEMENTS
from collections import Counter
from mechanics import Mechanics
import math

class Atom:
    def __init__(self, symbol, x, y, z):
        self.symbol = symbol
        self.position = np.array([x, y, z])

    def distance_to(self, other_atom):
        return np.linalg.norm(self.position - other_atom.position)

    def __repr__(self):
        return (f"{self.symbol} {self.position}\n")


class Molecule:
    def __init__(self, name="Unnamed Molecule", energy=None):

        # Attributes
        self.name = name
        self.atoms = []

        # Read only
        self._dimension = None
        self.energy = energy

    def add_atom(self, atom):
        """
        Add an Atom object to the molecule and update the formula and dimension.
        """
        self.atoms.append(atom)
        self._update_formula()
        #self._update_dimension()

    def remove_atom(self, atom):
        """
        Remove an Atom object from the molecule and update the formula and dimension.
        """
        if atom in self.atoms:
            self.atoms.remove(atom)
            self._update_formula()
            #self._update_dimension()
        else:
            raise ValueError("Atom not found in the molecule.")

    def get_atom(self, index):
        """
        Get an Atom object by its index.
        :param index: The atom's index (int).
        :return: Atom object.
        """
        if 0 <= index < len(self.atoms):
            return self.atoms[index]
        else:
            raise IndexError("Atom index out of range.")

    def get_atoms_by_symbol(self, symbol):
        """
        Get all Atom objects that match a given element symbol.
        :param symbol: The element symbol (e.g., 'H').
        :return: List of Atom objects.
        """
        return [atom for atom in self.atoms if atom.symbol == symbol]

    def molecular_weight(self):
        return self.formula.mass if self.formula else 0

    def formula_str(self):
        return self.formula.formula if self.formula else "N/A"

    def center_of_mass(self):
        total_mass = sum(ELEMENTS[atom.symbol].mass for atom in self.atoms)
        weighted_positions = sum(ELEMENTS[atom.symbol].mass * atom.position for atom in self.atoms)
        return weighted_positions / total_mass if total_mass else np.array([0.0, 0.0, 0.0])

    def _generate_formula(self):
        """
        Generate the molecular formula based on current atoms.
        """
        element_symbols = [atom.symbol for atom in self.atoms]
        atom_counts = Counter(element_symbols)
        formula_str = ''.join(f"{symbol}{count if count > 1 else ''}" for symbol, count in atom_counts.items())
        return Formula(formula_str)

    def _update_formula(self):
        """
        Update the molecular formula after atoms are added or removed.
        """
        self.formula = self._generate_formula() if self.atoms else None



    # Property for vibrational_frequencies
    @property
    def vibrational_frequencies(self):
        return self._vibrational_frequencies

    @vibrational_frequencies.setter
    def vibrational_frequencies(self, frequencies):
        if not isinstance(frequencies, list):
            raise ValueError("Vibrational frequencies must be provided as a list.")
        if not all(isinstance(f, (int, float)) for f in frequencies):
            raise ValueError("All vibrational frequencies must be numeric values.")
        self._vibrational_frequencies = frequencies

    def calculate_mechanics(self, vibrational_frequencies=None, 
            temperature=298.15, pressure=1.0, volume=None, qRRHO=100, pos_freqs=80):
        """
        Calculate statistical mechanics for the molecule

        qRRHO (float): threshold in wavenumbers for treating low frequencies as free rotors
        pos_freqs (float): threshold in wavenumbers for treating low imaginary frequencies as positive frequencies 
        (Do not go below lowest frequency of transition states)
        """
        if volume is None:
            volume=self.volume

        if vibrational_frequencies is None:
            vibrational_frequencies=self.vibrational_frequencies
        else:
            self.vibrational_frequencies=vibrational_frequencies
        if self.vibrational_frequencies is None:
            raise ValueError(f"Expected vibrational frequencies in function calculate_mechanics, \n {print(self)}.")
        
        linear = self.is_linear(vibrational_frequencies)

        self.mechanics = Mechanics(self, linear, vibrational_frequencies, 
                temperature=298.15, pressure=1.0, volume=volume)

    @staticmethod
    def is_linear(frequencies):
        """
        Determine if a molecule is linear based on its vibrational frequencies.
    
        Parameters:
        frequencies (list of float): Vibrational frequencies.
    
        Returns:
        bool: True if the molecule is linear (5 zero frequencies),
              False if non-linear (6 zero frequencies).
    
        Raises:
        ValueError: If the number of zero frequencies is not 5 or 6.
        """
        num_zero_freqs = sum(math.isclose(freq, 0.0) for freq in frequencies)
        if num_zero_freqs == 5:
            return True  # Linear molecule
        elif num_zero_freqs == 6:
            return False  # Non-linear molecule
        else:
            raise ValueError(f"Unexpected number of zero frequencies ({num_zero_freqs}).")

    @property
    def dimension(self):
        self._dimension=self._update_dimension()
        return self._dimension

    def _update_dimension(self, linearity_threshold=1e-5):
        """
        Update the dimension of the molecule based on the number and arrangement of atoms.
        Determines if the molecule is linear or 3D.
    
        Parameters:
        linearity_threshold (float): Tolerance for determining linearity of the molecule.
                                     Default is 1e-5.
        """
        num_atoms = len(self.atoms)
    
        if num_atoms == 0:
            self._dimension = None
        elif num_atoms == 1:
            self._dimension = 1
        elif num_atoms == 2:
            self._dimension = 2
        else:
            # For 3 or more atoms, check if they are linear (dimension 2) or 3D (dimension 3)
            atom_positions = np.array([atom.position for atom in self.atoms])
    
            # Define the line using the first two atoms
            point_on_line = atom_positions[0]
            direction_vector = atom_positions[1] - atom_positions[0]
            norm_direction = np.linalg.norm(direction_vector)
    
            if norm_direction == 0:
                # First two atoms are at the same position; cannot define a line
                self._dimension = 3  # Considered 3D due to undefined direction
                return
    
            # Normalize the direction vector
            direction_unit = direction_vector / norm_direction
    
            # Check if all other atoms lie on the line within the threshold
            linear = True
            for i in range(2, num_atoms):
                vec = atom_positions[i] - point_on_line
                vec_norm = np.linalg.norm(vec)
                if vec_norm == 0:
                    # Atom coincides with the point on the line
                    continue  # No deviation, move to the next atom
                # Compute the sine of the angle between vec and direction_unit
                cross_prod = np.cross(direction_unit, vec)
                sin_theta = np.linalg.norm(cross_prod) / vec_norm
                if sin_theta > linearity_threshold:
                    linear = False
                    break
    
            if linear:
                self._dimension = 2  # Linear molecule
            else:
                self._dimension = 3  # 3D molecule
        return
    
    
    def __repr__(self):
        return (f"Molecule: {self.name}\n {self.atoms}")
        #return (f"Molecule({self.name}, Electronic Energy: {self.energy}, ")

    @classmethod
    def from_lists(cls, element_symbols, xyz_coords, molecule_name="Unnamed Molecule"):
        """
        Create a Molecule object from lists of element symbols and XYZ coordinates.
    
        :param element_symbols: List of element symbols (e.g., ['H', 'O', 'H']).
        :param xyz_coords: List of XYZ coordinates (e.g., [[0,0,0], [0.76,0.58,0], [-0.76,0.58,0]]).
        :param molecule_name: Name of the molecule (default is "Unnamed Molecule").
        :return: Molecule object.
        """
        molecule = cls(molecule_name)
    
        if len(element_symbols) != len(xyz_coords):
            raise ValueError("Number of element symbols must match the number of XYZ coordinate sets.")
    
        # Iterate over the symbols and coordinates, creating atoms and adding them to the molecule
        for symbol, coords in zip(element_symbols, xyz_coords):
            atom = Atom(symbol, *coords)  # Unpacking the xyz coordinates
            molecule.add_atom(atom)
    
        return molecule


    @staticmethod
    def extract_energy(comment_line):
        """
        Extracts the energy from the comment line using multiple regex patterns.

        Parameters:
        comment_line (str): The comment line from which to extract the energy.

        Returns:
        float or None: The extracted energy, or None if no energy is found.
        """
        # List of regex patterns to match different energy formats
        energy_patterns = [
            r"Energy\s*=\s*(-?\d+\.\d+)",   # Matches 'Energy = -1234.56789'
            r"dE\s*=\s*(-?\d+\.\d+)",       # Matches 'dE = -1234.56789'
            r"\s*(-?\d+\.\d+)"              # Matches floating-point number with optional spaces
        ]

        for pattern in energy_patterns:
            match = re.search(pattern, comment_line)
            if match:
                return float(match.group(1))

        return None

#    @classmethod
#    def from_ensemble_file(cls, filename):
#        """
#        Class method to parse an XYZ file with multiple molecular structures and energies.
#
#        Parameters:
#        filename (str): Path to the XYZ file.
#
#        Returns:
#        list: A list of Molecule objects with their respective atoms and energy.
#        """
#        with open(filename, 'r') as file:
#            lines = file.readlines()
#
#        molecules = []
#        i = 0
#        while i < len(lines):
#            natoms = int(lines[i].strip())  # Number of atoms
#            comment_line = lines[i + 1].strip()
#            energy = cls.extract_energy(comment_line)  # Extract energy from the comment line
#
#            molecule = cls(name=f"Molecule_{len(molecules)}", energy=energy)
#
#            # Parse atoms and their coordinates
#            for j in range(natoms):
#                parts = lines[i + 2 + j].split()
#                symbol = parts[0]
#                coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
#                atom = Atom(symbol, *coords)
#                molecule.add_atom(atom)
#
#            molecules.append(molecule)
#            i += 2 + natoms  # Move to the next molecule block
#
#        return molecules

    @classmethod
    def from_xyz_file(cls, filename, molecule_name="Unnamed Molecule"):
        """
        Parse an XYZ file and create a Molecule object.
        
        :param filename: Path to the XYZ file.
        :param molecule_name: Name of the molecule (default is "Unnamed Molecule").
        :return: Molecule object.
        """
    
        with open(filename, 'r') as file:
            # Read the number of atoms (first line) and optional comment (second line)
            num_atoms = int(file.readline().strip())
            comment_line = file.readline()  # Skip the comment line
            energy = cls.extract_energy(comment_line)  # Extract energy from the comment line
            molecule = cls(molecule_name, energy=energy)
            
            # Read atom data
            for line in file:
                symbol, x, y, z = line.split()
                atom = Atom(symbol, float(x), float(y), float(z))
                molecule.add_atom(atom)

        return molecule
    
#
#    @classmethod
#    def from_dataframe(cls, df, element_column_str='Elements', xyz_column_str='xyz Coordinates'):
#        """
#        Create a list of Molecule objects from a DataFrame.
#    
#        :param df: DataFrame with 'Elements' and 'xyz Coordinates' columns.
#        :return: List of Molecule objects.
#        """
#        molecules = []
#    
#        for index, row in df.iterrows():
#            elements = [elem.upper() for elem in row[element_column_str]]
#            coordinates = row[xyz_column_str]
#            molecule_name = f"{index}"  # You can customize the naming
#            molecule = cls.from_lists(elements, coordinates, molecule_name=molecule_name)
#            molecules.append(molecule)
#    
#        return molecules



# Example usage
if __name__ == "__main__":
    # Element symbols and XYZ coordinates for a water molecule
    element_symbols = ['O', 'H', 'H']
    xyz_coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]


    # Create the water molecule from lists
    water_molecule = Molecule.from_lists(element_symbols, xyz_coords, molecule_name="Water")
    #water_molecule = Molecule.from_xyz_file("water.xyz", molecule_name="Water")

    # Display molecule details
    print(water_molecule)

    # Add an additional atom to the molecule (e.g., adding an oxygen atom)
    new_oxygen = Atom("O", 1.0, 1.0, 0.0)
    water_molecule.add_atom(new_oxygen)
    print("After adding another oxygen atom:")
    print(water_molecule)

    # Use get_atom to fetch an atom by index
    first_atom = water_molecule.get_atom(0)
    print(f"First atom: {first_atom.symbol} at {first_atom.position}")

    # Use get_atoms_by_symbol to fetch all hydrogen atoms
    hydrogen_atoms = water_molecule.get_atoms_by_symbol('H')
    for idx, hydrogen_atom in enumerate(hydrogen_atoms, 1):
        print(f"Hydrogen atom {idx}: {hydrogen_atom.symbol} at {hydrogen_atom.position}")

    # Remove an atom (e.g., removing the newly added oxygen)
    water_molecule.remove_atom(new_oxygen)
    print("After removing the added oxygen atom:")
    print(water_molecule)

    # Calculate center of mass
    center_of_mass = water_molecule.center_of_mass()
    print(f"Center of Mass: {center_of_mass}")


    # Create a molecule
    mol = Molecule(name="Example Molecule")

    # Add atoms and see the dimension update
    mol.add_atom(Atom("H", 0.0, 0.0, 0.0))
    print(f"Dimension after adding 1 atom: {mol.dimension}")  # Should be 1

    mol.add_atom(Atom("H", 1.0, 0.0, 0.0))
    print(f"Dimension after adding 2 atoms: {mol.dimension}")  # Should be 2 (linear)

    mol.add_atom(Atom("O", 2.0, 0.000000000000001, 0.0))
    print(f"{mol}")  # Should be 3 (3D)
    print(f"{len(mol.atoms)}")  # Should be 3 (3D)
    print(f"Dimension after adding 3 atoms: {mol.dimension}")  # Should be 3 (3D)

    # Remove an atom and check the dimension
    mol.remove_atom(mol.atoms[2])
    print(f"Dimension after removing 1 atom: {mol.dimension}")  # Should go back to 2 (linear)




