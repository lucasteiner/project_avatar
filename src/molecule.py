import numpy as np
import re
import inspect
import copy
from dataclasses import fields
from src.reorder import ReorderMixin
from src.bonding import Bonding
from src.mechanics import Mechanics
from src.config import atomic_masses
from src.config import covalent_radii


class Molecule(ReorderMixin):

    def __init__(self, 
                 symbols, 
                 coordinates, 
                 energy=None, 
                 frequencies=None, 
                 point_group='C1', 
                 symmetry_number=1, 
                 electronic_energy=None, 
                 thermal_corrections=None, 
                 solvation_enthalpy=None, 
                 dipole=None, 
                 volume_correction=None,
                 qRRHO_bool=None,
                 temperature=None):
        self._init_geometry(symbols, coordinates)
        self._init_properties(energy, dipole)
        self._init_symmetry(point_group, symmetry_number)
        self._init_energy_data(electronic_energy, thermal_corrections, solvation_enthalpy)
        self._init_mechanics_if_applicable(frequencies, volume_correction, qRRHO_bool, temperature)

    def _init_geometry(self, symbols, coordinates):
        self.natoms = len(symbols)
        self.symbols = np.array([symbol.capitalize() for symbol in symbols])
        self.coordinates = np.asarray(coordinates, dtype=float)

        if self.coordinates.shape != (self.natoms, 3):
            raise ValueError("Coordinates must have shape (N_atoms, 3).")

        if len(self.symbols) != len(self.coordinates):
            raise ValueError("The number of symbols and coordinate sets must be the same.")

        self.bonding = Bonding(self.symbols, self.coordinates)

    def _init_properties(self, energy, dipole):
        self.energy = energy
        self.dipole = dipole

    def _init_symmetry(self, point_group, symmetry_number):
        self.point_group = point_group
        self.symmetry_number = symmetry_number

    def _init_energy_data(self, electronic_energy, thermal_corrections, solvation_enthalpy):
        self.electronic_energy = electronic_energy
        self.thermal_corrections = thermal_corrections
        self.solvation_enthalpy = solvation_enthalpy

    def _init_mechanics_if_applicable(self, frequencies, volume_correction, qRRHO_bool, temperature):
        self.temperature = temperature
        self.frequencies = np.array(frequencies) if frequencies is not None else None
        self.volume_correction = volume_correction
        self.qRRHO_bool = qRRHO_bool

        # Frequencies are necessary to calculate useful statistical mechanics including vibrations,
        # but if natoms == 1, there are no vibrational frequencies
        if self.frequencies is not None or self.natoms == 1:
            self.mechanics = Mechanics(
                self.frequencies,
                self.symbols,
                self.moments_of_inertia(),
                self.molecular_mass(),
                self.symmetry_number,
                volume_correction=self.volume_correction,
                qRRHO_bool=self.qRRHO_bool,
                temperature=self.temperature
            )

    def molecular_mass(self):
        """
        Calculate the molecular mass in g/mol.

        Returns:
        float: Molecular mass.
        """
        masses = np.array([atomic_masses[symbol] for symbol in self.symbols])
        return masses.sum()

    def center_of_mass(self):
        """
        Calculate and return the center of mass of the molecule.
        """
        masses = np.array([atomic_masses[symbol] for symbol in self.symbols])
        total_mass = masses.sum()
        com = np.dot(masses, self.coordinates) / total_mass
        return com

    def moments_of_inertia(self):
        """
        Calculate and return the principal moments of inertia of the molecule in amu*A^2.
        """
        # Dirty units:
        # atomic_masses : atomic mass units (amu)
        # coordinates : A
        # moi : amu*A^2? this should be changed, but take care of reorder mixin which has certain thresholds 
        # and the mechanics module, which converts them to kg*m^2 by itself
        masses = np.array([atomic_masses[symbol] for symbol in self.symbols])
        coords = self.coordinates - self.center_of_mass()  # Centering around the center of mass
    
        # Initialize the inertia tensor as a 3x3 matrix
        inertia_tensor = np.zeros((3, 3))
    
        for mass, (x, y, z) in zip(masses, coords):
            inertia_tensor[0, 0] += mass * (y**2 + z**2)
            inertia_tensor[1, 1] += mass * (x**2 + z**2)
            inertia_tensor[2, 2] += mass * (x**2 + y**2)
            inertia_tensor[0, 1] -= mass * x * y
            inertia_tensor[0, 2] -= mass * x * z
            inertia_tensor[1, 2] -= mass * y * z
    
        # Complete the symmetric tensor by copying the values
        inertia_tensor[1, 0], inertia_tensor[2, 0], inertia_tensor[2, 1] = inertia_tensor[0, 1], inertia_tensor[0, 2], inertia_tensor[1, 2]
    
        # Calculate eigenvalues (principal moments of inertia)
        moments, _ = np.linalg.eigh(inertia_tensor)
    
        return moments

    def recenter(self):
        """
        Returns centered molecules coordinates so that its center of mass is at the origin.
        """
        com = self.center_of_mass()
        return self.coordinates - com

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

    @classmethod
    def from_xyz(cls, filename, energy=None, frequencies=None):
        """
        Create a Molecule instance from an XYZ-format file.

        Parameters:
        filename (str): Path to the XYZ file.
        energy (float, optional): Energy of the molecule.
        frequencies (list, optional): Vibrational frequencies.

        Returns:
        Molecule: An instance of the Molecule class.
        """
        symbols = []
        coordinates = []
        with open(filename, 'r') as file:
            lines = file.readlines()
            if len(lines) < 3:
                raise ValueError("The XYZ file is incomplete or corrupted.")
            try:
                num_atoms = int(lines[0].strip())
            except ValueError:
                raise ValueError("The first line of the XYZ file should be the number of atoms.")
            atom_lines = lines[2:2+num_atoms]
            for line in atom_lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    raise ValueError("Each atom line must have at least 4 entries: symbol, x, y, z.")
                symbol = parts[0]
                if symbol not in atomic_masses:
                    raise ValueError(f"Unknown element symbol: {symbol}")
                x, y, z = map(float, parts[1:4])
                symbols.append(symbol)
                coordinates.append([x, y, z])
        return cls(symbols, coordinates, energy, frequencies)

    def to_xyz(self, comment=None, attributes_comment=None):
        """
        Convert the Molecule instance to an XYZ-formatted string.
        Comment section is filled with energy data, if present.

        Parameters:
        - comment (str, optional): Comment or title for the XYZ file, 
            will substitute energetic attributes and attributes_comment.
        - attributes_comment (str, optional): Comment or title for the XYZ file. 
            will be appended to energetic attributes.

        Returns:
        - xyz_str (str): The XYZ-formatted string representing the molecule.
        """
        if comment is None:
            comment = self.format_energetic_attributes(additional_string=attributes_comment)
        num_atoms = len(self.symbols)
        xyz_lines = [f"{num_atoms}", comment]
        for symbol, coord in zip(self.symbols, self.coordinates):
            # Ensure coordinates are formatted to three decimal places
            #line = f"{symbol} \t{coord[0]:.5f} \t{coord[1]:.5f} \t{coord[2]:.5f}"
            line = f"{symbol} {coord[0]:>12.5f} {coord[1]:>12.5f} {coord[2]:>12.5f}"
            xyz_lines.append(line)
        xyz_str = "\n".join(xyz_lines)
        return xyz_str

    def format_energetic_attributes(self, additional_string=""):
        """
        Formats and returns a string of energetic attributes with prefixes if set, and appends an additional string.

        Args:
            additional_string (str): An optional string to append to the formatted attributes.

        Returns:
            str: A string containing "E_el=", "G_thr=", and "G_solv=" with their corresponding values,
            followed by the additional string if provided, or an empty string if none are set.
        """
        result = []

        if self.electronic_energy is not None:
            result.append(f"E_el={self.electronic_energy:.2f}")

        if self.thermal_corrections is not None:
            result.append(f"G_thr={self.thermal_corrections:.2f}")

        if self.solvation_enthalpy is not None:
            result.append(f"G_solv={self.solvation_enthalpy:.2f}")

        # Join the attributes and append the additional string if provided
        formatted_output = ", ".join(result)
        if additional_string:
            formatted_output += f", {additional_string}"

        return formatted_output

    def g_total(self):
        """
        Sums the energetic attributes, treating None as zero.

        Returns:
            float: The sum of `electronic_energy`, `thermal_corrections`, and `solvation_enthalpy`,
            where missing values (None) are treated as zero.
        """
        if self.thermal_corrections is None:
            raise ValueError('Missing thermal corrections or frequencies')
        return sum(attr or 0 for attr in [self.electronic_energy, self.thermal_corrections, self.solvation_enthalpy])

    def compare_energy_and_moments(self, other, precision=1.0):
        """
        Compare the energy and moments of inertia of two molecules.
        Raises an error if the molecules are not comparable.
    
        Parameters:
        other (Molecule): The other molecule to compare.
        precision (float): The allowable difference for moments of inertia comparison (default is 1.0).
    
        Returns:
        bool: True if both energy and moments of inertia match within the given precision, False otherwise.
        """
        # First check if molecules are comparable
        if not self.is_comparable(other):
            raise ValueError("Molecules are not comparable (different elements or quantities).")
            #return False
    
        # Compare energy
        if not np.isclose(self.energy, other.energy, atol=precision):
            return False
    
        # Compare moments of inertia
        return np.allclose(self.moments_of_inertia(), other.moments_of_inertia(), atol=precision)
        #return self.compare_moments_of_inertia(other, precision)
 
    def is_comparable(self, other):
        """
        Check if two molecules are comparable by having the same number of elements.
        
        Parameters:
        other (Molecule): The other molecule to compare with.
        
        Returns:
        bool: True if the molecules are comparable, False otherwise.
        """
        return sorted(self.symbols) == sorted(other.symbols)
 
    def set_property(self, name, value):
        """
        Add or modify an optional property of the molecule.
        """
        setattr(self, name, value)

    def get_property(self, name):
        """
        Retrieve a property of the molecule.
        """
        return getattr(self, name, None)

    def __repr__(self):
        """
        Return a string representation of the Molecule.
        """
        lines = []
        for symbol, coord in zip(self.symbols, self.coordinates):
            x, y, z = coord
            lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
        return '\n'.join(lines)

    def copy(self, **kwargs):
        """
        Return a deep copy of the object with optional field overrides.
        All fields are copied using deepcopy to avoid shared references.
        """
        allowed = {
            name
            for name, param in inspect.signature(self.__class__.__init__).parameters.items()
            if name != 'self'
        }
        invalid = set(kwargs) - allowed
        if invalid:
            raise ValueError(f"Invalid fields: {', '.join(invalid)}")
    
        # Deepcopy all current attributes unless explicitly overridden
        args = {
            key: kwargs.get(key, copy.deepcopy(getattr(self, key)))
            for key in allowed
        }
        return self.__class__(**args)

    def with_changes(self, **kwargs):
        """
        Return the object with changes.
        Actually, a new molecule is initialized for this purpose.
        However, objects are reference copies.
        Use "mol.copy()" if you need deep copies.
        """
        allowed = {
            name
            for name, param in inspect.signature(self.__class__.__init__).parameters.items()
            if name != 'self'
        }
        invalid = set(kwargs) - allowed
        if invalid:
            raise ValueError(f"Invalid fields: {', '.join(invalid)}")
 
        args = {key: kwargs.get(key, getattr(self, key)) for key in allowed}
        return self.__class__(**args)

    def get_cavity_volume(self, num_samples=1000000):
        """
        Estimates the cavity volume formed by the van der Waals radii of the atoms 
        using the Monte Carlo integration method.

        Args:
            num_samples (int): Number of random points for Monte Carlo sampling.

        Returns:
            float: Estimated cavity volume in Å³.
        
        The method works as follows:
        1. Each atom is treated as a sphere with its respective van der Waals radius.
        2. A bounding box is computed to define the sampling space.
        3. Random points are generated within the bounding box.
        4. The proportion of points that fall inside any atomic sphere is used to 
           approximate the cavity volume based on the bounding box volume.
        """
        # Extract van der Waals radii for each atom
        radii = np.array([covalent_radii[sym] for sym in self.symbols])

        # Compute the bounding box for sampling
        min_bounds = np.min(self.coordinates - radii[:, np.newaxis], axis=0)
        max_bounds = np.max(self.coordinates + radii[:, np.newaxis], axis=0)
        box_volume = np.prod(max_bounds - min_bounds)

        # Generate random points within the bounding box
        random_points = np.random.uniform(min_bounds, max_bounds, (num_samples, 3))

        # Count the number of points inside any van der Waals sphere
        inside_count = 0
        for i, center in enumerate(self.coordinates):
            distances = np.linalg.norm(random_points - center, axis=1)
            inside_count += np.sum(distances < radii[i])

        # Compute the estimated cavity volume
        cavity_volume = box_volume * (inside_count / num_samples)
        return cavity_volume

    def is_duplicate(self, other):
        """Compares two molecules and returns true if they are duplicates, False otherwise

        Parameters:
            other (molecule): Molecule for comparison

        Returns:
        duplicate (bool): True, if molecules are duplicates
        """
        coord, symbols, _ = self.reorder_after(other)
        reference = Molecule(symbols.copy(), coord.copy(), energy=float(other.energy)) # reorder to get isomers and align coordinates, too
        try:
            tmp = other.compare_molecule(reference)
            duplicate = np.all(tmp)
        except ValueError:
            #print('ValueError, handle RMSD calculation better!')
            duplicate = False
        return duplicate