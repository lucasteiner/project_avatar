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

    def __repr__(self):
        atoms_str = '\n '.join([repr(atom) for atom in self.atoms])
        return f""" Molecule:\t{self.name}
 Dimensions:\t{self.dimension}
 {atoms_str}"""


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



