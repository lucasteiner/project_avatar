from molecule import Molecule
from molecule_utils import compute_rmsd, compare_molecules
import numpy as np
import pandas as pd

class Ensemble:
    def __init__(self):
        self.molecules = []

    def add_molecule(self, molecule):
        self.molecules.append(molecule)

    def filter_molecules(self, condition):
        # Returns a new Ensemble with molecules that satisfy the condition
        filtered = Ensemble()
        filtered.molecules = [mol for mol in self.molecules if condition(mol)]
        return filtered

    def calculate_ensemble_average(self, property_name):
        values = [mol.properties.get(property_name) for mol in self.molecules]
        if None in values:
            raise AttributeError(f"One or more molecules do not have property '{property_name}'")
        return sum(values) / len(values) if values else 0

    @classmethod
    def from_list(cls, molecule_list):
        ensemble = cls()
        ensemble.molecules = molecule_list
        return ensemble

    def add_molecule(self, molecule):
        self.molecules.append(molecule)

    def remove_duplicates(self, energy_threshold=1e-5, rmsd_threshold=1e-3, rotational_constant_threshold=1e-5):
        unique_molecules = []
        for mol in self.molecules:
            is_duplicate = False
            for unique_mol in unique_molecules:
                if compare_molecules(
                    mol, unique_mol,
                    energy_threshold=energy_threshold,
                    rmsd_threshold=rmsd_threshold,
                    rotational_constant_threshold=rotational_constant_threshold
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_molecules.append(mol)
        self.molecules = unique_molecules

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
def boltzmann_average_and_entropy(gibbs_energies, temperature=298):
    """
    Calculate the Boltzmann average of Gibbs free energies, conformational entropy,
    and entropy-corrected average Gibbs free energy.

    Parameters:
    gibbs_energies (pd.Series or list of float): Gibbs free energies of the conformers (in kJ/mol).
    temperature (float): Temperature in Kelvin (default is 298 K).

    Returns:
    float: Boltzmann average of Gibbs free energies (in kJ/mol).
    float: Conformational entropy (in kJ/mol-K).
    float: Entropy-corrected average Gibbs free energy (in kJ/mol).
    """
    R = 8.3145  # Gas constant in J/(mol*K)
    R_kJ = R / 1000  # Convert gas constant to kJ/(mol*K)

    # Ensure gibbs_energies is a numpy array for calculations
    gibbs_energies = np.array(gibbs_energies)

    # Convert Gibbs free energies to dimensionless form using kT (using kJ/mol units)
    dimensionless_energies = (gibbs_energies - gibbs_energies.min()) / (R_kJ * temperature)

    # Calculate Boltzmann factors
    boltzmann_factors = np.exp(-dimensionless_energies)

    # Calculate partition function Z
    partition_function = np.sum(boltzmann_factors)

    # Calculate probabilities (Boltzmann distribution)
    probabilities = boltzmann_factors / partition_function

    # Calculate Boltzmann average of Gibbs free energies
    boltzmann_average_gibbs = np.sum(probabilities * gibbs_energies)

    # Calculate conformational entropy
    conformational_entropy = -R_kJ * np.sum(probabilities * np.log(probabilities))

    # Calculate entropy-corrected average Gibbs free energy
    entropy_corrected_average = boltzmann_average_gibbs - temperature * conformational_entropy

    return boltzmann_average_gibbs, conformational_entropy, entropy_corrected_average

# Example usage
# Creating a pandas DataFrame with Gibbs free energies
#data = {'Gibbs_Free_Energy_kJ_per_mol': [50.0, 51.0, 49.5]}  # Energies in kJ/mol
#df = pd.DataFrame(data)

# Using the function with a DataFrame column
#gibbs_energies_column = df['Gibbs_Free_Energy_kJ_per_mol']
#boltzmann_average, entropy, entropy_corrected = boltzmann_average_and_entropy(gibbs_energies_column)
#print(f"Boltzmann Average Gibbs Free Energy: {boltzmann_average} kJ/mol")
#print(f"Conformational Entropy: {entropy} kJ/mol-K")
#print(f"Entropy-Corrected Average Gibbs Free Energy: {entropy_corrected} kJ/mol")
