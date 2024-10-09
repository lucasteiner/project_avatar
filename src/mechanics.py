import numpy as np
#from src.molecule import Molecule  # Assuming Molecule class is saved in molecule.py
import scipy.constants as const
from src.config import config

class Mechanics():
    """
    A class to calculate thermodynamic properties of a molecule.
    Inherits from the Molecule class.
    """

    def __init__(self, frequencies, symbols, moments_of_inertia, molecular_mass, symmetry_number):

        self.volume = config['VOLUME'] * 1e-3  # Convert liters to cubic meters (m³)
        self.temperature = config['TEMPERATURE']
        self.pressure = config['PRESSURE']
        self.frequency_scaling = config['FREQUENCY_SCALING'] # to be implemented, config?
        self.qrrho_cutoff = config['qRRHO_CUTOFF']
        self.gas_phase = config['GAS_PHASE']

        #self.parent_molecule = parent_molecule
        #self.natoms = len(parent_molecule.symbols)
        #self.frequencies = parent_molecule.frequencies
        #self.is_linear = parent_molecule.is_linear

        self.natoms = len(symbols)
        self.symmetry_number = symmetry_number
        self.moments_of_inertia = moments_of_inertia
        self.molecular_mass = molecular_mass
        self.frequencies = frequencies[np.where(frequencies > 0)]

    def zero_point_energy(self):
        """
        Calculate the zero-point energy (ZPE) of the molecule.

        Returns:
        float: The zero-point energy in kJ/mol.
        """
        if self.frequencies is None:
            raise ValueError("Vibrational frequencies are required to calculate zero-point energy.")

        # Convert frequencies from wavenumbers (cm^-1) to Joules
        freq_in_hz = self.frequencies * const.c * 100  # Convert cm^-1 to Hz
        zpe = 0.5 * np.sum(const.h * freq_in_hz)
        zpe_per_mol = zpe * const.N_A / 1000  # Convert to kJ/mol
        return zpe_per_mol

    def calculate_partition_functions(self, temperature):
        """
        Calculate the translational, rotational, vibrational, and electronic partition functions.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        dict: A dictionary containing partition functions.
        """
        q_trans = self.translational_partition_function(temperature)
        q_rot = self.rotational_partition_function(temperature)
        q_vib = self.vibrational_partition_function(temperature)
        q_elec = self.electronic_partition_function(temperature)
        return {
            'translational': q_trans,
            'rotational': q_rot,
            'vibrational': q_vib,
            'electronic': q_elec
        }

    def translational_partition_function(self, temperature):
        """
        Calculate the translational partition function.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: The translational partition function.
        """
        mass_kg = self.molecular_mass * 1e-3 / const.N_A  # Convert g/mol to kg per molecule

        if self.gas_phase:
            # Ideal gas volume per molecule at standard pressure (1 atm)
            volume = const.k * temperature / self.pressure  # V = kT/p
        else:
            volume = self.volume  # Use the set volume in m³

        q_trans = ((2 * np.pi * mass_kg * const.k * temperature) ** 1.5 * volume) / (const.h ** 3)
        return q_trans

    def rotational_partition_function(self, temperature):
        """
        Calculate the rotational partition function.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: The rotational partition function.
        """
        sigma = self.symmetry_number
        print(sigma)

        if self.natoms == 1:
            # Monoatomic gas has no rotational degrees of freedom
            return 1.0

        moments = self.moments_of_inertia
        print(moments)
        moments = moments[moments > 1e-10]  # Exclude zero moments

        if self.is_linear():
            # Linear molecule: one non-zero moment of inertia
            I = moments[0] * const.physical_constants['atomic mass constant'][0] * 1e-20  # Convert amu·Å² to kg·m²
            theta_rot = const.hbar ** 2 / (2 * I * const.k)
            q_rot = temperature / (sigma * theta_rot)
        else:
            # Non-linear molecule: three moments of inertia
            I1, I2, I3 = moments * const.physical_constants['atomic mass constant'][0] * 1e-20  # kg·m²
            q_rot = np.pi ** 0.5 / sigma * (8* np.pi**2 * const.k * temperature / const.h / const.h)**(3/2) * (I1*I2*I3)**(1/2)

        return q_rot

    def vibrational_partition_function(self, temperature):
        """
        Calculate the vibrational partition function.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: The vibrational partition function.
        """
        if self.frequencies is None:
            return 1.0  # No vibrational modes

        theta_vib = (const.h * self.frequencies * const.c * 100) / const.k  # Vibrational temperatures
        q_vib = np.prod(1 / (1 - np.exp(-theta_vib / temperature)))
        return q_vib
        # q_vib is tested :)

    def electronic_partition_function(self, temperature):
        """
        Calculate the electronic partition function.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: The electronic partition function.
        """
        # For simplicity, assume ground state degeneracy is 1
        return 1.0

    def thermodynamic_properties(self, temperature):
        """
        Calculate thermodynamic properties at a given temperature.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        dict: A dictionary containing Gibbs free energy, enthalpy, entropy, internal energy, and zero-point energy.
        """
        q = self.calculate_partition_functions(temperature)
        beta = 1 / (const.k * temperature)

        # Internal Energy U
        U_trans = 1.5 * const.k * temperature
        U_rot = self.internal_energy_rotational(temperature)
        U_vib = self.internal_energy_vibrational(temperature)
        U_elec = 0  # Assuming ground state
        U = U_trans + U_rot + U_vib + U_elec

        # Enthalpy H = U + pV (for ideal gas, pV = nRT, and n=1 here)
        #H = U + const.k * temperature  # H = U + RT per molecule
        H = U + const.k * temperature  # H = U + RT per molecule

        # Entropy S
        S_trans = self.entropy_translational(temperature)
        S_rot = self.entropy_rotational(temperature)
        S_vib = self.entropy_vibrational(temperature)
        S_elec = 0  # Assuming ground state
        S = S_trans + S_rot + S_vib + S_elec

        # Gibbs Free Energy G = H - T*S
        G = H - temperature * S
        # chem.pot.=ZPE-RT*ln(qtrans*qrot*qvib)
        #G = - const.k *temperature*np.log(q['translational'] * q['rotational'] * q['vibrational'])

        # Convert energies to kJ/mol
        factor = const.N_A / 1000  # To convert J per molecule to kJ/mol
        U *= factor
        H *= factor
        G *= factor
        S *= factor  # Entropy in kJ/mol·K

        return {
            'Gibbs free energy': G,
            'Enthalpy': H,
            'Entropy': S,
            'Internal energy': U,
            'Zero-point energy': self.zero_point_energy()
        }

    def internal_energy_rotational(self, temperature):
        """
        Calculate the rotational contribution to internal energy.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: Rotational internal energy in Joules per molecule.
        """
        if self.natoms == 1:
            return 0.0  # Atoms have no rotational energy

        if self.is_linear():
            return const.k * temperature
        else:
            return const.k * temperature * 1.5

    def internal_energy_vibrational(self, temperature):
        """
        Calculate the vibrational contribution to internal energy.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: Vibrational internal energy in Joules per molecule.
        """
        if self.frequencies is None:
            return 0.0

        theta_vib = (const.h * self.frequencies * const.c * 100) / const.k  # Vibrational temperatures
        U_vib = np.sum((theta_vib / (np.exp(theta_vib / temperature) - 1)) * const.k)
        return U_vib

    def entropy_translational(self, temperature):
        """
        Calculate the translational contribution to entropy.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: Translational entropy in J/K per molecule.
        """
        mass_kg = self.molecular_mass * 1e-3 / const.N_A  # Convert g/mol to kg per molecule

        if self.gas_phase:
            # Ideal gas volume per molecule at standard pressure (1 atm)
            volume = const.k * temperature / self.pressure  # V = kT/p
        else:
            volume = self.volume  # Use the set volume in m³

        q_trans = self.translational_partition_function(temperature)
        print('qtrans:', q_trans)
        S_trans = const.k * (np.log(q_trans) + 1.5 + 1)
        return S_trans

    def entropy_rotational(self, temperature):
        """
        Calculate the rotational contribution to entropy.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: Rotational entropy in J/K per molecule.
        """
        if self.natoms == 1:
            return 0.0  # Atoms have no rotational entropy

        q_rot = self.rotational_partition_function(temperature)
        if self.is_linear():
            S_rot = const.k * (np.log(q_rot) + 1)
        else:
            S_rot = const.k * (np.log(q_rot) + 1.5)
            print(q_rot)
        return S_rot

    def entropy_vibrational(self, temperature):
        """
        Calculate the vibrational contribution to entropy.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: Vibrational entropy in J/K per molecule.
        """
        if self.frequencies is None:
            return 0.0

        theta_vib = (const.h * self.frequencies * const.c * 100) / const.k  # Vibrational temperatures
        S_vib = np.sum((theta_vib / (temperature * (np.exp(theta_vib / temperature) - 1)) - np.log(1 - np.exp(-theta_vib / temperature))) * const.k)
        return S_vib

    def is_linear(self, tolerance=1e-3):
        """
        Determine if the molecule is linear within a specified tolerance.

        Parameters:
        tolerance (float): The threshold below which a moment of inertia is considered zero.

        Returns:
        bool: True if the molecule is linear, False otherwise.
        """
        moments = self.moments_of_inertia
        # Sort the moments to ensure consistent order
        moments = np.sort(moments)
        # For a linear molecule, two moments should be approximately zero
        zero_moments = moments < tolerance
        if np.sum(zero_moments) >= 1:
            return True
        else:
            return False

if __name__ == '__main__':

    mechanics = Mechanics(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.1, 3565.12, 4000.2]), 
            np.array(['H', 'O', 'H']), 
            np.array([20.3, 30.4, 10.2]),
            18.0,
            1,
            )
    mechanics.thermodynamic_properties(298.15)
