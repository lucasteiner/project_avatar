import numpy as np
#from src.molecule import Molecule  # Assuming Molecule class is saved in molecule.py
import scipy.constants as const
from src.config import config

class Mechanics():
    """
    A class to calculate thermodynamic properties of a molecule.
    Inherits from the Molecule class.
    """

    def __init__(self, frequencies, symbols, moments_of_inertia, molecular_mass, symmetry_number, volume_correction=None, qRRHO_bool=None, electronic_energy=None, solvation_enthalpy=None, temperature=None):

        self.symbols = symbols
        if temperature:
            self.temperature = temperature
        elif config['TEMPERATURE']:
            self.temperature = config['TEMPERATURE']
        else:
            raise KeyError('Please append "TEMPERATURE : 298.15" in the file config/.config.yaml or set argument temperature')
        self.pressure = config['PRESSURE']
        self.frequency_scaling = config['FREQUENCY_SCALING']

        self.natoms = len(symbols)
        self.symmetry_number = symmetry_number
        self.moments_of_inertia = moments_of_inertia
        self.molecular_mass = molecular_mass
        self.moles = 1

        self.electronic_energy = electronic_energy
        self.volume_correction = volume_correction
        self.qRRHO_bool = qRRHO_bool
        self.solvation_enthalpy = solvation_enthalpy
        if self.volume_correction and self.solvation_enthalpy:
            raise ValueError("Don't use volume correction (volume_correction=None), if solvation enthalpy is calculated separately (e.g. with COSMO-RS)")

        # Correcting translational partition function for liquid phase concentration assuming 1 L instead of ideal gas volume of 22.4 L
        if self.volume_correction:
            self.volume = config['VOLUME'] * 1e-3  # Convert default of 1 liters to 0.001 cubic meters (m³)
        else:
            self.volume = self.moles * const.R * self.temperature / self.pressure  # V = kT/p

        if frequencies is None and not self.natoms == 1:
            raise ValueError("Vibrational frequencies are required")

        # Set vibrational thermodynamic functions
        if self.natoms == 1:
            self.frequencies = None
            self.zpe = 0
            self.q_vib = 1
            self.U_vib = 0
        else:
            self.frequencies = frequencies[np.where(frequencies > 0)]
            self.zpe = 0.5 * np.sum(const.h * self.frequencies * const.c * 100) * const.N_A / 1000
            theta_vib = (const.h * self.frequencies * const.c * 100) / const.k
            self.U_vib = np.sum((theta_vib / (np.exp(theta_vib / self.temperature) - 1)) * const.k) * const.N_A / 1000


        # Set rotational thermodynamic functions
        if self.natoms == 1:
            self.q_rot = 1
            self.U_rot = 0
        elif self.is_linear():
            #self.q_rot = const.R / const.kilo * self.temperature / self.moments_of_inertia / self.symmetry_number
            self.U_rot = self.moles * const.R * self.temperature / const.kilo
        else:
            self.U_rot = 1.5*self.moles * const.R * self.temperature / const.kilo

        # Translational thermodynamic functions are independent of dimension of molecule 
        self.U_trans = 1.5 * self.moles * const.R * self.temperature / const.kilo

        # Calculate partition functions
        self.q_elec = 1
        self.q_rot = self.rotational_partition_function()
        self.q_vib = self.vibrational_partition_function()
        self.q_trans = self.translational_partition_function()
        
        # Set electronic thermodynamic functions
        self.U_elec = 0

        self.q = self.q_trans * self.q_vib * self.q_rot * self.q_elec
        self.U = self.zpe + self.U_trans + self.U_vib + self.U_rot + self.U_elec
        self.H = self.U + self.moles * const.R * self.temperature / const.kilo
        self.S = (self.U - self.zpe) / self.temperature + const.R / const.kilo * np.log(self.q) + const.R / const.kilo
        self.G = self.zpe - self.temperature * np.log(self.q) * const.R / const.kilo

        # qRRHO correction
        self.qrrho_cutoff = config['qRRHO_CUTOFF']
        if qRRHO_bool is None:
            self.qRRHO_config = config['qRRHO']
        elif qRRHO_bool:
            self.qRRHO_config = True
        else:
            self.qRRHO_config = False

        if self.qRRHO_config and self.frequencies is not None:
            self.qRRHO = self.qRRHO_correcture(self.frequencies)
        else:
            self.qRRHO = 0

        self.G_total = self.total_gibbs_free_energy()


        #print('U:', self.U, self.U_trans, self.U_vib, self.U_rot, self.U_elec)
        #print('q:', self.q, self.q_trans, self.q_vib, self.q_rot, self.q_elec)
        #print('S:', self.S)
        #print('R:', const.R)
        #print('S dirty:', (self.H - self.G) / self.temperature)
        #print('kilo:', const.kilo)
        #print('temperature:', self.temperature)
        #print('MOI:', self.moments_of_inertia)
    def total_gibbs_free_energy(self):
        tmp = self.G
        if self.electronic_energy:
            tmp += self.electronic_energy
        if self.qRRHO:
            tmp -= self.qRRHO
        if self.solvation_enthalpy:
            tmp += self.solvation_enthalpy
        return tmp


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

    def translational_partition_function(self, temperature=None):
        """
        Calculate the translational partition function.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: The translational partition function.
        """
        if not temperature:
            temperature = self.temperature
        mass_kg = self.molecular_mass / const.kilo / const.N_A  # Convert g/mol to kg per molecule

        q_trans = ((2 * np.pi * mass_kg * const.k * temperature) ** 1.5 * self.volume) / (const.h ** 3 * const.N_A * self.moles)

        #return (mass_kg*temperature*2*np.pi*const.k/const.h/const.h)**1.5 * volume /n_part/N_A
        return q_trans

    def rotational_partition_function(self, temperature=None):
        """
        Calculate the rotational partition function.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: The rotational partition function.
        """
        if not temperature:
            temperature = self.temperature
        sigma = self.symmetry_number
        #print(sigma)

        if self.natoms == 1:
            # Monoatomic gas has no rotational degrees of freedom
            return 1.0

        moments = self.moments_of_inertia
        #print(moments)
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

    def vibrational_partition_function(self, temperature=None):
        """
        Calculate the vibrational partition function.

        Parameters:
        temperature (float): Temperature in Kelvin.

        Returns:
        float: The vibrational partition function.
        """
        if not temperature:
            temperature = self.temperature
        if self.frequencies is None:
            return 1.0  # No vibrational modes

        theta_vib = (const.h * self.frequencies * const.c * 100) / const.k  # Vibrational temperatures
        q_vib = np.prod(1 / (1 - np.exp(-theta_vib / temperature)))
        return q_vib
        # q_vib is tested :)

    #def internal_energy_rotational(self, temperature=None):
    #    """
    #    Calculate the rotational contribution to internal energy.

    #    Parameters:
    #    temperature (float): Temperature in Kelvin.

    #    Returns:
    #    float: Rotational internal energy in Joules per molecule.
    #    """

    #    if not temperature:
    #        temperature = self.temperature
    #    if self.natoms == 1:
    #        return 0.0  # Atoms have no rotational energy

    #    if self.is_linear():
    #        return const.k * temperature
    #    else:
    #        return const.k * temperature * 1.5

    #def internal_energy_vibrational(self, temperature=None):
    #    """
    #    Calculate the vibrational contribution to internal energy.

    #    Parameters:
    #    temperature (float): Temperature in Kelvin.

    #    Returns:
    #    float: Vibrational internal energy in Joules per molecule.
    #    """
    #    if not temperature:
    #        temperature = self.temperature
    #    if self.frequencies is None:
    #        return 0.0

    #    theta_vib = (const.h * self.frequencies * const.c * 100) / const.k  # Vibrational temperatures
    #    U_vib = np.sum((theta_vib / (np.exp(theta_vib / temperature) - 1)) * const.k)
    #    return U_vib

    #def entropy_translational(self, temperature=None):
    #    """
    #    Calculate the translational contribution to entropy.

    #    Parameters:
    #    temperature (float): Temperature in Kelvin.

    #    Returns:
    #    float: Translational entropy in J/K per molecule.
    #    """
    #    if not temperature:
    #        temperature = self.temperature
    #    q_trans = self.translational_partition_function(temperature)
    #    # print('qtrans:', q_trans)
    #    S_trans = const.k * (np.log(q_trans) + 1.5 + 1)
    #    return S_trans

    #def entropy_rotational(self, temperature=None):
    #    """
    #    Calculate the rotational contribution to entropy.

    #    Parameters:
    #    temperature (float): Temperature in Kelvin.

    #    Returns:
    #    float: Rotational entropy in J/K per molecule.
    #    """
    #    if not temperature:
    #        temperature = self.temperature
    #    if self.natoms == 1:
    #        return 0.0  # Atoms have no rotational entropy

    #    q_rot = self.rotational_partition_function(temperature)
    #    if self.is_linear():
    #        S_rot = const.k * (np.log(q_rot) + 1)
    #    else:
    #        S_rot = const.k * (np.log(q_rot) + 1.5)
    #        print(q_rot)
    #    return S_rot

    #def entropy_vibrational(self, temperature=None):
    #    """
    #    Calculate the vibrational contribution to entropy.

    #    Parameters:
    #    temperature (float): Temperature in Kelvin.

    #    Returns:
    #    float: Vibrational entropy in J/K per molecule.
    #    """
    #    if not temperature:
    #        temperature = self.temperature
    #    if self.frequencies is None:
    #        return 0.0

    #    theta_vib = (const.h * self.frequencies * const.c * 100) / const.k  # Vibrational temperatures
    #    S_vib = np.sum((theta_vib / (temperature * (np.exp(theta_vib / temperature) - 1)) - np.log(1 - np.exp(-theta_vib / temperature))) * const.k)
    #    return S_vib

    def qRRHO_correcture (self, freq_cm, temperature=None):
        """
        takes np array with positive vibrational frequencies in wavenumbers
        Return value corrects the Gibbs free enthalpy.
        """
        if not temperature:
            temperature = self.temperature
        Bav = 1e-44 #kg*m^2
        freq_s = freq_cm*100.0*const.c #1/s
        xx = freq_s*const.h/const.k/temperature #no unit
        #print('xx=',xx)
        Sv = xx * (1.0 / (np.exp(xx)-1.0))  -  np.log(1.0 - np.exp(-xx)) #no unit
        mue = const.h/(8.0 * np.pi**2.0 * freq_s) #J*s^2 = kgm^2
        Sr = (1 + np.log(8.0 *np.pi*np.pi*np.pi *mue*Bav / (mue + Bav) *const.k*temperature /const.h/const.h) ) / 2 #no unit
        w_damp = 1.0 / (1.0 + (1e2/freq_cm)**4) #m^4
        S_final = w_damp*const.R*Sv + ( (1.0-w_damp) *const.R*(1 + np.log(8.0*np.pi*np.pi*np.pi*mue*Bav /(mue + Bav) *const.k*temperature/const.h/const.h) ) /2) #m^4*J/mol/K
        return np.sum((S_final - const.R*Sv )/const.kilo)*temperature #m^4*kJ/mol

    def is_linear(self, tolerance=1e-3):
        """
        Determine if the molecule is linear within a specified tolerance.

        Parameters:
        tolerance (float): The threshold below which a moment of inertia is considered zero.

        Returns:
        bool: True if the molecule is linear, False otherwise.
        """
        return np.sum(self.moments_of_inertia < tolerance) == 1

if __name__ == '__main__':

    mechanics = Mechanics(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.1, 3565.12, 4000.2]), 
            np.array(['H', 'O', 'H']), 
            np.array([20.3, 30.4, 10.2]),
            18.0,
            1,
            )
