from scipy.constants import Planck, Boltzmann, Avogadro, gas_constant, pi, c
import numpy as np

class Molecule3DMechanics:
    def __init__(self, mass, rotational_constants, vibrational_frequencies, temperature=298.15, pressure=1.0, volume=None):
        """
        Initialize the Molecule3DMechanics class with basic thermodynamic properties for 3D molecules.
        
        Parameters:
        mass (float): Mass of the molecule in kg.
        rotational_constants (list of float): Rotational constants (A, B, C) in m^-1.
        vibrational_frequencies (list of float): Vibrational frequencies in cm-1.
        temperature (float): Temperature in Kelvin (default 298.15 K).
        pressure (float): Pressure in atmospheres (default 1 atm).
        volume (float): Volume in liters (optional, can be calculated or provided).
        """
        # Constants from scipy
        self.h = Planck  # Planck constant
        self.kb = Boltzmann  # Boltzmann constant
        self.na = Avogadro  # Avogadro's number
        self.r = gas_constant  # Gas constant
        self.c = c  # Gas constant
 
        self.mass = mass
        self.rotational_constants = rotational_constants
        self.vibrational_frequencies = vibrational_frequencies * c * 100 # conversion to Hz
        self.temperature = temperature
        self.pressure = pressure
        self.volume = None
        self.volume = volume or self.calculate_volume()

   
    def calculate_volume(self):
        """Estimate the volume if not provided (using the ideal gas law)."""
        if self.volume is None:
            return (self.r * self.temperature) / (self.pressure * 1e5)  # Convert pressure to Pa
        return self.volume

    def translational_partition_function(self):
        """
        Calculate the translational partition function for a 3D molecule.
        
        Returns:
        float: Translational partition function.
        """
        q_trans = ((2 * pi * self.mass * self.kb * self.temperature) / (self.h ** 2)) ** (3 / 2)
        q_trans *= self.volume
        return q_trans
    
    def rotational_partition_function(self):
        """
        Calculate the rotational partition function for a 3D molecule.
        
        Returns:
        float: Rotational partition function.
        """
        A, B, C = self.rotational_constants
        q_rot = ((pi ** (1/2)) * ((self.temperature ** 3) / (A * B * C))) ** (1/2)
        return q_rot

    def vibrational_partition_function(self):
        """
        Calculate the vibrational partition function for a 3D molecule.
        
        Returns:
        float: Vibrational partition function.
        """
        q_vib = 1.0
        for freq in self.vibrational_frequencies:
            q_vib *= 1 / (1 - np.exp(-self.h * freq / (self.kb * self.temperature)))
        return q_vib
    
    def translational_energy(self):
        """
        Calculate the translational contribution to the internal energy.
        
        Returns:
        float: Translational energy in Joules.
        """
        return (3 / 2) * self.kb * self.temperature

    def rotational_energy(self):
        """
        Calculate the rotational contribution to the internal energy.
        
        Returns:
        float: Rotational energy in Joules.
        """
        return (3 / 2) * self.kb * self.temperature
    
    def vibrational_energy(self):
        """
        Calculate the vibrational contribution to the internal energy.
        
        Returns:
        float: Vibrational energy in Joules.
        """
        u_vib = 0.0
        for freq in self.vibrational_frequencies:
            u_vib += self.h * freq / (np.exp(self.h * freq / (self.kb * self.temperature)) - 1)
        return u_vib

    def translational_entropy(self):
        """
        Calculate the translational contribution to the entropy.
        
        Returns:
        float: Translational entropy in J/K.
        """
        q_trans = self.translational_partition_function()
        return self.kb * (np.log(q_trans) + 5 / 2)
    
    def rotational_entropy(self):
        """
        Calculate the rotational contribution to the entropy.
        
        Returns:
        float: Rotational entropy in J/K.
        """
        q_rot = self.rotational_partition_function()
        return self.kb * (np.log(q_rot) + 3 / 2)
    
    def vibrational_entropy(self):
        """
        Calculate the vibrational contribution to the entropy.
        
        Returns:
        float: Vibrational entropy in J/K.
        """
        s_vib = 0.0
        for freq in self.vibrational_frequencies:
            x = self.h * freq / (self.kb * self.temperature)
            s_vib += x / (np.exp(x) - 1) - np.log(1 - np.exp(-x))
        return self.kb * s_vib

    def vibrational_zero_point_energy(self):
        """
        Calculate the vibrational zero-point energy (ZPE) of a molecule.
    
        Returns:
        float: Vibrational zero-point energy in Hartrees.
        """
        zpe = 0.0
        for freq in self.vibrational_frequencies:
            zpe += 0.5 * self.h * freq * self.c  # freq in cm⁻¹ to Hz and then to Joules
        return zpe / self.hartree_to_joule  # Convert Joules to Hartrees

    def calculate_state_functions(self):
        """
        Calculate the thermodynamic properties for the 3D molecule, including the Gibbs free energy.
        
        Returns:
        dict: A dictionary containing partition functions, energy, entropy, and Gibbs free energy.
        """
        q_trans = self.translational_partition_function()
        q_rot = self.rotational_partition_function()
        q_vib = self.vibrational_partition_function()
        
        u_trans = self.translational_energy()
        u_rot = self.rotational_energy()
        u_vib = self.vibrational_energy()
        
        s_trans = self.translational_entropy()
        s_rot = self.rotational_entropy()
        s_vib = self.vibrational_entropy()
        
        # Total internal energy and entropy
        u_total = u_trans + u_rot + u_vib
        s_total = s_trans + s_rot + s_vib
        
        # Calculate Gibbs free energy: G = U + PV - TS
        p_in_pa = self.pressure * 1e5  # Convert pressure from atm to Pa
        gibbs_free_energy = u_total + p_in_pa * self.volume - self.temperature * s_total
        print(u_total + p_in_pa * self.volume - self.temperature * s_total)

        vibrational_zero_point_energy = self.vibrational_zero_point_energy():
        
        return {
            "translational_partition_function": q_trans,
            "rotational_partition_function": q_rot,
            "vibrational_partition_function": q_vib,
            "translational_energy": u_trans,
            "rotational_energy": u_rot,
            "vibrational_energy": u_vib,
            "translational_entropy": s_trans,
            "rotational_entropy": s_rot,
            "vibrational_entropy": s_vib,
            "gibbs_free_energy": gibbs_free_energy
            "vibrational_zero_point_energy": vibrational_zero_point_energy
        }

# Example usage:
mass_of_molecule = 3.34757e-26  # in kg (example for a diatomic molecule)
rotational_constants = [1e-40, 1e-40, 1e-40]  # example values for A, B, C in m^-1
vibrational_frequencies = [1e13, 2e13]  # example values in Hz
molecule_mechanics = Molecule3DMechanics(mass_of_molecule, rotational_constants, vibrational_frequencies)
properties = molecule_mechanics.calculate_state_functions()
print(properties)

