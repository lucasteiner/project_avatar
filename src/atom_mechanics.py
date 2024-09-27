from scipy.constants import Planck, Boltzmann, Avogadro, gas_constant
import numpy as np

class AtomMechanics:
    def __init__(self, mass, temperature=298.15, pressure=1.0, volume=None):
        """
        Initialize the AtomMechanics class with basic thermodynamic properties.
        
        Parameters:
        mass (float): Mass of the atom in kg.
        temperature (float): Temperature in Kelvin (default 298.15 K).
        pressure (float): Pressure in atmospheres (default 1 atm).
        volume (float): Volume in liters (optional, can be calculated or provided).
        """
        # Constants from scipy
        self.h = Planck  # Planck constant
        self.kb = Boltzmann  # Boltzmann constant
        self.na = Avogadro  # Avogadro's number
        self.r = gas_constant  # Gas constant

        self.mass = mass
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
        Calculate the translational partition function for an atom.
        
        Returns:
        float: Translational partition function.
        """
        q_trans = ((2 * np.pi * self.mass * self.kb * self.temperature) / (self.h ** 2)) ** (3 / 2)
        q_trans *= self.volume
        return q_trans
    
    def translational_energy(self):
        """
        Calculate the translational contribution to the internal energy.
        
        Returns:
        float: Translational energy in Joules.
        """
        return (3 / 2) * self.kb * self.temperature
    
    def translational_entropy(self):
        """
        Calculate the translational contribution to the entropy.
        
        Returns:
        float: Translational entropy in J/K.
        """
        q_trans = self.translational_partition_function()
        return self.kb * (np.log(q_trans) + 5 / 2)
    
    def calculate_state_functions(self):
        """
        Calculate the thermodynamic properties for the atom, including the Gibbs free energy.
        
        Returns:
        dict: A dictionary containing translational partition function, energy, entropy, and Gibbs free energy.
        """
        q_trans = self.translational_partition_function()
        u_trans = self.translational_energy()
        s_trans = self.translational_entropy()
        
        # Calculate the Gibbs free energy: G = U + PV - TS
        p_in_pa = self.pressure * 1e5  # Convert pressure from atm to Pa
        gibbs_free_energy = u_trans + p_in_pa * self.volume - self.temperature * s_trans
        
        return {
            "translational_partition_function": q_trans,
            "ln(translational_partition_function)": np.log(q_trans),
            "translational_energy": u_trans,
            "translational_entropy": s_trans,
            "gibbs_free_energy": gibbs_free_energy
        }

# Example usage:
mass_of_hydrogen = 1.6735575e-27  # in kg
atom_mechanics = AtomMechanics(mass_of_hydrogen)
properties = atom_mechanics.calculate_state_functions()
print(properties)

