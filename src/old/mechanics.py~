#!/bin/env python3
import numpy as np

class Mechanics:
    def __init__(self, temperature=298.15, pressure=1.0, volume=None,
                 vibrational_frequencies=None):
        """
        Initialize the Mechanics class with basic thermodynamic properties.
        
        Parameters:
        temperature (float): Temperature in Kelvin (default 298.15 K).
        pressure (float): Pressure in atmospheres (default 1 atm).
        volume (float): Volume in liters (default None, can be calculated or provided).
        """
        self.temperature = temperature
        self.pressure = pressure
        self.volume = volume


        # Properties
        #self._electronic_energy = electronic_energy
        self._vibrational_frequencies = vibrational_frequencies
        self._gibbs_free_energy = gibbs_free_energy


        self._inner_energy = inner_energy
        self._enthalpy = enthalpy
        self._entropy = entropy
        self._helmholz_free_energy = helmholz_free_energy

        # Property for gibbs_free_energy
        @property
        def gibbs_free_energy(self):
            return self._gibbs_free_energy
     
        @gibbs_free_energy.setter
        def gibbs_free_energy(self, energy):
            if energy is not None and not isinstance(energy, (int, float)):
                raise ValueError("Gibbs free energy must be a numeric value.")
            self._gibbs_free_energy = energy
     
        # Property for temperature
        @property
        def temperature(self):
            return self._temperature
     
        @temperature.setter
        def temperature(self, temp):
            if temp < 0:
                raise ValueError("Temperature cannot be below absolute zero.")
            self._temperature = temp
     
        # Property for inner_energy
        @property
        def inner_energy(self):
            return self._inner_energy
     
        @inner_energy.setter
        def inner_energy(self, energy):
            if energy is not None and not isinstance(energy, (int, float)):
                raise ValueError("Inner energy must be a numeric value.")
            self._inner_energy = energy
     
        # Property for enthalpy
        @property
        def enthalpy(self):
            return self._enthalpy
     
        @enthalpy.setter
        def enthalpy(self, energy):
            if energy is not None and not isinstance(energy, (int, float)):
                raise ValueError("Enthalpy must be a numeric value.")
            self._enthalpy = energy
     
        # Property for entropy
        @property
        def entropy(self):
            return self._entropy
     
        @entropy.setter
        def entropy(self, energy):
            if energy is not None and not isinstance(energy, (int, float)):
                raise ValueError("Entropy must be a numeric value.")
            self._entropy = energy
     
        # Property for helmholz_free_energy
        @property
        def helmholz_free_energy(self):
            return self._helmholz_free_energy
     
        @helmholz_free_energy.setter
        def helmholz_free_energy(self, energy):
            if energy is not None and not isinstance(energy, (int, float)):
                raise ValueError("Helmholtz free energy must be a numeric value.")
            self._helmholz_free_energy = energy
     
     
     
