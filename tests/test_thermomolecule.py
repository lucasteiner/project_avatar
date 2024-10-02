import pytest
from thermomolecule import ThermoMolecule
import numpy as np
import scipy.constants as const

# Physical constants
TOLERANCE = 0.1  # Tolerance for comparing floating-point numbers

def test_water_molecule_properties():
    # Data for water molecule
    symbols = ['O', 'H', 'H']
#    coordinates = [
#        [0.0000000, -0.0177249, 0.0000000],
#        [0.7586762, 0.5948624, 0.0000000],
#        [-0.7586762, 0.5948624, 0.0000000]
#    ]

    coordinates = [
          [  0.00000006207532, 0.01178596715832, -0.00000002057039],
          [  0.77274638535169, 0.58010292019012, -0.00000370986080],
          [ -0.77274031085702, 0.58011101263371, 0.00000168490548]
    ]
    frequencies = np.array([1608.67, 3684.12, 3783.07])  # in cm^-1
    temperature = 298.15  # in K
    symmetry_number = 1  # Given symmetry number

    # Expected results (from user data)
    expected_zpe = 53.82  # kJ/mol
    expected_gibbs_free_energy = 5.56 - expected_zpe  # kJ/mol
    expected_internal_energy = 61.26 - expected_zpe  # kJ/mol
    expected_enthalpy = 63.74 - expected_zpe # kJ/mol
    expected_entropy = 0.19514  # J/mol·K

    # Create ThermoMolecule instance
    water = ThermoMolecule(
        symbols=symbols,
        coordinates=coordinates,
        frequencies=frequencies,
        symmetry_number=symmetry_number
    )

    # Calculate zero-point energy
    zpe = water.zero_point_energy()
    assert abs(zpe - expected_zpe) < 0.5, f"ZPE: Expected {expected_zpe}, got {zpe}"

    # Calculate thermodynamic properties
    thermo_props = water.thermodynamic_properties(temperature)

    # Compare internal energy
    internal_energy = thermo_props['Internal energy']
    assert abs(internal_energy - expected_internal_energy) < TOLERANCE, \
        f"Internal Energy: Expected {expected_internal_energy}, got {internal_energy}"

    # Compare enthalpy
    enthalpy = thermo_props['Enthalpy']
    assert abs(enthalpy - expected_enthalpy) < TOLERANCE, \
        f"Enthalpy: Expected {expected_enthalpy}, got {enthalpy}"

    # Compare entropy
    entropy = thermo_props['Entropy']
    assert abs(entropy - expected_entropy) < 0.05, \
        f"Entropy: Expected {expected_entropy}, got {entropy}"

    # Compare Gibbs free energy
    gibbs_free_energy = thermo_props['Gibbs free energy']
    assert abs(gibbs_free_energy - expected_gibbs_free_energy) < 0.3, \
        f"Gibbs Free Energy: Expected {expected_gibbs_free_energy}, got {gibbs_free_energy}"

def test_vibrational_entropy_water():
    """
    Test the vibrational entropy calculation for the water molecule.
    """
    # Data for water molecule
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.00000006207532, 0.01178596715832, -0.00000002057039],
        [0.77274638535169, 0.58010292019012, -0.00000370986080],
        [-0.77274031085702, 0.58011101263371, 0.00000168490548]
    ]
    frequencies = np.array([1538.61, 3642.19, 3650.89])  # in cm^-1
    temperature = 298.15  # Standard temperature in Kelvin
    symmetry_number = 2  # Correct symmetry number for water molecule

    # Expected vibrational entropy in cal/K/mol
    expected_entropy_vib_cal_per_mol = 0.010  # cal/K/mol

    # Create ThermoMolecule instance
    water = ThermoMolecule(
        symbols=symbols,
        coordinates=coordinates,
        frequencies=frequencies,
        symmetry_number=symmetry_number
    )

    # Calculate vibrational entropy (in J/K per molecule)
    entropy_vib_per_molecule_J_per_K = water.entropy_vibrational(temperature)

    # Convert to J/K/mol
    entropy_vib_J_per_K_per_mol = entropy_vib_per_molecule_J_per_K * const.N_A

    # Convert to cal/K/mol (1 cal = 4.184 J)
    entropy_vib_cal_per_mol = entropy_vib_J_per_K_per_mol / 4.184

    # Tolerance for comparison
    tolerance = 0.001  # cal/K/mol

    # Assert
    assert abs(entropy_vib_cal_per_mol - expected_entropy_vib_cal_per_mol) < tolerance, \
        f"Vibrational Entropy: Expected {expected_entropy_vib_cal_per_mol} cal/K/mol, got {entropy_vib_cal_per_mol:.3f} cal/K/mol"

def test_rotational_entropy_water():
    """
    Test the rotational entropy calculation for the water molecule.
    """
    # Data for water molecule
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.00000006207532, 0.01178596715832, -0.00000002057039],
        [0.77274638535169, 0.58010292019012, -0.00000370986080],
        [-0.77274031085702, 0.58011101263371, 0.00000168490548]
    ]
    frequencies = np.array([1538.61, 3642.19, 3650.89])  # in cm^-1
    temperature = 298.15  # Standard temperature in Kelvin
    symmetry_number = 2  # Correct symmetry number for water molecule

    # Expected rotational entropy in cal/K/mol
    expected_entropy_rot_cal_per_mol = 10.434  # cal/K/mol

    # Create ThermoMolecule instance
    water = ThermoMolecule(
        symbols=symbols,
        coordinates=coordinates,
        frequencies=frequencies,
        symmetry_number=symmetry_number
    )

    # Calculate rotational entropy (in J/K per molecule)
    entropy_rot_per_molecule_J_per_K = water.entropy_rotational(temperature)

    # Convert to J/K/mol
    entropy_rot_J_per_K_per_mol = entropy_rot_per_molecule_J_per_K * const.N_A

    # Convert to cal/K/mol (1 cal = 4.184 J)
    entropy_rot_cal_per_mol = entropy_rot_J_per_K_per_mol / 4.184

    # Tolerance for comparison
    tolerance = 0.1  # cal/K/mol

    # Assert
    assert abs(entropy_rot_cal_per_mol - expected_entropy_rot_cal_per_mol) < tolerance, \
        f"Rotational Entropy: Expected {expected_entropy_rot_cal_per_mol} cal/K/mol, got {entropy_rot_cal_per_mol:.3f} cal/K/mol"

def test_translational_entropy_water():
    """
    Test the translational entropy calculation for the water molecule.
    """
    # Data for water molecule
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.00000006207532, 0.01178596715832, -0.00000002057039],
        [0.77274638535169, 0.58010292019012, -0.00000370986080],
        [-0.77274031085702, 0.58011101263371, 0.00000168490548]
    ]
    frequencies = np.array([1538.61, 3642.19, 3650.89])  # in cm^-1
    temperature = 298.15  # Standard temperature in Kelvin
    symmetry_number = 2  # Correct symmetry number for water molecule

    # Expected translational entropy in cal/K/mol
    expected_entropy_trans_cal_per_mol = 34.593  # cal/K/mol

    # Create ThermoMolecule instance
    water = ThermoMolecule(
        symbols=symbols,
        coordinates=coordinates,
        frequencies=frequencies,
        symmetry_number=symmetry_number
    )

    # Calculate translational entropy (in J/K per molecule)
    entropy_trans_per_molecule_J_per_K = water.entropy_translational(temperature)

    # Convert to J/K/mol
    entropy_trans_J_per_K_per_mol = entropy_trans_per_molecule_J_per_K * const.N_A

    # Convert to cal/K/mol (1 cal = 4.184 J)
    entropy_trans_cal_per_mol = entropy_trans_J_per_K_per_mol / 4.184

    # Tolerance for comparison
    tolerance = 0.05  # cal/K/mol

    # Assert
    assert abs(entropy_trans_cal_per_mol - expected_entropy_trans_cal_per_mol) < tolerance, \
        f"Translational Entropy: Expected {expected_entropy_trans_cal_per_mol} cal/K/mol, got {entropy_trans_cal_per_mol:.3f} cal/K/mol"


if __name__ == "__main__":
    pytest.main()

