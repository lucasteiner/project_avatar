import pytest
import numpy as np

# Physical constants
TOLERANCE = 0.1  # Tolerance for comparing floating-point numbers

def test_mechanics_qrrho_with_paf2(paf2):
    """ Compares own thermodynamic functions with those of Orca
    Tests for qRRHO correction compared with Orca
    """
    results = []
    results.append(np.isclose(paf2.mechanics.G_total, 1763.38864, atol=0.03))
    print(paf2.mechanics.G_total, 1763.38864)
    assert all(results), f"{results}"

def test_mechanics_with_smi2_ace_thf_for_liquids(smi2_ace_thf4_for_liquids):
    """ Compares own thermodynamic functions with those of Turbomole/QCDC
    Tests if volume correction is applied correctly
    """
    results = []
    print('G:', smi2_ace_thf4_for_liquids.mechanics.G, smi2_ace_thf4_for_liquids.thermal_corrections)
    results.append(np.isclose(smi2_ace_thf4_for_liquids.mechanics.G, smi2_ace_thf4_for_liquids.thermal_corrections, atol=0.3))
    assert all(results), f"{results}"

def test_mechanics_with_smi2_ace_thf(smi2_ace_thf4):
    """ Compares own thermodynamic functions with those of Turbomole, freeh.out
    """
    results = []
    results.append(np.isclose(smi2_ace_thf4.mechanics.zpe, 1475., atol=0.1))
    results.append(np.isclose(smi2_ace_thf4.mechanics.U, 1575.05, atol=0.03))
    results.append(np.isclose(smi2_ace_thf4.mechanics.H, 1577.53, atol=0.03))
    results.append(np.isclose(np.log(smi2_ace_thf4.mechanics.q_trans), 20.53, atol=0.03))
    results.append(np.isclose(np.log(smi2_ace_thf4.mechanics.q_vib), 49.24, atol=0.03))
    results.append(np.isclose(np.log(smi2_ace_thf4.mechanics.q_rot), 17.24, atol=0.03))
    results.append(np.isclose(smi2_ace_thf4.mechanics.S, 1.06765, atol=0.0003))
    results.append(np.isclose(smi2_ace_thf4.mechanics.G, 1259.21, atol=0.3))
    print('qRRHO2:', smi2_ace_thf4.thermal_corrections - smi2_ace_thf4.mechanics.G)
    print('G_total(qcdc):', smi2_ace_thf4.thermal_corrections)
    print('G_total:', smi2_ace_thf4.mechanics.G_total)
    results.append(np.isclose(smi2_ace_thf4.mechanics.G_total, smi2_ace_thf4.thermal_corrections, atol=2.3))
    #print('Q_trans:', np.log(smi2_ace_thf4.mechanics.q_trans), 20.53)
    #print('Q_vib:', np.log(smi2_ace_thf4.mechanics.q_vib), 49.24)
    #print('Q_rot:', np.log(smi2_ace_thf4.mechanics.q_rot), 17.24)
    #print('S:', smi2_ace_thf4.mechanics.S, 1.06765)
    #print('G:', smi2_ace_thf4.mechanics.G, 1259.21)

    assert all(results), f"{results}"

def test_mechanics_with_li_atom(li_atom):
    """ Compares own thermodynamic functions with those of Turbomole for a single atom
    """
    results = []
    results.append(np.isclose(li_atom.mechanics.zpe, 0.0, atol=0.1))
    results.append(np.isclose(li_atom.mechanics.U, 3.72, atol=0.03))
    results.append(np.isclose(li_atom.mechanics.H, 6.20, atol=0.03))
    results.append(np.isclose(np.log(li_atom.mechanics.q_trans), 13.51, atol=0.03))
    results.append(np.isclose(np.log(li_atom.mechanics.q_vib), 0.0, atol=0.03))
    results.append(np.isclose(np.log(li_atom.mechanics.q_rot), 0.0, atol=0.03))
    results.append(np.isclose(li_atom.mechanics.S, 0.13310, atol=0.0003))
    results.append(np.isclose(li_atom.mechanics.G, -33.49, atol=0.3))
    #print('Q_trans:', np.log(li_atom.mechanics.q_trans), 20.53)
    #print('Q_vib:', np.log(li_atom.mechanics.q_vib), 49.24)
    #print('Q_rot:', np.log(li_atom.mechanics.q_rot), 17.24)
    #print('S:', li_atom.mechanics.S, 1.06765)
    #print('G:', li_atom.mechanics.G, 1259.21)

    assert all(results), f"{results}"

def test_mechanics_with_linear_lih(linear_lih):
    """ Compares own thermodynamic functions with those of Turbomole for a single atom
    """
    results = []
    results.append(np.isclose(linear_lih.mechanics.zpe, 6.973, atol=0.1))
    results.append(np.isclose(linear_lih.mechanics.U, 13.22, atol=0.03))
    results.append(np.isclose(linear_lih.mechanics.H, 15.70, atol=0.03))
    results.append(np.isclose(np.log(linear_lih.mechanics.q_trans), 13.71, atol=0.03))
    results.append(np.isclose(np.log(linear_lih.mechanics.q_vib), 0.00, atol=0.03))
    results.append(np.isclose(np.log(linear_lih.mechanics.q_rot), 3.44, atol=0.03))
    results.append(np.isclose(linear_lih.mechanics.S, 0.17192, atol=0.0003))
    results.append(np.isclose(linear_lih.mechanics.G, -35.56, atol=0.3))
    print(results)
    #print('Q_trans:', np.log(linear_lih.mechanics.q_trans), 20.53)
    #print('Q_vib:', np.log(linear_lih.mechanics.q_vib), 49.24)
    #print('Q_rot:', np.log(linear_lih.mechanics.q_rot), 17.24)
    #print('S:', linear_lih.mechanics.S, 1.06765)
    #print('G:', linear_lih.mechanics.G, 1259.21)

    assert all(results), f"{results}"

#Test mechanics for 1-3D molecules



#def test_water_molecule_properties():
#    # Data for water molecule
#    symbols = ['O', 'H', 'H']
#    coordinates = [
#        [0.0000000, -0.0177249, 0.0000000],
#        [0.7586762, 0.5948624, 0.0000000],
#        [-0.7586762, 0.5948624, 0.0000000]
#    ]
#
#    coordinates = [
#          [  0.00000006207532, 0.01178596715832, -0.00000002057039],
#          [  0.77274638535169, 0.58010292019012, -0.00000370986080],
#          [ -0.77274031085702, 0.58011101263371, 0.00000168490548]
#    ]
#    frequencies = np.array([1608.67, 3684.12, 3783.07])  # in cm^-1
#    temperature = 298.15  # in K
#    symmetry_number = 1  # Given symmetry number
#
#    # Expected results (from user data)
#    expected_zpe = 53.82  # kJ/mol
#    expected_gibbs_free_energy = 5.56 - expected_zpe  # kJ/mol
#    expected_internal_energy = 61.26 - expected_zpe  # kJ/mol
#    expected_enthalpy = 63.74 - expected_zpe # kJ/mol
#    expected_entropy = 0.19514  # J/mol·K
#
#    # Create Molecule instance
#    water = Molecule(
#        symbols=symbols,
#        coordinates=coordinates,
#        frequencies=frequencies,
#        symmetry_number=symmetry_number
#    )
#
#    # Calculate zero-point energy
#    zpe = water.mechanics.zero_point_energy()
#    assert abs(zpe - expected_zpe) < 0.5, f"ZPE: Expected {expected_zpe}, got {zpe}"
#
#    # Calculate thermodynamic properties
#    thermo_props = water.mechanics.thermodynamic_properties(temperature)
#
#    # Compare internal energy
#    internal_energy = thermo_props['Internal energy']
#    assert abs(internal_energy - expected_internal_energy) < TOLERANCE, \
#        f"Internal Energy: Expected {expected_internal_energy}, got {internal_energy}"
#
#    # Compare enthalpy
#    enthalpy = thermo_props['Enthalpy']
#    assert abs(enthalpy - expected_enthalpy) < TOLERANCE, \
#        f"Enthalpy: Expected {expected_enthalpy}, got {enthalpy}"
#
#    # Compare entropy
#    entropy = thermo_props['Entropy']
#    assert abs(entropy - expected_entropy) < 0.05, \
#        f"Entropy: Expected {expected_entropy}, got {entropy}"
#
#    # Compare Gibbs free energy
#    gibbs_free_energy = thermo_props['Gibbs free energy']
#    assert abs(gibbs_free_energy - expected_gibbs_free_energy) < 0.3, \
#        f"Gibbs Free Energy: Expected {expected_gibbs_free_energy}, got {gibbs_free_energy}"
#
#def test_vibrational_entropy_water():
#    """
#    Test the vibrational entropy calculation for the water molecule.
#    """
#    # Data for water molecule
#    symbols = ['O', 'H', 'H']
#    coordinates = [
#        [0.00000006207532, 0.01178596715832, -0.00000002057039],
#        [0.77274638535169, 0.58010292019012, -0.00000370986080],
#        [-0.77274031085702, 0.58011101263371, 0.00000168490548]
#    ]
#    frequencies = np.array([1538.61, 3642.19, 3650.89])  # in cm^-1
#    temperature = 298.15  # Standard temperature in Kelvin
#    symmetry_number = 2  # Correct symmetry number for water molecule
#
#    # Expected vibrational entropy in cal/K/mol
#    expected_entropy_vib_cal_per_mol = 0.010  # cal/K/mol
#
#    # Create Molecule instance
#    water = Molecule(
#        symbols=symbols,
#        coordinates=coordinates,
#        frequencies=frequencies,
#        symmetry_number=symmetry_number
#    )
#
#    # Calculate vibrational entropy (in J/K per molecule)
#    entropy_vib_per_molecule_J_per_K = water.mechanics.entropy_vibrational(temperature)
#
#    # Convert to J/K/mol
#    entropy_vib_J_per_K_per_mol = entropy_vib_per_molecule_J_per_K * const.N_A
#
#    # Convert to cal/K/mol (1 cal = 4.184 J)
#    entropy_vib_cal_per_mol = entropy_vib_J_per_K_per_mol / 4.184
#
#    # Tolerance for comparison
#    tolerance = 0.001  # cal/K/mol
#
#    # Assert
#    assert abs(entropy_vib_cal_per_mol - expected_entropy_vib_cal_per_mol) < tolerance, \
#        f"Vibrational Entropy: Expected {expected_entropy_vib_cal_per_mol} cal/K/mol, got {entropy_vib_cal_per_mol:.3f} cal/K/mol"
#
#def test_rotational_entropy_water():
#    """
#    Test the rotational entropy calculation for the water molecule.
#    """
#    # Data for water molecule
#    symbols = ['O', 'H', 'H']
#    coordinates = [
#        [0.00000006207532, 0.01178596715832, -0.00000002057039],
#        [0.77274638535169, 0.58010292019012, -0.00000370986080],
#        [-0.77274031085702, 0.58011101263371, 0.00000168490548]
#    ]
#    frequencies = np.array([1538.61, 3642.19, 3650.89])  # in cm^-1
#    temperature = 298.15  # Standard temperature in Kelvin
#    symmetry_number = 2  # Correct symmetry number for water molecule
#
#    # Expected rotational entropy in cal/K/mol
#    expected_entropy_rot_cal_per_mol = 10.434  # cal/K/mol
#
#    # Create Molecule instance
#    water = Molecule(
#        symbols=symbols,
#        coordinates=coordinates,
#        frequencies=frequencies,
#        symmetry_number=symmetry_number
#    )
#
#    # Calculate rotational entropy (in J/K per molecule)
#    entropy_rot_per_molecule_J_per_K = water.mechanics.entropy_rotational(temperature)
#
#    # Convert to J/K/mol
#    entropy_rot_J_per_K_per_mol = entropy_rot_per_molecule_J_per_K * const.N_A
#
#    # Convert to cal/K/mol (1 cal = 4.184 J)
#    entropy_rot_cal_per_mol = entropy_rot_J_per_K_per_mol / 4.184
#
#    # Tolerance for comparison
#    tolerance = 0.1  # cal/K/mol
#
#    # Assert
#    assert abs(entropy_rot_cal_per_mol - expected_entropy_rot_cal_per_mol) < tolerance, \
#        f"Rotational Entropy: Expected {expected_entropy_rot_cal_per_mol} cal/K/mol, got {entropy_rot_cal_per_mol:.3f} cal/K/mol"
#
#def test_translational_entropy_water():
#    """
#    Test the translational entropy calculation for the water molecule.
#    """
#    # Data for water molecule
#    symbols = ['O', 'H', 'H']
#    coordinates = [
#        [0.00000006207532, 0.01178596715832, -0.00000002057039],
#        [0.77274638535169, 0.58010292019012, -0.00000370986080],
#        [-0.77274031085702, 0.58011101263371, 0.00000168490548]
#    ]
#    frequencies = np.array([1538.61, 3642.19, 3650.89])  # in cm^-1
#    temperature = 298.15  # Standard temperature in Kelvin
#    symmetry_number = 2  # Correct symmetry number for water molecule
#
#    # Expected translational entropy in cal/K/mol
#    expected_entropy_trans_cal_per_mol = 34.593  # cal/K/mol
#
#    # Create Molecule instance
#    water = Molecule(
#        symbols=symbols,
#        coordinates=coordinates,
#        frequencies=frequencies,
#        symmetry_number=symmetry_number
#    )
#
#    # Calculate translational entropy (in J/K per molecule)
#    entropy_trans_per_molecule_J_per_K = water.mechanics.entropy_translational(temperature)
#
#    # Convert to J/K/mol
#    entropy_trans_J_per_K_per_mol = entropy_trans_per_molecule_J_per_K * const.N_A
#
#    # Convert to cal/K/mol (1 cal = 4.184 J)
#    entropy_trans_cal_per_mol = entropy_trans_J_per_K_per_mol / 4.184
#
#    # Tolerance for comparison
#    tolerance = 0.05  # cal/K/mol
#
#    # Assert
#    assert abs(entropy_trans_cal_per_mol - expected_entropy_trans_cal_per_mol) < tolerance, \
#        f"Translational Entropy: Expected {expected_entropy_trans_cal_per_mol} cal/K/mol, got {entropy_trans_cal_per_mol:.3f} cal/K/mol"
#
#def test_is_linear():
#    """
#    Test the is_linear method.
#    """
#    # Linear molecule: CO2
#    symbols_linear = ['O', 'C', 'O']
#    coordinates_linear = [
#        [-1.16, 0.0, 0.0],  # O
#        [0.0, 0.0, 0.0],    # C
#        [1.16, 0.0, 0.0]    # O
#    ]
#    frequencies = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1000.16, 2000.0, 3000.0])
#
#    molecule_linear = Molecule(symbols_linear, coordinates_linear, frequencies=frequencies)
#    assert molecule_linear.mechanics.is_linear(), "CO2 should be linear"
#
#    # Non-linear molecule: H2O
#    symbols_nonlinear = ['O', 'H', 'H']
#    coordinates_nonlinear = [
#        [0.0000000, -0.0177249, 0.0000000],  # O
#        [0.7586762, 0.5948624, 0.0000000],   # H
#        [-0.7586762, 0.5948624, 0.0000000]   # H
#    ]
#    frequencies = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.16, 2000.0, 3000.0])
#    molecule_nonlinear = Molecule(symbols_nonlinear, coordinates_nonlinear, frequencies=frequencies)
#    assert not molecule_nonlinear.mechanics.is_linear(), "H2O should not be linear"


if __name__ == "__main__":
    pytest.main()

