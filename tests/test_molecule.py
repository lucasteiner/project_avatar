import pytest
import numpy as np
from molecule import Molecule

def test_molecule_initialization():
    """
    Test the initialization of the Molecule class.
    """
    symbols = ['H', 'O', 'H']
    coordinates = [
        [0.0, 0.757, 0.586],  # H
        [0.0, 0.0, 0.0],      # O
        [0.0, -0.757, 0.586]  # H
    ]
    molecule = Molecule(symbols, coordinates)
    assert isinstance(molecule.symbols, np.ndarray), "Symbols should be a NumPy array"
    assert isinstance(molecule.coordinates, np.ndarray), "Coordinates should be a NumPy array"
    assert np.array_equal(molecule.symbols, np.array(symbols)), "Symbols do not match"
    assert np.array_equal(molecule.coordinates, np.array(coordinates)), "Coordinates do not match"

def test_center_of_mass():
    """
    Test the center_of_mass method.
    """
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.0000000, -0.0177249, 0.0000000],  # O
        [0.7586762, 0.5948624, 0.0000000],   # H
        [-0.7586762, 0.5948624, 0.0000000]   # H
    ]
    molecule = Molecule(symbols, coordinates)
    com = molecule.center_of_mass()
    # Manually calculated center of mass
    atomic_masses = {
        'H': 1.00784,
        'O': 15.9994
    }
    masses = np.array([atomic_masses[s] for s in symbols])
    total_mass = masses.sum()
    expected_com = np.dot(masses, coordinates) / total_mass
    assert np.allclose(com, expected_com, atol=1e-6), "Center of mass calculation is incorrect"

def test_moments_of_inertia():
    """
    Test the moments_of_inertia method.
    """
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.0000000, -0.0177249, 0.0000000],  # O
        [0.7586762, 0.5948624, 0.0000000],   # H
        [-0.7586762, 0.5948624, 0.0000000]   # H
    ]
    molecule = Molecule(symbols, coordinates)
    moments = molecule.moments_of_inertia()
    # Verify that moments are a NumPy array of length 3
    assert isinstance(moments, np.ndarray), "Moments of inertia should be a NumPy array"
    assert moments.shape == (3,), "Moments of inertia should be a vector of length 3"
    # Since exact values are complex to calculate, we can check for reasonable ranges
    assert np.all(moments > 0), "Moments of inertia should be positive"

def test_moments_of_inertia_water():
    """
    Test the moments_of_inertia method for the water molecule.
    """
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.0000000, -0.0177249, 0.0000000],  # O
        [0.7586762, 0.5948624, 0.0000000],   # H
        [-0.7586762, 0.5948624, 0.0000000]   # H
    ]
    molecule = Molecule(symbols, coordinates)
    moments = molecule.moments_of_inertia()
    # Expected moments of inertia in amu·Å²
    expected_moments = np.array([0.67177502, 1.1602044, 1.83197942])

    # The calculated moments may not be in the same order as expected
    moments_sorted = np.sort(moments)
    expected_moments_sorted = np.sort(expected_moments)

    assert np.allclose(moments_sorted, expected_moments_sorted, atol=1e-6), \
        f"Moments of inertia do not match expected values:\nCalculated: {moments_sorted}\nExpected: {expected_moments_sorted}"

def test_recenter():
    """
    Test the recenter method.
    """
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.0000000, -0.0177249, 0.0000000],  # O
        [0.7586762, 0.5948624, 0.0000000],   # H
        [-0.7586762, 0.5948624, 0.0000000]   # H
    ]
    molecule = Molecule(symbols, coordinates)
    original_com = molecule.center_of_mass()
    molecule.coordinates = molecule.recenter()
    com = molecule.center_of_mass()
    expected_com = np.array([0.0, 0.0, 0.0])
    assert np.allclose(com, expected_com, atol=1e-3), "Center of mass should be at the origin after recentering"
    # Ensure coordinates have been shifted appropriately
    shifted_coordinates = np.array(coordinates) - original_com
    assert np.allclose(molecule.coordinates, shifted_coordinates, atol=1e-3), "Coordinates not shifted correctly after recentering"

def test_reorder_atoms():
    """
    Test the reorder_atoms method.
    """
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.0000000, -0.0177249, 0.0000000],  # O
        [0.7586762, 0.5948624, 0.0000000],   # H1
        [-0.7586762, 0.5948624, 0.0000000]   # H2
    ]
    molecule = Molecule(symbols, coordinates)
    new_order = [1, 2, 0]  # Move O to the end
    molecule.symbols, molecule.coordinates = molecule.reorder_atoms(new_order)
    expected_symbols = ['H', 'H', 'O']
    expected_coordinates = [
        [0.7586762, 0.5948624, 0.0],
        [-0.7586762, 0.5948624, 0.0],
        [0.0, -0.0177249, 0.0]
    ]
    assert np.array_equal(molecule.symbols, np.array(expected_symbols)), "Symbols not reordered correctly"
    assert np.allclose(molecule.coordinates, expected_coordinates, atol=1e-6), "Coordinates not reordered correctly"

def test_set_and_get_property():
    """
    Test the set_property and get_property methods.
    """
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.0000000, -0.0177249, 0.0000000],
        [0.7586762, 0.5948624, 0.0000000],
        [-0.7586762, 0.5948624, 0.0000000]
    ]
    molecule = Molecule(symbols, coordinates)
    # Set a new property
    molecule.set_property('energy', -76.4)
    energy = molecule.get_property('energy')
    assert energy == -76.4, "Property 'energy' not set or retrieved correctly"
    # Test retrieval of a non-existent property
    dipole_moment = molecule.get_property('dipole_moment')
    assert dipole_moment is None, "Non-existent property should return None"

def test_from_xyz(tmp_path):
    """
    Test the from_xyz class method.
    """
    # Create a temporary XYZ file
    xyz_content = """3
Water molecule
O   0.0000000   -0.0177249    0.0000000
H   0.7586762    0.5948624    0.0000000
H  -0.7586762    0.5948624    0.0000000
"""
    xyz_file = tmp_path / "water.xyz"
    xyz_file.write_text(xyz_content)
    molecule = Molecule.from_xyz(str(xyz_file))
    expected_symbols = ['O', 'H', 'H']
    expected_coordinates = np.array([
        [0.0, -0.0177249, 0.0],
        [0.7586762, 0.5948624, 0.0],
        [-0.7586762, 0.5948624, 0.0]
    ])
    assert np.array_equal(molecule.symbols, np.array(expected_symbols)), "Symbols from XYZ not read correctly"
    assert np.allclose(molecule.coordinates, expected_coordinates, atol=1e-6), "Coordinates from XYZ not read correctly"

def test_invalid_reorder_atoms():
    """
    Test that providing an invalid new_order raises an error.
    """
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.0, 0.0, 0.0],
        [0.7586762, 0.5948624, 0.0],
        [-0.7586762, 0.5948624, 0.0]
    ]
    molecule = Molecule(symbols, coordinates)
    new_order = [0, 1]  # Incorrect length
    with pytest.raises(ValueError, match="new_order must be a permutation of indices 0 to N-1."):
        molecule.reorder_atoms(new_order)

def test_kabsch_align():
    """
    Test the kabsch_rotate method to ensure that it modifies the coordinates
    of the molecule correctly by aligning it with another molecule.
    """
    symbols = ['O', 'H', 'H']
    coordinates1 = [
        [0.0000000, -0.0177249, 0.0000000],  # O
        [0.7586762, 0.5948624, 0.0000000],   # H1
        [-0.7586762, 0.5948624, 0.0000000]   # H2
    ]
    coordinates2 = [
        [0.0000000, -0.0170000, 0.0000000],  # O (slightly different)
        [0.7590000, 0.5940000, 0.0000000],   # H1 (slightly different)
        [-0.7590000, 0.5940000, 0.0000000]   # H2 (slightly different)
    ]

    molecule1 = Molecule(symbols, coordinates1)
    molecule2 = Molecule(symbols, coordinates2)

    # Perform Kabsch rotation to align molecule1 to molecule2
    molecule1.coordinates, rotation_matrix = molecule1.kabsch_align(molecule2)

    # Calculate the RMSD after alignment
    rmsd_value = molecule1.calculate_rmsd(molecule2)

    # Ensure the RMSD is very small after rotation
    assert rmsd_value < 1e-1, "RMSD should be close to zero after Kabsch alignment and rotation"

    # Ensure that the coordinates have been modified (not the original ones)
    assert not np.allclose(molecule1.coordinates, coordinates1, atol=1e-6), \
        "The coordinates of molecule1 should have changed after apply_kabsch"

def test_kabsch_align2():
    """
    Test the Kabsch algorithm to align two molecules.
    """
    symbols1 = ['H', 'H', 'O']
    coordinates1 = [
        [0.0,  0.5,  0.5],  # H
        [0.0,  0.5, -0.5],  # H
        [0.0,  0.0,  0.0],      # O
    ]

    symbols2 = ['H', 'H', 'O']
    coordinates2 = [
        [0.0,  0.5,-0.5],  # H(shifted version of coordinates1)
        [0.0,  0.5, 0.5],  # H
        [0.0,  0.0, 0.0],  # O
    ]

    molecule1 = Molecule(symbols1, coordinates1)
    molecule2 = Molecule(symbols2, coordinates2)

    aligned_coords, rotation_matrix = molecule1.kabsch_align(molecule2)

    # Check that the aligned coordinates are close to molecule2's coordinates
    assert np.allclose(aligned_coords, molecule2.coordinates, atol=1e-1), \
        "Aligned coordinates do not match expected values."

    # Check that the rotation matrix is valid (orthogonal matrix)
    assert np.allclose(np.dot(rotation_matrix, rotation_matrix.T), np.eye(3), atol=1e-6), \
        "Rotation matrix is not orthogonal."

def test_calculate_rmsd():
    """
    Test the RMSD calculation between two molecules.
    """
    symbols1 = ['H', 'O', 'H']
    coordinates1 = [
        [0.0, 0.757, 0.586],  # H
        [0.0, 0.0, 0.0],      # O
        [0.0, -0.757, 0.586]  # H
    ]

    symbols2 = ['H', 'O', 'H']
    coordinates2 = [
        [0.0, 0.757, 0.586],  # H (same as coordinates1)
        [0.0, 0.0, 0.0],      # O
        [0.0, -0.757, 0.586]  # H
    ]

    molecule1 = Molecule(symbols1, coordinates1)
    molecule2 = Molecule(symbols2, coordinates2)

    rmsd_value = molecule1.calculate_rmsd(molecule2)

    # Since the molecules are identical, the RMSD should be close to zero
    assert np.isclose(rmsd_value, 0.0, atol=1e-1), "RMSD should be close to zero for identical molecules."

def test_apply_hungarian():
    # Create a test molecule with symbols and coordinates
    symbols = ['O', 'H', 'H']
    coordinates = [
        [1.0, 0.0, 0.0],  # Oxygen
        [0.0, 1.0, 0.0],  # Hydrogen 1
        [0.0, -1.0, 0.0]  # Hydrogen 2
    ]
    molecule = Molecule(symbols, coordinates)
    
    # Create another molecule for comparison (same atoms, different order)
    other_symbols = ['H', 'O', 'H']
    other_coordinates = [
        [0.0, 1.0, 0.0],  # Hydrogen 1
        [1.0, 0.0, 0.0],  # Oxygen
        [0.0, -1.0, 0.0]  # Hydrogen 2
    ]
    other_molecule = Molecule(other_symbols, other_coordinates)
    
    # Get the Hungarian reorder indices
    (molecule.symbols, molecule.coordinates, reorder_indices) = molecule.hungarian_reorder(other_molecule)
    print(reorder_indices)
    
    ## Apply the Hungarian reordering to the test molecule
    #molecule.apply_hungarian(reorder_indices)
    
    # Expected result after reordering (should match the other molecule)
    expected_symbols = np.array(['H', 'O', 'H'])
    expected_coordinates = np.array([
        [0.0, 1.0, 0.0],  # Hydrogen 1
        [1.0, 0.0, 0.0],  # Oxygen
        [0.0, -1.0, 0.0]  # Hydrogen 2
    ])
    
    # Assert that the symbols and coordinates are correctly reordered
    assert np.all(molecule.symbols == expected_symbols)
    np.testing.assert_array_equal(molecule.coordinates, np.array(expected_coordinates))

def test_copy_method():
    # Create a test molecule
    symbols = np.array(['O', 'H', 'H'])
    coordinates = np.array([
        [0.0, 0.0, 0.0],  # Oxygen
        [1.0, 0.0, 0.0],  # Hydrogen 1
        [0.0, 1.0, 0.0]   # Hydrogen 2
    ])
    energy = -75.0
    frequencies = np.array([1000, 1500, 2000])

    original_molecule = Molecule(symbols=symbols, coordinates=coordinates, energy=energy, frequencies=frequencies)

    # Create a copy of the molecule
    copied_molecule = original_molecule.copy()

    # Assert that the copied molecule has the same properties as the original
    np.testing.assert_array_equal(copied_molecule.symbols, original_molecule.symbols)
    np.testing.assert_array_equal(copied_molecule.coordinates, original_molecule.coordinates)
    np.testing.assert_array_equal(copied_molecule.frequencies, original_molecule.frequencies)
    assert copied_molecule.energy == original_molecule.energy

    # Modify the copy and ensure the original molecule is not affected
    copied_molecule.symbols[0] = 'C'
    copied_molecule.coordinates[0] = [2.0, 2.0, 2.0]
    copied_molecule.frequencies[0] = 3000
    copied_molecule.energy = -74.0

    # Ensure the original molecule remains unchanged
    assert original_molecule.symbols[0] == 'O'
    np.testing.assert_array_equal(original_molecule.coordinates[0], [0.0, 0.0, 0.0])
    assert original_molecule.frequencies[0] == 1000
    assert original_molecule.energy == -75.0

@pytest.fixture
def water_molecule():
    """Fixture to create a water molecule (H₂O) with arbitrary coordinates."""
    symbols = ['O', 'H', 'H']
    coordinates = [
        [0.0, 0.0, 0.0],   # Oxygen
        [0.0, 0.96, 0.0],  # Hydrogen 1
        [0.92, -0.36, 0.0] # Hydrogen 2
    ]
    return symbols, coordinates


def test_format_energetic_attributes(water_molecule):
    symbols, coordinates = water_molecule

    # Initialize molecule with energetic attributes
    molecule = Molecule(symbols=symbols, coordinates=coordinates, electronic_energy=-75.0, thermal_corrections=0.8, solvation_enthalpy=-2.5)
    expected_output = "E_el=-75.00, G_thr=0.80, G_solv=-2.50"
    assert molecule.format_energetic_attributes() == expected_output

    # Test with some attributes set to None
    molecule = Molecule(symbols=symbols, coordinates=coordinates, electronic_energy=-75.0, thermal_corrections=None, solvation_enthalpy=-2.5)
    expected_output = "E_el=-75.00, G_solv=-2.50"
    assert molecule.format_energetic_attributes() == expected_output


def test_g_total(water_molecule):
    symbols, coordinates = water_molecule

    # Initialize molecule with energetic attributes
    molecule = Molecule(symbols=symbols, coordinates=coordinates, electronic_energy=-75.0, thermal_corrections=0.8, solvation_enthalpy=-2.5)
    assert molecule.g_total() == pytest.approx(-76.7)

    # Test with some attributes set to None
    molecule = Molecule(symbols=symbols, coordinates=coordinates, electronic_energy=-75.0, thermal_corrections=None, solvation_enthalpy=-2.5)
    assert molecule.g_total() == pytest.approx(-77.5)


if __name__ == "__main__":
    pytest.main()

