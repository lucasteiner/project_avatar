import pytest
import numpy as np
from molecule import Molecule  # Assuming Molecule class is defined and imported
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
xyz_path = os.path.join(base_dir, 'xyz_structures/', )

@pytest.fixture
def molecule1():
    # Create a simple test molecule (for example, water molecule H2O)
    symbols = np.array(['O', 'H', 'H'])
    coordinates = np.array([
        [0.0, 0.0, 0.0],  # Oxygen
        [1.0, 0.0, 0.0],  # Hydrogen 1
        [0.0, 1.0, 0.0]   # Hydrogen 2
    ])
    energy = -75.0
    #moments_of_inertia = np.array([1.0, 2.0, 3.0])
    return Molecule(symbols, coordinates, energy)

@pytest.fixture
def molecule2():
    # Create another test molecule (slightly different water molecule H2O)
    symbols = np.array(['O', 'H', 'H'])
    coordinates = np.array([
        [0.0, 0.0, 0.0],  # Oxygen
        [1.0, 0.1, 0.0],  # Hydrogen 1
        [0.0, 1.0, 0.1]   # Hydrogen 2
    ])
    energy = -74.9
    #moments_of_inertia = np.array([1.01, 2.01, 3.01])
    return Molecule(symbols, coordinates, energy)

@pytest.fixture
def cc_alpha():
    return Molecule.from_xyz(xyz_path+'cc-alpha.xyz')

@pytest.fixture
def cc_beta():
    return Molecule.from_xyz(xyz_path+'cc-beta.xyz')

def test_compare_moments_of_inertia(molecule1, molecule2):
    # Test comparing moments of inertia
    result = molecule1.compare_moments_of_inertia(molecule2, tolerance=0.05)
    assert result is True, "Moments of inertia should be considered matching within tolerance."

def test_compare_energy(molecule1, molecule2):
    # Test comparing energy
    result = molecule1.compare_energy(molecule2, tolerance=0.2)
    assert result is np.bool(True), "Energy should be considered matching within tolerance."

def test_is_rmsd_below_threshold(molecule1, molecule2):
    # Test if RMSD is below the threshold
    result = molecule1.is_rmsd_below_threshold(molecule2, threshold=0.15)
    assert result is np.bool(True), "RMSD should be below the threshold."

def test_compare_molecule(molecule1, molecule2):
    # Test the combined comparison method
    result = molecule1.compare_molecule(molecule2, tolerance_inertia=0.05, tolerance_energy=0.2, rmsd_threshold=0.15)
    assert result == (True, True, True), "All comparisons should pass within given tolerances."

def test_reorder_after():
    # Create two example molecules for testing
    # Reference molecule: A linear molecule (like H2O but extended)
    reference_symbols = ["O", "H", "H"]
    reference_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    # Target molecule: The same structure but slightly rotated and translated
    target_symbols = ["O", "H", "H"]
    target_coords = np.array([[0.1, 0.1, 0.0], [1.1, 0.1, 0.0], [-0.9, 0.1, 0.0]])

    # Create Molecule instances
    reference_molecule = Molecule(reference_symbols, reference_coords)
    target_molecule = Molecule(target_symbols, target_coords)

    # Call the reorder_after method
    final_coords, combined_order = target_molecule.reorder_after(reference_molecule)

    # Expected final coordinates after alignment should closely match the reference molecule
    expected_final_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    # Check if the final coordinates are close to the expected coordinates
    assert np.allclose(final_coords, expected_final_coords, atol=1e-2), \
        f"Expected {expected_final_coords}, but got {final_coords}"

    # Check if the combined order correctly reordered the elements
    expected_order = np.array([0, 1, 2])  # Since they should be in the same order
    assert np.array_equal(combined_order, expected_order), \
        f"Expected order {expected_order}, but got {combined_order}"

    print("Test passed!")

def test_reorder_after(cc_alpha, cc_beta):
    # Create Molecule instances
    target_molecule = cc_alpha
    reference_molecule = cc_beta

    # Call the reorder_after method
    final_coords, combined_order = target_molecule.reorder_after(reference_molecule)

    # Expected final coordinates after alignment should closely match the reference molecule
    expected_final_coords = reference_molecule.coordinates

    # Check if the final coordinates are close to the expected coordinates
    assert np.allclose(final_coords, expected_final_coords, atol=1e-0), \
        f"Difference of coordinates {expected_final_coords-final_coords}"

    expected_order = reference_molecule.symbols  # Since they should be in the same order
    assert np.array_equal(target_molecule.symbols[combined_order], expected_order), \
        f"Expected order {expected_order}, but got {combined_order}"

def create_rotation_matrix(axis, theta):
    """
    Create a rotation matrix for rotating 'theta' radians around the given axis.

    Parameters:
    - axis (str): Axis to rotate around ('x', 'y', or 'z').
    - theta (float): Rotation angle in radians.

    Returns:
    - R (np.ndarray): 3x3 rotation matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    if axis.lower() == 'x':
        R = np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ])
    elif axis.lower() == 'y':
        R = np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'z':
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    return R

def test_icp_alignment_asymmetric():
    """
    Test the ICP method of the Molecule class by aligning a source molecule
    to a target molecule that has been rotated and translated by known amounts.
    Uses an asymmetric molecule to ensure unique point correspondences.
    """

    # Define atomic symbols (unique to ensure clear correspondences)
    symbols = np.array(['C', 'H', 'O', 'N', 'Cl'])

    # Define source coordinates (asymmetric molecule)
    source_coordinates = np.array([
        [0.0, 0.0, 0.0],      # Carbon
        [1.0, 0.0, 0.0],      # Hydrogen
        [0.0, 1.0, 0.0],      # Oxygen
        [0.0, 0.0, 1.0],      # Nitrogen
        [1.0, 1.0, 1.0]       # Chlorine
    ])

    # Create source and target Molecule instances
    source_mol = Molecule(symbols, source_coordinates.copy())
    target_mol = Molecule(symbols, source_coordinates.copy())

    # Define known rotation (e.g., 30 degrees around the z-axis)
    theta = np.radians(30)
    R_known = create_rotation_matrix('z', theta)

    # Define known translation
    t_known = np.array([1.0, 2.0, 3.0])

    # Apply rotation and translation to create target coordinates
    target_coordinates = (R_known @ target_mol.coordinates.T).T + t_known
    target_mol.coordinates = target_coordinates

    # Perform ICP to align source_mol to target_mol
    source_mol.coordinates, final_R, final_t, mse_history = source_mol.icp(target_mol, max_iterations=100, tolerance=1e-8)

    # Assertions to verify the correctness of ICP

    # 1. Check if the recovered rotation matrix is close to the known rotation
    assert np.allclose(final_R, R_known, atol=1e-6), "Recovered rotation matrix is not close to the known rotation."

    # 2. Check if the recovered translation vector is close to the known translation
    assert np.allclose(final_t, t_known, atol=1e-6), "Recovered translation vector is not close to the known translation."

    # 3. Check if the final mean squared error is below the tolerance
    assert mse_history[-1] < 1e-10, "Final mean squared error is not below the expected threshold."

    # 4. Check if the aligned source coordinates match the target coordinates
    # Due to numerical precision, use a tolerance
    assert np.allclose(source_mol.coordinates, target_mol.coordinates, atol=1e-6), "Aligned source coordinates do not match target coordinates."

def test_icp_no_transformation():
    """
    Test the ICP method when no transformation is applied.
    The source and target molecules are identical.
    """
    symbols = np.array(['C', 'H', 'O', 'N', 'Cl'])
    coordinates = np.array([
        [0.0, 0.0, 0.0],  # Carbon
        [1.0, 0.0, 0.0],  # Hydrogen
        [0.0, 1.0, 0.0],  # Oxygen
        [0.0, 0.0, 1.0],  # Nitrogen
        [1.0, 1.0, 1.0]   # Chlorine
    ])

    source_mol = Molecule(symbols, coordinates.copy())
    target_mol = Molecule(symbols, coordinates.copy())

    # Perform ICP
    source_mol.coordinates, final_R, final_t, mse_history = source_mol.icp(target_mol, max_iterations=50, tolerance=1e-8)

    # Assertions
    assert np.allclose(final_R, np.eye(3), atol=1e-6), "Recovered rotation should be identity."
    assert np.allclose(final_t, np.zeros(3), atol=1e-6), "Recovered translation should be zero."
    assert mse_history[-1] < 1e-10, "Final mean squared error should be near zero."
    assert np.allclose(source_mol.coordinates, target_mol.coordinates, atol=1e-6), "Coordinates should remain unchanged."

if __name__ == "__main__":
    pytest.main()

