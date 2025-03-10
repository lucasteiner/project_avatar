import pytest
import numpy as np
from molecule import Molecule  # Assuming Molecule class is defined and imported

def test_compare_moments_of_inertia(water1, water2):
    # Test comparing moments of inertia
    result = water1.compare_moments_of_inertia(water2, tolerance=0.05)
    assert result is True, "Moments of inertia should be considered matching within tolerance."

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

