class ComparisonMixin:
   
    #def compare_moments_of_inertia(self, other, tolerance=1):
    #    """
    #    Compare the moments of inertia of two molecules within a given tolerance.
    # 
    #    Parameters:
    #    other (Molecule): The other molecule to compare with.
    #    tolerance (float): The precision tolerance for comparison (default: 1).
    #
    #    Returns:
    #    bool: True if the moments of inertia are within the given tolerance, False otherwise.
    #    """
    #    return np.allclose(self.moments_of_inertia(), other.moments_of_inertia(), atol=tolerance)

   

#def compare_molecules(mol1: Molecule, mol2: Molecule, energy_threshold=1e-5, rmsd_threshold=1e-1, rotational_constant_threshold=1e-0) -> bool:
#    """
#    Compare two molecules based on electronic energy, RMSD, and rotational constants.
#    Returns True if they are considered duplicates.
#    """
#    # Compare electronic energy
#    if mol1.electronic_energy is not None and mol2.electronic_energy is not None:
#        energy_diff = abs(mol1.electronic_energy - mol2.electronic_energy)
#    else:
#        energy_diff = np.inf  # Cannot compare
#
#    # Compare rotational constants
#    if mol1.rotational_constants is not None and mol2.rotational_constants is not None:
#        rot_const_diff = np.linalg.norm(np.array(mol1.rotational_constants) - np.array(mol2.rotational_constants))
#    else:
#        rot_const_diff = np.inf  # Cannot compare
#
#    # Compute RMSD
#    try:
#        rmsd_value = compute_rmsd(mol1, mol2)
#    except ValueError:
#        rmsd_value = np.inf  # Cannot compute RMSD
#
#    # Check if all differences are below thresholds
#    is_duplicate = (
#        energy_diff <= energy_threshold and
#        rot_const_diff <= rotational_constant_threshold and
#        rmsd_value <= rmsd_threshold
#    )
#    return is_duplicate
