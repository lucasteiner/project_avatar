import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class ReorderMixin:
   
    def compare_moments_of_inertia(self, other, tolerance=1.0):
        """
        Compare the moments of inertia of two molecules within a given tolerance.
    
        Parameters:
        other (Molecule): The other molecule to compare.
        tolerance (float): The allowed difference in moments of inertia.
    
        Returns:
        bool: True if moments of inertia are within the given tolerance, False otherwise.
        """
        moments_self = self.moments_of_inertia()
        moments_other = other.moments_of_inertia()
    
        return np.allclose(moments_self, moments_other, atol=tolerance)
    
    def compare_energy(self, other, tolerance=1.0):
        """
        Compare the energy of two molecules within a given tolerance.
    
        Parameters:
        other (Molecule): The other molecule to compare.
        tolerance (float): The allowed difference in energy.
    
        Returns:
        bool: True if the energy difference is within the given tolerance, False otherwise.
        """
        return np.isclose(self.energy, other.energy, atol=tolerance)
    
    def is_rmsd_below_threshold(self, other, threshold=0.1):
        """
        Check if the RMSD between two molecules is below a certain threshold.
    
        Parameters:
        other (Molecule): The other molecule to compare.
        threshold (float): The RMSD threshold.
    
        Returns:
        bool: True if the RMSD is below the threshold, False otherwise.
        """
        rmsd_value = self.calculate_rmsd(other)
        return rmsd_value < threshold
    
    def compare_molecule(self, other, tolerance_inertia=1.0, tolerance_energy=1.0, rmsd_threshold=0.1):
        """
        Compare moments of inertia, energy, and RMSD for two molecules.
    
        Parameters:
        other (Molecule): The other molecule to compare.
        tolerance_inertia (float): Tolerance for comparing moments of inertia.
        tolerance_energy (float): Tolerance for comparing energy.
        rmsd_threshold (float): Threshold for RMSD comparison.
    
        Returns:
        tuple: A tuple of bool values (moments_match, energy_match, rmsd_below_threshold).
        """
        moments_match = self.compare_moments_of_inertia(other, tolerance_inertia)
        energy_match = self.compare_energy(other, tolerance_energy)
        rmsd_below_threshold = self.is_rmsd_below_threshold(other, rmsd_threshold)
    
        return moments_match, energy_match, rmsd_below_threshold
    
    def calculate_rmsd(self, other):
        """
        Calculate the RMSD (Root-Mean-Square Deviation) between this molecule and another molecule.

        Parameters:
        other (Molecule): The molecule to calculate RMSD against.

        Returns:
        float: The RMSD value.
        """
        # Ensure the molecules have the same number of atoms
        if len(self.symbols) != len(other.symbols):
            raise ValueError("Both molecules must have the same number of atoms to calculate RMSD.")
        # Ensure the molecules have the same order
        if np.any(self.symbols != other.symbols):
            raise ValueError("Both molecules must have the same order of atoms to calculate RMSD.")

        # Calculate the RMSD
        diff = self.coordinates - other.coordinates
        rmsd_value = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

        return rmsd_value

    def kabsch_align(self, other):
        """
        Perform the Kabsch algorithm to align this molecule to another molecule.
    
        Parameters:
        other (Molecule): The molecule to align to.
    
        Returns:
        np.ndarray: The aligned coordinates of this molecule.
        np.ndarray: The rotation matrix used to align the molecules.
        """
        # Ensure the molecules have the same number of atoms
        if len(self.symbols) != len(other.symbols):
            raise ValueError("Both molecules must have the same number of atoms to perform alignment.")
    
        # Center both sets of coordinates around their center of mass
        coords1 = self.coordinates - self.center_of_mass()
        coords2 = other.coordinates - other.center_of_mass()
    
        # Compute the covariance matrix
        covariance_matrix = np.dot(coords1.T, coords2)
    
        # Perform Singular Value Decomposition (SVD)
        V, S, Wt = np.linalg.svd(covariance_matrix)
    
        # Calculate the rotation matrix
        # Check for reflection and ensure a proper rotation (determinant check)
        d = np.linalg.det(np.dot(Wt.T, V.T))
        if d < 0:
            V[:, -1] *= -1
    
        rotation_matrix = np.dot(Wt.T, V.T)
    
        # Apply the rotation matrix to align the first molecule's coordinates
        aligned_coords = np.dot(coords1, rotation_matrix)
    
        return aligned_coords, rotation_matrix

    def hungarian_reorder(self, other):
        """
        Reorder the atoms in the molecule using the Hungarian algorithm to match another molecule.
        The cost matrix is based on the Euclidean distances between atomic coordinates.
    
        Parameters:
        other (Molecule): The molecule to reorder this molecule to match.
    
        Returns:
        tuple: A tuple containing the reordered symbols, reordered coordinates, and the original indices.
        """
        if not self.is_comparable(other):
            raise ValueError("Molecules are not comparable (different elements or quantities).")
        
        ## Calculate the cost matrix based on Euclidean distances between atoms
        cost_matrix = cdist(self.coordinates, other.coordinates)
        
        # Solve the assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Reorder symbols and coordinates based on the sorted indices
        reordered_symbols = self.symbols[col_ind]
        reordered_coordinates = self.coordinates[col_ind]

        # Return the reordered symbols, reordered coordinates, and the original sorted indices
        return (reordered_symbols, reordered_coordinates, col_ind)

    def reorder_by_centroid(self):
        """
        Reorders atoms in the molecule by their distance from the centroid.
    
        Returns:
        tuple: A tuple containing the reordered symbols, reordered coordinates, and the original indices.
        """
        # Use the molecule's coordinates to calculate the centroid
        centroid = np.mean(self.coordinates, axis=0)
    
        # Calculate distances of each atom from the centroid
        distances = np.linalg.norm(self.coordinates - centroid, axis=1)
    
        # Get the sorted indices based on distance from the centroid
        sorted_indices = np.argsort(distances)
    
        # Reorder symbols and coordinates based on the sorted indices
        reordered_symbols = self.symbols[sorted_indices]
        reordered_coordinates = self.coordinates[sorted_indices]
    
        # Return the reordered symbols, reordered coordinates, and the original sorted indices
        return (reordered_symbols, reordered_coordinates, sorted_indices)

    def reorder_atoms(self, new_order):
        """
        Reorder the symbols and coordinates in the molecule.

        Parameters:
        new_order (nd.array): New order obtained from sorting routines

        Returns:
        tuple: A tuple containing the reordered symbols, reordered coordinates
        """
        if sorted(new_order) != list(range(len(self.symbols))):
            raise ValueError("new_order must be a permutation of indices 0 to N-1.")
        return (self.symbols[new_order], self.coordinates[new_order])

    @staticmethod
    def invert_positions(arr):
        """
        Inverts the positions of an array (i.e., mapping index to its position).
    
        Parameters:
        arr (numpy.ndarray): The array to invert.
    
        Returns:
        numpy.ndarray: The inverted array.
        """
        inverted_array = np.zeros_like(arr, dtype=int)
        inverted_array[arr] = np.arange(len(arr))
        return inverted_array

    def reorder_after(self, reference_molecule):
        """
        Reorder this molecule after another molecule by following these steps:
        1. Create copies of both molecules.
        2. Reorder by centroid for both the molecule and the reference molecule.
        3. Align using the Kabsch algorithm.
        4. Reorder by the Hungarian algorithm.
        5. Align again using the Kabsch algorithm.
    
        Parameters:
        reference_molecule (Molecule): The reference molecule to reorder after.
    
        Returns:
        tuple: The final reordered coordinates and the final sort order.
        """
        # Step 1: Create copies of both molecules
        target_copy = self.copy()
        reference_copy = reference_molecule.copy()
    
        # Step 2: Reorder by centroid (on both the target and reference)
        target_symbols, target_coords, target_centroid_order = target_copy.reorder_by_centroid()
        reference_symbols, reference_coords, reference_centroid_order = reference_copy.reorder_by_centroid()
    
        # Apply the centroid sorting to both molecules
        target_copy.symbols = target_symbols
        target_copy.coordinates = target_coords
        reference_copy.symbols = reference_symbols
        reference_copy.coordinates = reference_coords
    
        # Step 3: Perform Kabsch alignment
        aligned_coords, _ = target_copy.kabsch_align(reference_copy)
    
        # Update the target coordinates with the aligned ones
        target_copy.coordinates = aligned_coords
    
        # Step 4: Reorder using Hungarian algorithm
        reordered_symbols_hungarian, reordered_coords_hungarian, hungarian_order = target_copy.hungarian_reorder(reference_copy)
    
        # Apply the Hungarian reordering
        target_copy.symbols = reordered_symbols_hungarian
        target_copy.coordinates = reordered_coords_hungarian
    
        # Step 5: Perform another Kabsch alignment
        final_aligned_coords, _ = target_copy.kabsch_align(reference_copy)
    
        # Update the final coordinates
        target_copy.coordinates = final_aligned_coords
    
        # Step 6: Combine the sorting orders
        # First, invert the reference centroid order
        inverted_reference_centroid_order = target_copy.invert_positions(reference_centroid_order)
    
        # Combine the centroid and Hungarian sorting orders
        combined_order = target_centroid_order[hungarian_order][inverted_reference_centroid_order]
        # combined_order = sort_indices[hungarian_indices[invert_positions(ref_sort_indices)]]
        target_copy.coordinates = self.coordinates[combined_order]
        final_coordinates,  _ = target_copy.kabsch_align(reference_molecule)
    
        #return target_copy.coordinates, combined_order
        return final_coordinates, combined_order
    
