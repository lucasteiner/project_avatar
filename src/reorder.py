import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

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
        Perform the Kabsch algorithm to create new coordinates of this molecule aligned after another molecule.
    
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
        Create a reordered set of atoms using the Hungarian algorithm to match another molecule.
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
        Reorders atoms of the molecule by their distance from the centroid.
    
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
        Reorder the symbols and coordinates of the molecule.

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

    def get_transformed_coordinates(self, rotation, translation):
        """
        Get a new set of coordinates after applying rotation and translation,
        without modifying the original coordinates.

        Parameters:
        - rotation (np.ndarray): 3x3 rotation matrix.
        - translation (np.ndarray): Translation vector of shape (3,).

        Returns:
        - new_coordinates (np.ndarray): Transformed coordinates.
        """
        return (rotation @ self.coordinates.T).T + translation
    
    #def apply_transformation(self, rotation, translation):
    #    """
    #    Apply rotation and translation to the molecule's coordinates.

    #    Parameters:
    #    - rotation (np.ndarray): 3x3 rotation matrix.
    #    - translation (np.ndarray): Translation vector of shape (3,).
    #    """
    #    self.coordinates = (rotation @ self.coordinates.T).T + translation

    def icp(self, reference, max_iterations=100, tolerance=1e-5):
        """
        Perform the Iterative Closest Point algorithm to create aligned coordinates of this molecule to the reference molecule.

        Parameters:
        - reference (Molecule): The reference molecule to align to.
        - max_iterations (int): Maximum number of ICP iterations.
        - tolerance (float): Convergence threshold based on change in error.

        Returns:
        - final_rotation (np.ndarray): The cumulative rotation matrix.
        - final_translation (np.ndarray): The cumulative translation vector.
        - errors (list): List of mean squared errors at each iteration.
        """
        # Make copies to avoid modifying the original molecules
        source_coords = self.coordinates.copy()
        reference_coords = reference.coordinates.copy()

        # Initialize cumulative rotation and translation
        final_rotation = np.eye(3)
        final_translation = np.zeros(3)

        errors = []

        for i in range(max_iterations):
            # Step 1: Find the closest points in reference for each source point, considering symbols
            closest_reference = self._find_closest_points(source_coords, reference_coords, reference.symbols)

            # Step 2: Compute the optimal rotation and translation
            R, t = self._compute_optimal_transform(source_coords, closest_reference)

            # Step 3: Apply the transformation
            source_coords = (R @ source_coords.T).T + t

            # Update cumulative transformation
            final_rotation = R @ final_rotation
            final_translation = R @ final_translation + t

            # Step 4: Compute mean squared error
            mse = np.mean(np.linalg.norm(source_coords - closest_reference, axis=1) ** 2)
            errors.append(mse)

            # Check for convergence
            if i > 0 and abs(errors[-2] - errors[-1]) < tolerance:
                print(f"ICP converged after {i+1} iterations.")
                break
        else:
            print(f"ICP did not converge after {max_iterations} iterations.")

        # Apply the final transformation to the molecule
        final_coords = self.get_transformed_coordinates(final_rotation, final_translation)

        return final_coords, final_rotation, final_translation, errors


    def _find_closest_points(self, source, reference, reference_symbols):
        """
        Find the closest points in the reference for each point in the source, considering atomic symbols.

        Parameters:
        - source (np.ndarray): Source coordinates of shape (N, 3).
        - reference (np.ndarray): Target coordinates of shape (M, 3).
        - reference_symbols (np.ndarray): Array of atomic symbols in the reference molecule.

        Returns:
        - closest (np.ndarray): Closest reference coordinates for each source point.
        """
        closest = np.zeros_like(source)
        used_reference_indices = set()

        for i, (symbol, coord) in enumerate(zip(self.symbols, source)):
            # Find all reference indices with the same symbol
            possible_indices = np.where(reference_symbols == symbol)[0]

            # Exclude already matched reference indices
            available_indices = [idx for idx in possible_indices if idx not in used_reference_indices]

            if not available_indices:
                raise ValueError(f"No available reference atoms for symbol '{symbol}'.")

            # Compute distances to available reference atoms
            distances = np.linalg.norm(reference[available_indices] - coord, axis=1)

            # Find the closest available reference atom
            min_idx = np.argmin(distances)
            closest_reference_idx = available_indices[min_idx]
            closest[i] = reference[closest_reference_idx]
            used_reference_indices.add(closest_reference_idx)

        return closest

    def _compute_optimal_transform(self, source, reference):
        """
        Compute the optimal rotation and translation to align source to reference.

        Parameters:
        - source (np.ndarray): Source coordinates of shape (N, 3).
        - reference (np.ndarray): Target coordinates of shape (N, 3).

        Returns:
        - R (np.ndarray): Optimal rotation matrix.
        - t (np.ndarray): Optimal translation vector.
        """
        # Compute centroids
        centroid_source = np.mean(source, axis=0)
        centroid_reference = np.mean(reference, axis=0)

        # Center the points
        source_centered = source - centroid_source
        reference_centered = reference - centroid_reference

        # Compute covariance matrix
        H = source_centered.T @ reference_centered

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = centroid_reference - R @ centroid_source

        return R, t

    def reorder_after(self, reference_molecule):
        """
        Create reordered atoms of this molecule by following these steps:
        1. Create copies of both molecules.
        2. Reorder by centroid for both the molecule and the reference molecule.
        3. Align using the Kabsch algorithm.
        4. Reorder by the Hungarian algorithm.
        5. Align again using the Kabsch algorithm.
        6. Align again using the ICP algorithm.
        7. Combine the sorting orders
    
        Parameters:
        reference_molecule (Molecule): The reference molecule to reorder after.
    
        Returns:
        tuple: The final reordered coordinates and the final sort order.
        """
        # Step 1: Create copies of both molecules
        source_copy = self.copy()
        reference_copy = reference_molecule.copy()
    
        # Step 2: Reorder by centroid (on both the source and reference)
        source_symbols, source_coords, source_centroid_order = source_copy.reorder_by_centroid()
        reference_symbols, reference_coords, reference_centroid_order = reference_copy.reorder_by_centroid()
    
        # Apply the centroid sorting to both molecules
        source_copy.symbols = source_symbols
        source_copy.coordinates = source_coords
        reference_copy.symbols = reference_symbols
        reference_copy.coordinates = reference_coords
    
        # Step 3: Perform Kabsch alignment
        aligned_coords, _ = source_copy.kabsch_align(reference_copy)
    
        # Update the source coordinates with the aligned ones
        source_copy.coordinates = aligned_coords
    
        # Step 4: Reorder using Hungarian algorithm
        reordered_symbols_hungarian, reordered_coords_hungarian, hungarian_order = source_copy.hungarian_reorder(reference_copy)
    
        # Apply the Hungarian reordering
        source_copy.symbols = reordered_symbols_hungarian
        source_copy.coordinates = reordered_coords_hungarian
    
        # Step 5: Perform another Kabsch alignment
        final_aligned_coords, _ = source_copy.kabsch_align(reference_copy)

        # Step 6: Perform an ICP alignment
        final_aligned_coords, _, _, error = source_copy.icp(reference_copy)
    
        # Update the final coordinates
        source_copy.coordinates = final_aligned_coords
    
        # Step 7: Combine the sorting orders
        # First, invert the reference centroid order
        inverted_reference_centroid_order = source_copy.invert_positions(reference_centroid_order)
    
        # Combine the centroid and Hungarian sorting orders
        combined_order = source_centroid_order[hungarian_order][inverted_reference_centroid_order]
        # combined_order = sort_indices[hungarian_indices[invert_positions(ref_sort_indices)]]
        source_copy.coordinates = self.coordinates[combined_order]
        source_copy.symbols = self.symbols[combined_order]
        source_copy.coordinates,  _ = source_copy.kabsch_align(reference_molecule)
        source_copy.coordinates,  _, _, _ = source_copy.icp(reference_molecule) # optimizes rmsd
    
        final_symbols = source_copy.symbols
        final_coordinates = source_copy.coordinates
        #return source_copy.coordinates, combined_order
        return final_coordinates, final_symbols, combined_order
    
    def is_duplicate(self, other):
        """Compares two molecules and returns true if they are duplicates, False otherwise

        Parameters:
            other (molecule): Molecule for comparison

        Returns:
        bool: True, if molecules are duplicates
        """
        try:
            coord, symbols, _ = self.reorder_after(other)
            reference = Molecule(symbols, coord, energy=other.energy) # reorder to get isomers and align coordinates, too
            bool = np.all(self.compare_molecule(self, reference))
        except ValueError:
            bool = False
        return bool
