import numpy as np
import networkx as nx
from atomic_masses import atomic_masses
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class GeometryMixin:

    def distance(self, atom_index1, atom_index2):
        """
        Calculate the distance between two atoms.

        Parameters:
        atom_index1 (int): Index of the first atom.
        atom_index2 (int): Index of the second atom.

        Returns:
        float: The Euclidean distance between the two atoms.
        """
        if atom_index1 >= len(self.symbols) or atom_index2 >= len(self.symbols):
            raise IndexError("Atom index out of range.")
        coord1 = self.coordinates[atom_index1]
        coord2 = self.coordinates[atom_index2]
        return np.linalg.norm(coord1 - coord2)

    def bond_angle(self, atom_index1, atom_index2, atom_index3):
        """
        Calculate the bond angle formed by three atoms.

        Parameters:
        atom_index1 (int): Index of the first atom.
        atom_index2 (int): Index of the central atom.
        atom_index3 (int): Index of the third atom.

        Returns:
        float: The bond angle in degrees.
        """
        if any(index >= len(self.symbols) for index in [atom_index1, atom_index2, atom_index3]):
            raise IndexError("Atom index out of range.")
        # Vectors from central atom to the two other atoms
        vec1 = self.coordinates[atom_index1] - self.coordinates[atom_index2]
        vec2 = self.coordinates[atom_index3] - self.coordinates[atom_index2]
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        # Compute angle
        cos_theta = np.dot(vec1_norm, vec2_norm)
        # Clamp cos_theta to [-1, 1] to avoid numerical errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        return angle

    def dihedral_angle(self, atom_index1, atom_index2, atom_index3, atom_index4):
        """
        Calculate the dihedral angle defined by four atoms.

        Parameters:
        atom_index1 (int): Index of the first atom.
        atom_index2 (int): Index of the second atom.
        atom_index3 (int): Index of the third atom.
        atom_index4 (int): Index of the fourth atom.

        Returns:
        float: The dihedral angle in degrees.
        """
        if any(index >= len(self.symbols) for index in [atom_index1, atom_index2, atom_index3, atom_index4]):
            raise IndexError("Atom index out of range.")

        p0 = self.coordinates[atom_index1]
        p1 = self.coordinates[atom_index2]
        p2 = self.coordinates[atom_index3]
        p3 = self.coordinates[atom_index4]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # Normalize b1 so that it does not influence magnitude of vector products
        b1 /= np.linalg.norm(b1)

        # Compute vectors normal to the planes
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)

        angle = np.degrees(np.arctan2(y, x))
        return angle

    def recenter(self):
        """
        Returns centered molecules coordinates so that its center of mass is at the origin.
        """
        com = self.center_of_mass()
        return self.coordinates - com

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

    def is_linear(self, tolerance=1e-3):
        """
        Determine if the molecule is linear within a specified tolerance.

        Parameters:
        tolerance (float): The threshold below which a moment of inertia is considered zero.

        Returns:
        bool: True if the molecule is linear, False otherwise.
        """
        moments = self.moments_of_inertia()
        # Sort the moments to ensure consistent order
        moments = np.sort(moments)
        # For a linear molecule, two moments should be approximately zero
        zero_moments = moments < tolerance
        if np.sum(zero_moments) >= 1:
            return True
        else:
            return False

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
