import numpy as np
import networkx as nx
from src.config import atomic_masses

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

