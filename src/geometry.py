import numpy as np
import networkx as nx
from src.config import atomic_masses

class GeometryMixin:

    def recenter(self):
        """
        Returns centered molecules coordinates so that its center of mass is at the origin.
        """
        com = self.center_of_mass()
        return self.coordinates - com

