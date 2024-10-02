import pytest
import numpy as np
from bonding import Bonding
from molecule import Molecule
import os

def test_atom_mapping():
    #symbols1 = np.array(['C', 'H', 'H', 'H', 'H'])
    symbols1 = np.array(['C', 'Br', 'H', 'F', 'Cl'])
    coordinates1 = np.array([
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0890],
        [1.0267, 0.0000, -0.3630],
        [-0.5133, -0.8892, -0.3630],
        [-0.5133, 0.8892, -0.3630]
    ])

    #symbols2 = np.array(['C', 'H', 'H', 'H', 'H'])
    symbols2 = np.array(['C', 'H', 'Br', 'Cl', 'F'])
    coordinates2 = np.array([
        [0.0000, 0.0000, 0.0000],
        [1.0267, 0.0000, -0.3630],
        [0.0000, 0.0000, 1.0890],
        [-0.5133, 0.8892, -0.3630],
        [-0.5133, -0.8892, -0.3630]
    ])

    # Create Bonding instances
    mol1 = Bonding(symbols1, coordinates1)
    mol2 = Bonding(symbols2, coordinates2)

    # Get atom mapping between the two molecules
    mapping = Bonding.get_atom_mapping(mol1, mol2)

    # Expected mapping (indices have changed)
    expected_mapping = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}

    assert mapping == expected_mapping, f"Expected mapping {expected_mapping}, got {mapping}"

    print("Test passed. Atom mapping:")
    for atom_index_mol1, atom_index_mol2 in mapping.items():
        print(f"Atom {atom_index_mol1} in mol1 maps to Atom {atom_index_mol2} in mol2")


def test_bonding():
    """
    Test the bonding class.
    """
    # Example molecule data (symbols and coordinates)
    symbols1 = np.array(['C', 'Br', 'H', 'F', 'Cl'])
    coordinates1 = np.array([
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0890],
        [1.0267, 0.0000, -0.3630],
        [-0.5133, -0.8892, -0.3630],
        [-0.5133, 0.8892, -0.3630]
    ])
    
    symbols2 = np.array(['C', 'H', 'Br', 'Cl', 'F'])
    coordinates2 = np.array([
        [0.0000, 0.0000, 0.0000],
        [1.0267, 0.0000, -0.3630],
        [0.0000, 0.0000, 1.0890],
        [-0.5133, 0.8892, -0.3630],
        [-0.5133, -0.8892, -0.3630]
    ])
    
    # Create Bonding instances
    mol1 = Bonding(symbols1, coordinates1)
    mol2 = Bonding(symbols2, coordinates2)
    
    # Get atom mapping between the two molecules
    mapping = Bonding.get_atom_mapping(mol1, mol2)
    expected_keys = [0, 2, 1, 4, 3]
    reordered_coordinates = Bonding.reorder_coordinates(mapping, mol2.coordinates)
    assert np.all(reordered_coordinates == mol1.coordinates)
    assert np.all(list(mapping.keys()) == expected_keys)

def test_molecule_bonding():
    """
    Test the bonding class.
    """
    # Example molecule data (symbols and coordinates)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xyz_path = os.path.join(base_dir, 'xyz_structures/', )
    mol1 = Molecule.from_xyz(xyz_path+'fcl-toluene.xyz')
    mol2 = Molecule.from_xyz(xyz_path+'fcl-toluene-reordered3.xyz')
    
    
    # Get atom mapping between the two molecules
    mapping = Bonding.get_atom_mapping(mol1.bonding, mol2.bonding)
    #expected_keys = [1, 0, 2, 3, 4, 9, 8, 7, 6, 11, 5, 10, 12, 13, 14]
    #expected_values = [0, 1, 3, 2, 4, 5, 6, 7, 8, 10, 11, 9, 12, 13, 14]
    #print(expected_values)
    #print('values',list(mapping.values()))
    #print('keys',list(mapping.keys()))
    #mol1.reorder_atoms(list(mapping.values()))
    #mol2.reorder_atoms(list(mapping.keys()))
    #print(mol1.coordinates)
    #print(mol2.coordinates)
    reordered_coordinates = Bonding.reorder_coordinates(mapping, mol1.coordinates)
    assert np.all(reordered_coordinates == mol2.coordinates)
    #assert np.all(list(mapping.keys()) == expected_keys)


