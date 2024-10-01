import pytest
import numpy as np
from molecule import Molecule, Atom

def test_molecule_initialization():
    # Create some atoms
    atom1 = Atom(position=[0.0, 0.0, 0.0])
    atom2 = Atom(position=[1.0, 0.0, 0.0])
    atom3 = Atom(position=[0.0, 1.0, 0.0])
    
    # Initialize a Molecule
    mol = Molecule(atoms=[atom1, atom2, atom3], identifier='TestMolecule')
    
    # Assertions
    assert mol.identifier == 'TestMolecule'
    assert len(mol.atoms) == 3
    assert mol.atoms[0].position.tolist() == [0.0, 0.0, 0.0]

def test_molecule_dimension():
    # Linear molecule along x-axis
    atom1 = Atom(position=[0.0, 0.0, 0.0])
    atom2 = Atom(position=[1.0, 0.0, 0.0])
    atom3 = Atom(position=[2.0, 0.0, 0.0])
    mol = Molecule(atoms=[atom1, atom2, atom3])
    assert mol.dimension == 2  # Linear

    # Non-linear molecule
    atom3.position = np.array([1.0, 1.0, 0.0])
    mol = Molecule(atoms=[atom1, atom2, atom3])
    assert mol.dimension == 3  # Non-linear

def test_molecule_compute_rmsd():
    # Two identical molecules
    coords = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])
    mol1 = Molecule(identifier='Mol1', coordinates=coords)
    mol2 = Molecule(identifier='Mol2', coordinates=coords)
    rmsd = mol1.compute_rmsd(mol2)
    assert rmsd == 0.0

    # Slightly different molecules
    mol2.coordinates += 1e-3
    rmsd = mol1.compute_rmsd(mol2)
    assert rmsd == pytest.approx(1e-3, rel=1e-2)

