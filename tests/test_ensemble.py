import pytest
from ensemble import Ensemble
from molecule import Molecule, Atom

def test_ensemble_initialization():
    ensemble = Ensemble()
    assert len(ensemble.molecules) == 0

def test_ensemble_add_molecule():
    ensemble = Ensemble()
    mol = Molecule(identifier='Mol1')
    ensemble.add_molecule(mol)
    assert len(ensemble.molecules) == 1
    assert ensemble.molecules[0] == mol

def test_ensemble_remove_duplicates():
    mol1 = Molecule(identifier='Mol1')
    mol2 = Molecule(identifier='Mol1')  # Duplicate identifier
    mol3 = Molecule(identifier='Mol3')
    ensemble = Ensemble.from_list([mol1, mol2, mol3])
    ensemble.remove_duplicates()
    assert len(ensemble.molecules) == 2

def test_ensemble_boltzmann_average():
    mol1 = Molecule(identifier='Mol1', gibbs_free_energy=-80.0, properties={'mass': 18.0})
    mol2 = Molecule(identifier='Mol2', gibbs_free_energy=-70.0, properties={'mass': 20.0})
    ensemble = Ensemble.from_list([mol1, mol2])
    avg_mass = ensemble.boltzmann_average('mass')
    expected_mass = pytest.approx(18.0, abs=0.1)  # Since mol1 has lower energy
    assert avg_mass == expected_mass

