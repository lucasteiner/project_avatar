import pytest
import numpy as np
import os
import pandas as pd

from src.molecule import Molecule  # Assuming Molecule class is defined and imported

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data/', )

data = pd.read_json(data_path+"data.json")
data = data.set_index('RootFile')

g_thr_string = 'Chemical Potential (sign inverted)'

@pytest.fixture
def smi2_ace_thf4_for_liquids():
    """Molecule from Turbomole calculation, project SmRevival
    """
    mol_path = './test_data/smi2_ace_thf4/control'
    molecule = Molecule(
        data['Elements'][mol_path],
        data['xyz Coordinates'][mol_path],
        electronic_energy=data['Single Point Energy'][mol_path], 
        solvation_enthalpy=None,
        thermal_corrections=data['Chemical Potential for Liquids'][mol_path] + data['qRRHO'][mol_path], 
        frequencies=data['Frequencies'][mol_path],
        dipole=data['Dipole'][mol_path]['total'],
        volume_correction=True
        )
    return molecule

@pytest.fixture
def paf2():
    """Molecule from Orca calculation, project pi-stacking
    """
    mol_path = './test_data/1-paf2/orca-freq.out'
    molecule = Molecule(
        data['Elements'][mol_path],
        data['xyz Coordinates'][mol_path],
        electronic_energy=data['Single Point Energy'][mol_path], 
        solvation_enthalpy=None,
        thermal_corrections=data[g_thr_string][mol_path], 
        frequencies=data['Frequencies'][mol_path],
        )
    return molecule

@pytest.fixture
def smi2_ace_thf4():
    """Molecule from Turbomole calculation, project SmRevival
    """
    mol_path = './test_data/smi2_ace_thf4/control'
    molecule = Molecule(
        data['Elements'][mol_path],
        data['xyz Coordinates'][mol_path],
        electronic_energy=data['Single Point Energy'][mol_path], 
        solvation_enthalpy=None,
        thermal_corrections=data[g_thr_string][mol_path], 
        frequencies=data['Frequencies'][mol_path],
        dipole=data['Dipole'][mol_path]['total'],
        )
    return molecule

@pytest.fixture
def li_atom():
    """Molecule from Turbomole calculation
    Data saved in qcdc_testdata_turbomole_sm was collected with qcdc
    """
    mol_path = './test_data/li_atom/control'
    molecule = Molecule(
        data['Elements'][mol_path],
        data['xyz Coordinates'][mol_path],
        electronic_energy=data['Single Point Energy'][mol_path], 
        solvation_enthalpy=None,
        thermal_corrections=data[g_thr_string][mol_path], 
        frequencies=data['Frequencies'][mol_path],
        dipole=data['Dipole'][mol_path]['total'],
        )
    return molecule

@pytest.fixture
def linear_lih():
    """Molecule from Turbomole calculation
    Data saved in qcdc_testdata_turbomole_sm was collected with qcdc
    """
    mol_path = './test_data/lih/control'
    molecule = Molecule(
        data['Elements'][mol_path],
        data['xyz Coordinates'][mol_path],
        electronic_energy=data['Single Point Energy'][mol_path], 
        solvation_enthalpy=None,
        thermal_corrections=data[g_thr_string][mol_path] - data['qRRHO'][mol_path], 
        frequencies=data['Frequencies'][mol_path],
        dipole=data['Dipole'][mol_path]['total'],
        qRRHO_bool=False,
        )
    return molecule

@pytest.fixture
def water1():
    # Create a simple test molecule (for example, water molecule H2O)
    symbols = np.array(['O', 'H', 'H'])
    coordinates = np.array([
        [0.0, 0.0, 0.0],  # Oxygen
        [1.0, 0.0, 0.0],  # Hydrogen 1
        [0.0, 1.0, 0.0]   # Hydrogen 2
    ])
    energy = -75.0
    #moments_of_inertia = np.array([1.0, 2.0, 3.0])
    return Molecule(symbols, coordinates, energy)

@pytest.fixture
def water2():
    # Create another test molecule (slightly different water molecule H2O)
    symbols = np.array(['O', 'H', 'H'])
    coordinates = np.array([
        [0.0, 0.0, 0.0],  # Oxygen
        [1.0, 0.1, 0.0],  # Hydrogen 1
        [0.0, 1.0, 0.1]   # Hydrogen 2
    ])
    energy = -74.9
    #moments_of_inertia = np.array([1.01, 2.01, 3.01])
    return Molecule(symbols, coordinates, energy)

@pytest.fixture
def cc_alpha():
    return Molecule.from_xyz(data_path + 'cc-alpha.xyz')

@pytest.fixture
def cc_beta():
    return Molecule.from_xyz(data_path + 'cc-beta.xyz')

