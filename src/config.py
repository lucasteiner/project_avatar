import yaml
import os

def load_covalent_radii(filename='covalent_radii.yaml'):
    # Get the current directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the YAML file
    file_path = os.path.join(base_dir, '../config/', filename)
    
    with open(file_path, 'r') as file:
        covalent_radii = yaml.safe_load(file)
    
    return covalent_radii

def load_atomic_masses(filename='atomic_masses.yaml'):
    # Get the current directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the YAML file
    file_path = os.path.join(base_dir, '../config/', filename)
    
    with open(file_path, 'r') as file:
        atomic_masses = yaml.safe_load(file)
    
    return atomic_masses

atomic_masses = load_atomic_masses()
covalent_radii = load_covalent_radii()
