import yaml
import os


# Get the current directory of this script



def load_yaml(filename, base_dir=os.path.dirname(os.path.abspath(__file__)), configdir='../config/'):
    """
    loads yaml data from yaml files

    Returns content: dictionary with yaml variables
    """

    # Construct the full path to the YAML file
    file_path = os.path.join(base_dir, configdir, filename)

    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
    
    return content


#def load_covalent_radii(filename='covalent_radii.yaml'):
#    # Get the current directory of this script
#    base_dir = os.path.dirname(os.path.abspath(__file__))
#    
#    # Construct the full path to the YAML file
#    file_path = os.path.join(base_dir, configdir, filename)
#    
#    with open(file_path, 'r') as file:
#        covalent_radii = yaml.safe_load(file)
#    
#    return covalent_radii
#
#def load_atomic_masses(filename='atomic_masses.yaml'):
#    # Get the current directory of this script
#    base_dir = os.path.dirname(os.path.abspath(__file__))
#    
#    # Construct the full path to the YAML file
#    file_path = os.path.join(base_dir, configdir, filename)
#    
#    with open(file_path, 'r') as file:
#        atomic_masses = yaml.safe_load(file)
#    
#    return atomic_masses
#
#def load_constants(filename='constants.yaml'):
#    # Get the current directory of this script
#    base_dir = os.path.dirname(os.path.abspath(__file__))
#    
#    # Construct the full path to the YAML file
#    file_path = os.path.join(base_dir, configdir, filename)
#    
#    with open(file_path, 'r') as file:
#        atomic_masses = yaml.safe_load(file)
#    
#    return atomic_masses


covalent_radii = load_yaml('covalent_radii.yaml')
atomic_masses = load_yaml('atomic_masses.yaml')
my_const = load_yaml('constants.yaml')
config = load_yaml('config.yaml')


# Overwrite constants with user definitions in calculation directory
if os.path.exists('config.yaml'):
    config = load_yaml('config.yaml', base_dir='', configdir='')
    for key, value in config.items():
        config[key] = value
else:
    pass
    #print(f"No config.yaml found, using defaults")

## Import the variables into the current namespace
#for key, value in const.items():
#    globals()[key] = value

# Optional: Print variables for verification
if __name__ == '__main__':
    print("Variables imported from YAML:")
    for key, value in atomic_masses.items():
        print(f"{key}: {value}")


