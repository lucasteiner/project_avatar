# Project Avatar

**Quantum Chemical Data Processing Tool**

## Overview

Project Avatar is a quantum chemical data processing program designed to facilitate the analysis and manipulation of molecular data. It works seamlessly with [CREST](https://github.com/grimme-lab/crest), which generates molecular ensembles, and is compatible with the upcoming **Quantum Chemical Data Collector (QCDC)** project.

## Features

- **Duplicate Molecule Identification**: Utilizes a reordering approach to identify duplicates in unsorted molecular datasets, essential for accurate data processing and analysis.
- **Thermodynamic Calculations**: The `ThermoMolecule` class calculates partition functions and Gibbs free energies using the Rigid Rotor Harmonic Oscillator (RRHO) approximation.
- **Molecular Topology Setup**: The `Bonding` subclass establishes the topology of molecules, enabling detailed comparison of different conformers.

## Requirements

- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [NetworkX](https://networkx.org/)
- [PyYAML](https://pyyaml.org/)

## Installation

Clone the repository and install the required packages:
```bash
git clone https://github.com/lucasteiner/project-avatar.git
cd project-avatar
pip install -r requirements.txt
```

## Usage

Project Avatar is designed to work seamlessly with the CREST program and **qcdc**. For more detailed usage instructions and examples, please refer to the [documentation](https://project-avatar.readthedocs.io/en/latest/) or consult the source code directly.

## Documentation

The full documentation for Project Avatar is available on [Read the Docs](https://project-avatar.readthedocs.io/en/latest/). 

## Planned Enhancements

- **Enhanced Molecular Data Handling:** Improvements to streamline the processing of complex molecular data.
- **Advanced Reordering Techniques:** Better detection and handling of duplicate molecules for more efficient data analysis.
- **Expanded Topology Features:** More functionality in the `bonding` subclass for detailed molecular comparisons.

## Contributing

Contributions are welcome! If you'd like to contribute to Project Avatar, please feel free to fork the repository, make changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions, suggestions, or feedback, please reach out via the GitHub repository's [Issues](https://github.com/lucasteiner/project-avatar/issues) page.
Renamed repository
