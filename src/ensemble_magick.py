import numpy as np
import pandas as pd

def boltzmann_average_and_entropy(gibbs_energies, temperature=298):
    """
    Calculate the Boltzmann average of Gibbs free energies, conformational entropy,
    and entropy-corrected average Gibbs free energy.

    Parameters:
    gibbs_energies (pd.Series or list of float): Gibbs free energies of the conformers (in kJ/mol).
    temperature (float): Temperature in Kelvin (default is 298 K).

    Returns:
    float: Boltzmann average of Gibbs free energies (in kJ/mol).
    float: Conformational entropy (in kJ/mol-K).
    float: Entropy-corrected average Gibbs free energy (in kJ/mol).
    """
    R = 8.3145  # Gas constant in J/(mol*K)
    R_kJ = R / 1000  # Convert gas constant to kJ/(mol*K)

    # Ensure gibbs_energies is a numpy array for calculations
    gibbs_energies = np.array(gibbs_energies)

    # Convert Gibbs free energies to dimensionless form using kT (using kJ/mol units)
    dimensionless_energies = (gibbs_energies - gibbs_energies.min()) / (R_kJ * temperature)

    # Calculate Boltzmann factors
    boltzmann_factors = np.exp(-dimensionless_energies)

    # Calculate partition function Z
    partition_function = np.sum(boltzmann_factors)

    # Calculate probabilities (Boltzmann distribution)
    probabilities = boltzmann_factors / partition_function

    # Calculate Boltzmann average of Gibbs free energies
    boltzmann_average_gibbs = np.sum(probabilities * gibbs_energies)

    # Calculate conformational entropy
    conformational_entropy = -R_kJ * np.sum(probabilities * np.log(probabilities))

    # Calculate entropy-corrected average Gibbs free energy
    entropy_corrected_average = boltzmann_average_gibbs - temperature * conformational_entropy

    return boltzmann_average_gibbs, conformational_entropy, entropy_corrected_average

# Example usage
# Creating a pandas DataFrame with Gibbs free energies
#data = {'Gibbs_Free_Energy_kJ_per_mol': [50.0, 51.0, 49.5]}  # Energies in kJ/mol
#df = pd.DataFrame(data)

# Using the function with a DataFrame column
#gibbs_energies_column = df['Gibbs_Free_Energy_kJ_per_mol']
#boltzmann_average, entropy, entropy_corrected = boltzmann_average_and_entropy(gibbs_energies_column)
#print(f"Boltzmann Average Gibbs Free Energy: {boltzmann_average} kJ/mol")
#print(f"Conformational Entropy: {entropy} kJ/mol-K")
#print(f"Entropy-Corrected Average Gibbs Free Energy: {entropy_corrected} kJ/mol")
