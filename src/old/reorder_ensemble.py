#!/bin/env python3
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import argparse

def kabsch(P, Q):
    """
    Implements the Kabsch algorithm to find the optimal rotation matrix that minimizes RMSD.

    Parameters:
    P (numpy.ndarray): First set of points.
    Q (numpy.ndarray): Second set of points.

    Returns:
    numpy.ndarray: The optimal rotation matrix.
    """
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
        V[:, -1] = -V[:, -1]
    return np.dot(V, W)

def reorder_by_centroid(atoms):
    """
    Reorders atoms by their distance from the centroid.

    Parameters:
    atoms (list): A list of atoms and their coordinates.

    Returns:
    tuple: A tuple of the reordered atoms and their original indices.
    """
    coords = np.array([atom[1] for atom in atoms])
    centroid = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    sorted_indices = np.argsort(distances)
    return [atoms[i] for i in sorted_indices], sorted_indices


def reorder_atoms_hungarian(reference_atoms, target_atoms):
    """
    Reorders atoms using the Hungarian algorithm based on minimal distance.

    Parameters:
    reference_atoms (list): A list of atoms from the reference structure.
    target_atoms (list): A list of atoms from the target structure.

    Returns:
    tuple: A tuple of reordered target atoms and the order indices.
    """
    ref_coords = np.array([atom[1] for atom in reference_atoms])
    target_coords = np.array([atom[1] for atom in target_atoms])
    cost_matrix = cdist(ref_coords, target_coords)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [target_atoms[i] for i in col_ind], col_ind


def invert_positions(arr):
    """
    Inverts the positions of an array (i.e., mapping index to its position).

    Parameters:
    arr (numpy.ndarray): The array to invert.

    Returns:
    numpy.ndarray: The inverted array.
    """
    inverted_array = np.zeros_like(arr, dtype=int)
    inverted_array[arr] = np.arange(len(arr))
    return inverted_array

def is_duplicate(atoms, ref_atoms):
    """
    Finds the best atom sort order based on RMSD between reference and target molecule.
    This routine does not change the functionality of the script.

    Parameters:
    molecule : The data of the target molecule.
    reference_molecule : The data of the reference molecule.
    data structure of a molecule: [elem, np.array(atom_coords) for all elems]

    Returns:
    tuple: The reordered atoms, the best sort order
    """

    moi = calculate_moments_of_inertia(atoms)
    ref_moi = calculate_moments_of_inertia(ref_atoms)

    #for energy, atoms in ensemble_molecules:

    ref_atoms_sorted, ref_sort_indices = reorder_by_centroid(ref_atoms)
    target_atoms_sorted, sort_indices = reorder_by_centroid(atoms)

    ref_coords_sorted = np.array([atom[1] for atom in ref_atoms_sorted])
    target_coords_sorted = np.array([atom[1] for atom in target_atoms_sorted])
    R = kabsch(ref_coords_sorted, target_coords_sorted)
    target_coords_aligned = np.dot(target_coords_sorted, R)

    reordered_atoms, hungarian_indices = reorder_atoms_hungarian(
        ref_atoms_sorted, [(atom[0], coord) for atom, coord in zip(target_atoms_sorted, target_coords_aligned)]
    )
    reordered_coords = np.array([atom[1] for atom in reordered_atoms])

    R_final = kabsch(ref_coords_sorted, reordered_coords)
    final_coords_aligned = np.dot(reordered_coords, R_final)

    rmsd = calculate_rmsd(ref_coords_sorted, final_coords_aligned)

    if rmsd < 0.2:
        #combined_order = sort_indices[hungarian_indices[invert_positions(ref_sort_indices)]]
        #return reordered_atoms, combined_order
        duplicate = True
    else:
        duplicate = False
    return duplicate


def find_best_sort_order_ensemble(ensemble_molecules, reference_molecules):
    """
    Finds the best atom sort order based on RMSD between reference and target ensembles.

    Parameters:
    ensemble_filename (str): The XYZ file of the target ensemble.
    reference_ensemble_filename (str): The XYZ file of the reference ensemble.

    Returns:
    tuple: The reordered atoms, the best sort order, reference energy, and target energy.
    """

    for ref_energy, ref_atoms in reference_molecules:
        ref_moi = calculate_moments_of_inertia(ref_atoms)

        for energy, atoms in ensemble_molecules:
            if abs(energy - ref_energy) > 2e-3:
                continue

            target_moi = calculate_moments_of_inertia(atoms)
            if np.linalg.norm(ref_moi - target_moi) > 1e2: #tested, and 1e2 catches some more ... unweighted moi
                continue

            ref_atoms_sorted, ref_sort_indices = reorder_by_centroid(ref_atoms)
            target_atoms_sorted, sort_indices = reorder_by_centroid(atoms)

            ref_coords_sorted = np.array([atom[1] for atom in ref_atoms_sorted])
            target_coords_sorted = np.array([atom[1] for atom in target_atoms_sorted])
            R = kabsch(ref_coords_sorted, target_coords_sorted)
            target_coords_aligned = np.dot(target_coords_sorted, R)

            reordered_atoms, hungarian_indices = reorder_atoms_hungarian(
                ref_atoms_sorted, [(atom[0], coord) for atom, coord in zip(target_atoms_sorted, target_coords_aligned)]
            )
            reordered_coords = np.array([atom[1] for atom in reordered_atoms])

            R_final = kabsch(ref_coords_sorted, reordered_coords)
            final_coords_aligned = np.dot(reordered_coords, R_final)

            rmsd = calculate_rmsd(ref_coords_sorted, final_coords_aligned)

            if rmsd < 0.2:
                combined_order = sort_indices[hungarian_indices[invert_positions(ref_sort_indices)]]
                return reordered_atoms, combined_order, ref_energy, energy

    return None, None, None, None


def reorder_molecule(molecule, combined_order):
    """
    Reorders the atoms of a molecule based on a given sort order.

    Parameters:
    molecule (tuple): A tuple containing the energy and atom list of the molecule.
    combined_order (numpy.ndarray): The atom reorder indices.

    Returns:
    tuple: The reordered molecule.
    """
    _, atoms = molecule
    reordered_atoms = [atoms[i] for i in combined_order]
    return molecule[0], reordered_atoms


def write_xyz(filename, molecules):
    """
    Writes the reordered molecular structures to an XYZ file.

    Parameters:
    filename (str): The output filename.
    molecules (list): A list of reordered molecules.
    """
    with open(filename, 'w') as file:
        for energy, atoms in molecules:
            file.write(f"{len(atoms)}\n")
            file.write(f"{energy}\n")
            for atom in atoms:
                file.write(f"{atom[0]} {atom[1][0]:.6f} {atom[1][1]:.6f} {atom[1][2]:.6f}\n")


def main():
    """
    Main function that handles command-line arguments and executes the reordering process.
    """
    parser = argparse.ArgumentParser(description='Reorder molecular ensembles using RMSD matching.')
    parser.add_argument('ensemble_filename', type=str, help='XYZ file for the ensemble of target structures.')
    parser.add_argument('reference_ensemble_filename', type=str, help='XYZ file for the ensemble of reference structures.')
    parser.add_argument('output_filename', type=str, help='XYZ file to write the reordered structures.')

    args = parser.parse_args()

    ensemble_molecules = parse_xyz(args.ensemble_filename)
    reference_molecules = parse_xyz(args.reference_ensemble_filename)

    best_order_atoms, best_sort_order, ref_energy, target_energy = find_best_sort_order_ensemble(
        ensemble_molecules, reference_molecules
    )

    if best_sort_order is not None:
        print(f"Match found with RMSD < 0.1. Reference energy: {ref_energy}, Target energy: {target_energy}")
        ensemble_molecules = parse_xyz(args.ensemble_filename)
        reordered_ensemble = [reorder_molecule(molecule, best_sort_order) for molecule in ensemble_molecules]
        write_xyz(args.output_filename, reordered_ensemble)
        print(f"Reordered ensemble written to {args.output_filename}")
    else:
        print("No suitable order found in the reference ensemble.")


if __name__ == '__main__':
    main()

