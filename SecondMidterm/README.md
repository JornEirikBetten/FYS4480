# Pairing Model
The Pairing model contains only paired states, and the interaction is defined as a pairwise interaction.

## FCI
Takes in the number of basis functions and the number of particles and calculates the Hamiltonian matrix with the pairing interaction before it diagonalizes it. Highly specialized code that struggles to find doubly, triply, quadruply, etc excited states.

## HartreeFock
Hartree-Fock scheme with the pairing interaction. Takes in the number of particles and number of basis functions as input, and creates the density matrix before calculating the HF matrix which is diagonalized iteratively until convergence in the HF energies.

## RSPT
Rayleigh-Schroedinger perturbation theory implementation, with the pairing interaction. Takes in the number of basis functions and the number of particles and performs the calculation to the specified order.
