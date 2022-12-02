import numpy as np



class PairingModel:
    def __init__(self, state, g):
        self.nbasis = len(state)
        self.noccupied = np.sum(state)
        self.ground_state = state
        self.g = g
        self.HMat = self.H_paired()
        #print(self.HMat)

    def H_paired(self):
        """
        Setting up the initial Hamiltonian matrix.
        """
        H = np.zeros((self.nbasis, self.nbasis))
        for i in range(self.nbasis):
            for j in range(self.nbasis):
                H[i,j] = 2*i*(i==j) - self.g/2

        return H

    def solve_paired(self):
        energies, coefficients = np.linalg.eigh(self.HMat)
        ground_state_energy = np.sum(energies[0:self.noccupied])
        return energies, coefficients, ground_state_energy
