import numpy as np


class QuantumState:
    """
    Finds the quantum numbers related to
    the binary string representing the many-body
    state in the basis.
    n :
        Principal quantum number of single-particle
        state related to basis.
    ms :
        True -> spin up. False -> spin down.
    """
    def __init__(self, bin_state):
        nbasis = len(bin_state)
        nparticles = np.sum(bin_state)
        indices = np.array([k for k in range(nbasis) if bin_state[k]])
        self.state = bin_state
        self.n = indices // 2
        ms = np.zeros(nparticles, dtype=bool)
        for i, idx in enumerate(indices):
            if idx%2==0:
                ms[i] = True
        self.ms = ms
        ground_state = np.zeros(nbasis, dtype=bool)
        for i in range(nparticles):
            ground_state[i] = True
        # Finds particle and hole, with their respective
        # quantum numbers
        ph = bin_state^ground_state
        self.ph = ph
        phindices = np.array([k for k in range(nbasis) if ph[k]])
        self.nph = phindices // 2
        msph = np.zeros(len(phindices), dtype=bool)
        for i, idx in enumerate(phindices):
            if idx%2==0:
                msph[i] = True
        self.msph = msph


class HF:
    def __init__(self, ground_state, Z, interaction_integrals):

        """
        Args:
            ground_state            np.array(size=(nbasis,), dtype=bool)
                ground state ansatz in basis
            Z                       int
                electric charge on nucleus
            interaction_integrals np.array (size=(nbasis, nbasis,
                                                    nbasis, nbasis))
                Contains all antisymmetrized integral values of the electron-electron
                interactions.
        """

        self.ground_state = ground_state
        self.nbasis = len(ground_state)
        self.nparticles = np.sum(ground_state)
        self.V = interaction_integrals
        self.Z = Z

        self.C = np.eye(self.nbasis)
        self.density_matrix = self.prepare_density_matrix()
        self.basis_E = self.basis_energies()


    def prepare_density_matrix(self):
        """
        Makes the density matrix.
        """

        nbasis = self.nbasis
        nparticles = self.nparticles
        density_matrix = np.zeros((nbasis, nbasis))
        for gamma in range(nbasis):
            for delta in range(nbasis):
                s = 0.0
                for i in range(nparticles):
                    s += self.C[gamma,i]*self.C[delta,i]
                density_matrix[gamma,delta] = s
        return density_matrix


    def basis_energies(self):
        """
        Makes an array of single-particle energies.
        Only 1s, 2s, 3s, 4s, etc single-particle orbitals
        are taken into account.
        """
        energies = []
        for i in range(self.nbasis//2):
            energies.append(-self.Z*self.Z/(2*(i+1)*(i+1)))
            energies.append(-self.Z*self.Z/(2*(i+1)*(i+1)))

        return np.array(energies)

    def HFiter(self, old_energies):
        """
        One iteration of the HF algorithm.
        Makes all the HF matrix elements from the
        HF operator, and diagonalizes it.
        """


        nbasis = self.nbasis
        HFMatrix = np.zeros((nbasis, nbasis))
        basis_energies = self.basis_energies()
        # Filling the HF matrix
        for alpha in range(nbasis):
            HFMatrix[alpha][alpha] += basis_energies[alpha]
            for beta in range(nbasis):
                sumFock = 0
                for gamma in range(nbasis):
                    for delta in range(nbasis):
                        sumFock += self.density_matrix[gamma,delta]*\
                                    self.V[alpha][gamma][beta][delta]
                HFMatrix[alpha, beta] += sumFock

        # Diagonalizes the HF matrix
        new_energies, self.C = np.linalg.eigh(HFMatrix)
        # Prepares density matrix for the new coefficients
        self.density_matrix = self.prepare_density_matrix()
        # Finds absolute difference between the previous
        # and next HF single-particle energies
        difference = np.sum(abs(new_energies-old_energies))
        return new_energies, difference

    def HFsolve(self, max_iters, tolerance):
        difference = 1.0
        energies = np.zeros(self.nbasis)
        iterations = 0
        basis_energies = self.basis_energies()

        while iterations<max_iters and difference > tolerance:
            """
            Runs while the difference in the HF single-particle
            energies is lower than the tolerance.
            """
            energies, difference = self.HFiter(energies)
            print(f"HF single-particle energies at iter {iterations+1}: ")
            print(f"{energies}")
            iterations += 1
            if iterations%1==0:
                energy = 0
                for i in range(2):
                    energy += energies[i]
                    for j in range(2):
                        energy -= 0.5*self.V[i, j, i, j]

            if iterations%1==0:
                """
                Calculates and prints the energy of the HF ground state.
                """
                energy = 0
                for i in range(self.nparticles):
                    for alpha in range(6):
                        energy += self.basis_E[alpha]*self.C[alpha, i]*self.C[alpha, i]
                        for j in range(self.nparticles):
                            for beta in range(6):
                                for gamma in range(6):
                                    for delta in range(6):
                                        energy += 0.5*self.C[alpha, i]*self.C[gamma, j]*\
                                                      self.C[beta, i]*self.C[delta, j]*\
                                                      self.V[alpha, gamma, beta, delta]
                print(f"Energy at iter {iterations}:{energy}")
                print("\n")


        return energies
