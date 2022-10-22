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
        self.ground_stateB1 = ground_state
        self.nbasis = len(ground_state)
        self.nparticles = np.sum(ground_state)
        self.V = interaction_integrals
        self.Z = Z

        self.C = np.eye(self.nbasis)
        self.density_matrix = self.prepare_density_matrix()


    def prepare_density_matrix(self):
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
        energies = []
        for i in range(self.nbasis//2):
                energies.append(-self.Z*self.Z/(2*(i+1)*(i+1)))
                energies.append(-self.Z*self.Z/(2*(i+1)*(i+1)))

        return np.array(energies)

    def HFiter(self, old_energies):
        nbasis = self.nbasis
        HFMatrix = np.zeros((nbasis, nbasis))
        basis_energies = self.basis_energies()
        for i in range(nbasis):
            HFMatrix[i, i] += basis_energies[i]
        for alpha in range(nbasis):
            for beta in range(nbasis):
                sumFock = 0
                for gamma in range(nbasis):
                    for delta in range(nbasis):
                        sumFock += self.density_matrix[gamma,delta]*\
                                    self.V[alpha][gamma][beta][delta]
                        sumFock -= self.density_matrix[gamma,delta]*\
                                    self.V[alpha,gamma,delta,beta]
                HFMatrix[alpha, beta] += sumFock
                print(HFMatrix)

                #if beta == alpha:
                #    HFMatrix[alpha][alpha] = np.sum(self.C[:, alpha]*basis_energies)

        new_energies, self.C = np.linalg.eigh(HFMatrix)
        self.density_matrix = self.prepare_density_matrix()
        """ Brute force computation of difference between previous and new sp HF energies """
        difference = np.sum(abs(new_energies-old_energies))
        print("Energy gs: ", new_energies)
        return new_energies, difference

    def HFsolve(self, max_iters, tolerance):
        difference = 1.0
        energies = np.zeros(self.nbasis)
        iterations = 0
        while iterations<max_iters and difference > tolerance:
            energies, difference = self.HFiter(energies)
            iterations += 1

        return energies
