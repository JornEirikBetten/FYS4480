import numpy as np


class FCI:
    def __init__(self, nbasis, nparticles, g, fullCI):
        self.g = g
        self.fullCI = fullCI 
        self.nparticles = nparticles
        self.nbasis = nbasis
        self.ground_state = np.array([1*(i<self.nparticles) for i in range(self.nbasis)], dtype=bool)
        self.mbstates = self.all_states()
        self.H = self.HamiltonianMatrix()


    def info(self):
        print(f"Initiated with g={self.g} and ground state:")
        print(self.ground_state)
        print("Many-body states as rows:")
        print(self.mbstates)
        print("Hamiltonian matrix: ")
        print(self.H)

    def ph(self, particle, hole):
        hole = [hole]
        particle = [particle]
        ph = np.array([1*(i in hole) or 1*(i in particle) for i in range(self.nbasis)], dtype=bool)
        return ph

    def all_states(self):
        possible_holes = [i for i in range(self.nparticles)]
        possible_particles = [i + self.nparticles for i in range(self.nbasis-self.nparticles)]
        mbstates = []
        mbstates.append(self.ground_state)
        for i in possible_holes:
            for j in possible_particles:
                ph = self.ph(i, j)
                oneponeh = self.ground_state^ph
                mbstates.append(oneponeh)
        if self.fullCI:
            ph02 = self.ph(0, 2)
            ph13 = self.ph(1,3)
            twoptwoh = self.ground_state^ph02^ph13
            mbstates.append(twoptwoh)
        return np.array(mbstates)

    def HamiltonianMatrix(self):
        nstates = self.mbstates.shape[0]
        H = np.zeros((nstates, nstates))
        for i, state1 in enumerate(self.mbstates):
            for j, state2 in enumerate(self.mbstates):
                energy = 0
                if np.sum(state1==state2)==self.nbasis:
                    for p in range(self.nbasis):
                        energy += 2*p*state1[p]
                energy -= self.g/2*np.sum(state1==state2)*self.nparticles/self.nbasis
                H[i, j] = energy

        return H

    def solve(self):
        eigvals, eigvecs = np.linalg.eigh(self.H)
        return eigvals, eigvecs







if __name__ =="__main__":
    g = 1
    fci = FCI(4, 2, g)
    fci.info()
