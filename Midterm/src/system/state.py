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


class State:
    """
    State makes the 1p1h information from the ground state information
    and can perform CI1p1h calculations on many-body state. 
    """
    def __init__(self, ground_state, Z, interaction_integrals, interaction_integralsAS):
        """
        Args:
            ground_state            np.array(size=(nbasis,), dtype=bool)
                ground state ansatz in basis
            Z                       int
                electric charge on nucleus
            interaction_integrals   np.array (size=(nbasis//2, nbasis//2,
                                                    nbasis//2, nbasis//2))
                Contains all integral values of the electron-electron
                interactions.
            interaction_integralsAS np.array (size=(nbasis, nbasis,
                                                    nbasis, nbasis))
                Contains all antisymmetrized integral values of the electron-electron
                interactions.
        """
        self.Z = Z
        self.V = interaction_integrals
        # Creates the antisymmetrized integral values
        # for all indices p, q, r, s
        self.VAS = interaction_integralsAS
        self.nparticles = np.sum(ground_state)
        self.nbasis = len(ground_state)
        self.ground_state = QuantumState(ground_state)
        # Possible hole states (all i)
        self.holes = np.zeros(shape=(self.nparticles, self.nbasis), dtype=bool)
        for i in range(self.nparticles):
            self.holes[i,i] = True

        # Possible particle states (all a)
        self.particles = np.zeros(shape=(self.nbasis-self.nparticles, self.nbasis), dtype=bool)
        for i in range(self.nbasis-self.nparticles):
            self.particles[i, self.nparticles + i] = True

        # many-particle basis
        mpbasis = []
        # make set of states (only 1p1h excitations)
        states = self.make_set_of_states() # list of binary strings
        for state in states:
            qstate = QuantumState(state)
            mpbasis.append(qstate)
        self.mpbasis = mpbasis # list of QuantumStates
        print("Many-particle basis: ")
        for i in range(len(mpbasis)):
            print(f"State {i}: ")
            print(f"bin rep: {self.mpbasis[i].state}")
            print(f"n rep: {self.mpbasis[i].n}")
            print(f"ms rep: {self.mpbasis[i].ms}")
            print(f"particle-hole bin: {self.mpbasis[i].ph}")
            print(f"particle-hole n: {self.mpbasis[i].nph}")
            print(f"particle-hole ms: {self.mpbasis[i].msph}")
            print("\n")


    def make_set_of_states(self):
        """
        Makes all possible 1p1h excited states
        in our basis.
        """
        ground_state = self.ground_state.state
        states = [ground_state]
        num_excited, nbasis = self.particles.shape
        if num_excited > self.nparticles:
            for i in range(num_excited):
                ex_state = ground_state^self.particles[i, :]
                for j in range(self.nparticles):
                    if j%2 == i%2:
                        state = ex_state^self.holes[j, :]
                        states.append(state)

        else:
            for i in range(self.nparticles):
                hole_state = ground_state^self.holes[i, :]
                for j in range(num_excited):
                    if j%2 == i%2:
                        state = hole_state^self.particles[j, :]
                        states.append(state)

        return states

    def H0(self, idx):
        """
        Calculates onebody contribution to energy.
        Will only be calculated when <state[i]|H0|state[i]>,
        that is, on the diagonal of the Hamiltonian matrix.
        H0 = sum_p <p|h0|p> = -Z*Z/(2n*n)
        """
        n = np.array(self.mpbasis[idx].n) + 1
        energy = np.sum(-self.Z*self.Z/(2*n*n))
        return energy

    def HMat(self):
        """
        Finds all elements of the Hamiltonian
        matrix.
        """
        dim = len(self.mpbasis)
        HMat = np.zeros((dim, dim))
        for i in range(dim):
            HMat[i, i] += self.H0(i)
            for j in range(i, dim):
                HMat[i, j] += self.HI(i, j)
                HMat[j, i] = HMat[i, j]
        return HMat

    def HI(self, idx1, idx2):
        """
        Calculates the two-body interactions
        for an element in the Hamiltonian
        matrix.
        """
        state1 = self.mpbasis[idx1]
        state2 = self.mpbasis[idx2]
        energy = 0
        ph1 = state1.ph
        ph2 = state2.ph
        phidxs1 = [k for k in range(self.nbasis) if ph1[k]]
        phidxs2 = [k for k in range(self.nbasis) if ph2[k]]
        if np.sum(ph1)!=0:
            h1 = phidxs1[0]
            p1 = phidxs1[1]
        if np.sum(ph2)!=0:
            h2 = phidxs2[0]
            p2 = phidxs2[1]
        if idx1 == 0 and idx2 == 0:
            """
            Two-body interactions of
            ground state ansatz
            """
            for k in range(self.nparticles):
                for l in range(self.nparticles):
                    energy += 0.5*self.VAS[k, l, k, l]

        elif idx1 == 0:
            """
            Two body ints between ground state ansatz
            and singly-excited states
            """
            for k in range(self.nparticles):
                energy += self.VAS[h2, k, p2, k]

        else:
            """
            Two-body ints between two singly-excited
            states.
            """
            energy += self.VAS[p1, h2, h1, p2]
            for k in range(self.nparticles):
                energy += self.VAS[p1, k, p2, k]*(h1==h2)
                energy -= self.VAS[h2, k, h1, k]*(p1==p2)
                for l in range(self.nparticles):
                    if h1 == h2 and p1 == p2:
                        energy += 0.5*self.VAS[k, l, k, l]

        return energy

    def diagonalize(self):
        """
        Diagonalizes the Hamiltonian matrix and returns eigenvalues
        and eigenvectors associated with the diagonalization, along
        with the Hamiltonian matrix itself.
        """
        H = self.HMat()
        eigvals, eigvecs = np.linalg.eigh(H)
        return H, eigvals, eigvecs













ground_state = np.zeros(6, dtype=bool)
for i in range(2):
    ground_state[i] = True
