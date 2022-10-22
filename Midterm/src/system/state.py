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
    Performs FCI calculations on many-body state. .
    """
    def __init__(self, ground_state, Z, interaction_integrals, interaction_integralsAS):
        """
        Args:
            ground_state        np.array(size=(nbasis,), dtype=bool)
            interaction_integrals   np.array (size=(nbasis//2, nbasis//2,
                                                    nbasis//2, nbasis//2))
                Contains all integral values of the electron-electron
                interactions.
        """
        self.Z = Z
        self.V = interaction_integrals
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

    def HI(self, idx1, idx2):
        state1 = self.mpbasis[idx1]
        state2 = self.mpbasis[idx2]
        energy = 0
        ph1 = state1.ph
        ph2 = state2.ph


        if (np.sum(ph1) != 0):
            ni = state1.nph[0]; msi = state1.msph[0]
            na = state1.nph[1]; msa = state1.msph[1]
        if (np.sum(ph2) != 0):
            nj = state2.nph[0]; msj = state2.msph[0]
            nb = state2.nph[1]; msb = state2.msph[1]

        if ((np.sum(ph1) != 0) and (np.sum(ph2) != 0)):
            # <ai||ib>_AS
            # direct
            energy += self.V[na, nj, ni, nb]
            # exchange
            energy -= self.V[na,nj,nb,ni]*(msa == msb)
            # sum_{k<F}<ak|v|bk>AS\delta{ij} - <ik|v|ik>AS\delta{ab}
            for k in range(self.nparticles):
                nk = k // 2
                msk = (k%2 == 0)
                # direct
                if (ni == nk and msi == msk):
                    energy += 0
                else:
                    energy += self.V[na, nk, nb, nk]*(msi==msj and ni==nj)
                    # exhange
                    energy -= self.V[na, nk, nk, nb]*(msk == msa)*(msk == msb)*\
                              (msi==msj and ni==nj)*(ni != nk)
                if (ni == nk and msi == msk) or (nj == nk and msj == msk):
                    energy += 0
                else:
                    energy -= self.V[nj, nk, ni, nk]*(msa==msb and na==nb)
                    energy += self.V[nj, nk, nk, ni]*(msa==msb and na==nb)*(msi==msk)

        if (np.sum(ph1) == 0) and (np.sum(ph2)!=0):
            for k in range(self.nparticles):
                nk = k // 2
                msk = (k%2 == 0)
                if (nj==nk and msk==msj):
                    energy += 0
                else:
                    energy += self.V[nj, nk, nb, nk]
                    energy -= self.V[nj, nk, nk, nb]*(msk==msb)

        for k in range(self.nparticles):
            nk = k // 2
            msk = (k%2 == 0)
            for l in range(self.nparticles):
                nl = l // 2
                msl = (l%2 == 0)
                # direct
                if (nk == nl and msk == msl):
                    energy += 0
                else:
                    energy += 0.5*self.V[nk, nl, nk, nl]*\
                            (np.sum(ph1==ph2)==self.nbasis)
                    # exchange
                    energy -= 0.5*self.V[nk, nl, nl, nk]*\
                             (np.sum(ph1==ph2)==self.nbasis)*(msk==msl)

        return energy



    def HMat(self):
        dim = len(self.mpbasis)
        HMat = np.zeros((dim, dim))
        for i in range(dim):
            HMat[i, i] += self.H0(i)
            for j in range(i, dim):
                HMat[i, j] += self.HI(i, j)
                HMat[j, i] = HMat[i,j]
        return HMat

    def HMat2(self):
        dim = len(self.mpbasis)
        HMat = np.zeros((dim, dim))
        for i in range(dim):
            HMat[i, i] += self.H0(i)
            for j in range(i, dim):
                HMat[i, j] += self.HI2(i, j)
                HMat[j, i] = HMat[i, j]
        return HMat


    def solve(self):
        H = self.HMat()
        eigvals, eigvecs = np.linalg.eigh(H)
        return H, eigvals, eigvecs

    def HI2(self, idx1, idx2):
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
            for k in range(self.nparticles):
                for l in range(self.nparticles):
                    energy += 0.5*self.VAS[k, l, k, l]
                    energy -= 0.5*self.VAS[k, l, l, k]

        elif idx1 == 0:
            for k in range(self.nparticles):
                energy += self.VAS[h2, k, p2, k]
                energy -= self.VAS[h2, k, k, p2]

        else:
            energy += self.VAS[p1, h2, h1, p2]
            energy -= self.VAS[p1, h2, p2, h1]
            for k in range(self.nparticles):
                energy += self.VAS[p1, k, p2, k]*(h1==h2)
                energy -= self.VAS[p1, k, k, p2]*(h1==h2)
                energy -= self.VAS[h2, k, h1, k]*(p1==p2)
                energy += self.VAS[h2, k, k, h1]*(p1==p2)
                for l in range(self.nparticles):
                    if h1 == h2 and p1 == p2:
                        energy += 0.5*self.VAS[k, l, k, l]
                        energy -= 0.5*self.VAS[k, l, l, k]

        return energy

    def solve2(self):
        H = self.HMat2()
        eigvals, eigvecs = np.linalg.eigh(H)
        return H, eigvals, eigvecs













ground_state = np.zeros(6, dtype=bool)
for i in range(2):
    ground_state[i] = True
