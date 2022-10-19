import numpy as np 


class Hamiltonian:
    def __init__(self, Z, states, V):
        self.Z = Z
        self.brastates = states
        self.ketstates = states.T
        self.V = V
        self.nparticles = np.sum(self.ketstates[:,0])
        self.h0Energies = [-Z*Z/2, -Z*Z/2, -Z*Z/8,\
                           -Z*Z/8, -Z*Z/18, -Z*Z/18]
        if self.nparticles == 2:
            self.ground_state = [[0, 0], [0, 1]]
            self.a = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
            self.i = [[0, 0], [0, 0], [0, 1], [0, 0], [0, 1]]
        elif self.nparticles == 4:
            self.ground_state = [[0, 0], [0, 1], [1, 0], [1, 1]]
            self.a = [[0, 0], [2, 0], [2, 1], [2, 0], [2, 1]]
            self.i = [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1]]

        else:
            msg = 'States does not correspond to He or Be!'
            raise ValueError(msg)

        print(f"Number of particles: {self.nparticles}")

        print("States (ket) : ")
        print(self.ketstates)
        print("Single-particle energies: ")
        print(self.h0Energies)
        print("V:")
        print(self.V)
        print("First column of ket: ")
        print(self.ketstates[:, 0])

    def H0(self, p, q):
        """
        <p|h_0|q>
        """
        if p==q:
            energies = np.where(self.ketstates[:, p], \
                                self.h0Energies, 0.0)
            energy = np.sum(energies)
        else:
            energy = 0.
        return energy


    def v(self, p, q, r, s):
        """
        <pq|v|rs>
        """
        return self.V[p, q, r, s]

    def gs_H_ia(self, i, a):
        """
        Sum_k<ik||ak>
        """
        ni = i[0]; na = a[0]
        msi = i[1]; msa = a[1]
        energy = 0
        for k, spinstate in enumerate(self.ground_state):
            if spinstate == i:
                energy += 0
            else:
                energy += self.V[ni, spinstate[0], na, spinstate[0]]
        return energy

    def ai_H_jb(self, a, j, i, b):
        """
        <aj||bi> + sum_{k\neq i\leq F}[<ak||bk>\delta{ij}
        - <ik||ik>\delta{ab}]
        + 0.5\sum_{kl\leqF}<kl||kl>\delta{ab}\delta{ij}
        """
        na = a[0]; nj=j[0]; nb=b[0]; ni=i[0]
        energy = 0
        if a==b and i==j:
            energy += self.H0(na*2, na*2)
            energy += self.V[na, ni, ni, na]
            for k, kstate in enumerate(self.ground_state):
                if i==kstate:
                    energy += 0
                else:
                    energy += 0.5*self.V[na, kstate[0], na, kstate[0]]
                for l, lstate in enumerate(self.ground_state):
                    if lstate == kstate or lstate == i:
                        energy += 0
                    else:
                        energy += 0.5*self.V[lstate[0], kstate[0], lstate[0], kstate[0]]


        elif i==j:
            energy += self.V[na, ni, ni, nb]
            for k, spinstate in enumerate(self.ground_state):
                if i==spinstate:
                    energy += 0
                else:
                    energy += 0.5*self.V[na, spinstate[0], nb, spinstate[0]]

        elif a==b:
            energy += self.V[na, nj, ni, na]
            for k, spinstate in enumerate(self.ground_state):
                if i==spinstate or j==spinstate:
                    energy += 0
                else:
                    energy -= 0.5*self.V[nj, spinstate[0], ni, spinstate[0]]

        else:
            energy += self.V[na, ni, nj, nb]
        return energy



    def ERef(self):
        H0Energy = self.H0(0,0)
        vEnergy = 0
        for k, kstate in enumerate(self.ground_state):
            for l, lstate in enumerate(self.ground_state):
                if kstate == lstate:
                    vEnergy += 0
                else:
                    nk = kstate[0]; nl = lstate[0]
                    vEnergy += 0.5*self.V[nk, nl, nk, nl]

        return H0Energy + vEnergy

    def calculate_Hmat(self):
        Hmat = np.zeros((5, 5))
        a = self.a
        i = self.i
        Hmat[0, 0] = self.ERef()
        for k in range(1, 5):
            Hmat[0, k] = self.gs_H_ia(i[k], a[k])
            Hmat[k, 0] = Hmat[0, k]

        for k in range(1, 5):
            for l in range(1, 5):
                Hmat[k, l] = self.ai_H_jb(a[k], i[l], i[k], a[l])
        return Hmat
