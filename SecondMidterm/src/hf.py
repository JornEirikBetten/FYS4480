import numpy as np
import matplotlib.pyplot as plt





class HartreeFock:
    def __init__(self, nparticles, nbasis,g, info=True):
        self.g = g
        self.nparticles = nparticles
        self.nbasis = nbasis
        self.C = np.eye(nbasis)
        self.P = self.density_matrix()
        if info:
            print("Density matrix")
            print(self.P)
            print(self.interaction(0, 4, 1, 5))

    def density_matrix(self):
        nbasis = self.nbasis
        nparticles = self.nparticles
        P = np.zeros((nbasis, nbasis))
        for i in range(nbasis):
            for j in range(nbasis):
                for alpha in range(nparticles):
                    P[i, j] += self.C[i, alpha]*self.C[j, alpha]

        return P

    def interaction(self, p, q, r, s):
        # Spins
        s_p = p % 2; s_q = q % 2; s_r = r % 2; s_s = s % 2;
        # values
        p = p // 2; q = q // 2; r = r // 2; s = s // 2;
        #print(f"sp:{s_p}, sq:{s_q}, sr:{s_r}, s_s:{s_s}")
        #print(f"p: {p}, q: {q}, r: {r}, s: {s}")

        if p != r or q != s:
            val = 0
        elif s_p == s_r or s_q == s_s:
            val = 0
        elif s_p == s_q:
            val = -self.g/2
        else:
            val = self.g/2

        return val


    def update(self, old_energies):
        nbasis = self.nbasis
        nparticles = self.nparticles
        HF = np.zeros((nbasis, nbasis))

        for p in range(nbasis):
            HF[p, p] += p // 2
            for q in range(nbasis):
                field_interaction = 0
                for r in range(nbasis):
                    for s in range(nbasis):
                        field_interaction += self.interaction(p, q, r, s)*self.P[r, s]
                HF[p, q] += field_interaction


        #print("HF Matrix")
        #print(HF)
        # Diagonalizes the HF matrix
        new_energies, self.C = np.linalg.eigh(HF)
        #print("New energies")
        #print(new_energies)
        # Prepares density matrix for the new coefficients
        self.P = self.density_matrix()
        # Finds absolute difference between the previous
        # and next HF single-particle energies
        difference = np.sum(abs(new_energies-old_energies))
        return new_energies, difference

    def run(self, max_iters, tolerance=1e-10):
        difference = 1.0
        nbasis = self.nbasis
        nparticles = self.nparticles
        energies = np.zeros(nbasis)
        iterations = 0

        while iterations<max_iters and difference > tolerance:
            """
            Runs while the difference in the HF single-particle
            energies is lower than the tolerance.
            """
            energies, difference = self.update(energies)
            #print(f"HF single-particle energies at iter {iterations+1}: ")
            #print(f"{energies}")
            iterations += 1

            if iterations%1==0:
                """
                Calculates and prints the energy of the HF ground state.
                """
                energy = 0
                for alpha in range(nparticles):
                    for p in range(nbasis):
                        energy += p // 2 *self.C[p, alpha]*self.C[p, alpha]
                        for beta in range(nparticles):
                            for q in range(nbasis):
                                for r in range(nbasis):
                                    for s in range(nbasis):
                                        interaction = self.interaction(p, q, r, s)
                                        energy += 0.5*self.C[p, alpha]*self.C[r, beta]*\
                                                      self.C[q, alpha]*self.C[s, beta]*\
                                                      interaction
                                                      #self.V[alpha, gamma, beta, delta]
                #print(f"Energy at iter {iterations}:{energy}")
                #print("\n")


        return energies, energy





if __name__=="__main__":
    gvals = np.linspace(-1, 1, 101)
    energies = np.zeros(101)
    for i, g in enumerate(gvals):
        hf = HartreeFock(4, 8, 1)
        hfspenergies, energies[i] = hf.run(10)

    plt.plot(gvals, energies)
    plt.show()
