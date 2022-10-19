import numpy as np



class States:
    def __init__(self, nbasis, nparticles, Z, V):
        """
        Creates an instance of States, which
        contains all states of our many-
        particle system (with preserved spin).

        Args:
            nbasis int,
                number of spinorbitals
                in our single-particle
                basis.
            nparticles int,
                number of particles in
                many-particle system.
            ex_deg int,
                number of maximum
                excitations from ground
                state.

        """
        self.V = V
        self.Z = Z
        self.nbasis = nbasis
        self.nparticles = nparticles
        self.SPenergies = self.single_particle_energies()
        self.ground_state = np.zeros(nbasis, dtype=bool)
        for i in range(self.nparticles):
            self.ground_state[i] = True

        print("Initialized ground state: ")
        print(self.ground_state)
        print("\n")
        print(f"H0 ground state: {self.H0(self.ground_state)}")

        print("Single-particle energies: ")
        print(self.SPenergies)
        print("\n")

        # Possible hole states (all i)
        self.holes = np.zeros(shape=(self.nparticles, self.nbasis), dtype=bool)
        for i in range(self.nparticles):
            self.holes[i,i] = True

        # Possible particle states (all a)
        self.particles = np.zeros(shape=(self.nbasis-self.nparticles, self.nbasis), dtype=bool)
        for i in range(self.nbasis-self.nparticles):
            self.particles[i, self.nparticles + i] = True

        # Make set of all states
        self.states = self.make_set_of_states()
        print("States to be used in energy calculations: ")
        print(self.states)
        print("\n")
        self.dimMat, nbasis = self.states.shape
        # Check all states
        print("Conservation of spin and particle for each row in states:")
        for i in range(self.dimMat):
            print(self.check_spin(self.states[i, :]))
        print("\n")
        print("Calucalte H[i,j]: ")
        print(self.calculate_Hij(0, 0))
        print("\n")

    def add_states(self, state1, state2):
        """
        Makes the state that is the XOR
        evaluation between state1 and
        state2.
        """
        combined_state = state1^state2
        return combined_state

    def check_spin(self, state):
        """
        if i%2 = 1 spin up
        if i%2 = 0 spin down

        Checks total spin in state
        """
        # first check number of particles:
        if np.sum(state) != np.sum(self.ground_state):
            print("The number of particles is not conserved.")

        indices = [i for i in range(self.nbasis) if state[i]]
        spin = 0
        for index in indices:
            if index%2:
                spin += 1
            else:
                spin -= 1

        return spin==0

    def make_set_of_states(self):
        ground_state = self.ground_state
        ex_deg = self.ex_deg
        states = [ground_state]
        num_excited, nbasis = self.particles.shape
        if num_excited > self.nparticles:
            for i in range(num_excited):
                ex_state = self.add_states(ground_state, self.particles[i, :])
                for j in range(self.nparticles):
                    if j%2 == i%2:
                        state = self.add_states(ex_state, self.holes[j, :])
                        states.append(state)

        else:
            for i in range(self.nparticles):
                hole_state = self.add_states(ground_state, self.holes[i, :])
                for j in range(num_excited):
                    if j%2 == i%2:
                        state = self.add_states(hole_state, self.particles[j, :])
                        states.append(state)

        return np.array(states)


    def single_particle_energies(self):
        energies = []
        for i in range(self.nbasis):
            n = i//2 + 1
            energy = -self.Z*self.Z/(2*n*n)
            energies.append(energy)
        return np.array(energies)

    def H0(self, state):
        """
        Onebody contribution
        = <state|H0|state>
        """
        energies = np.where(state, self.SPenergies, 0)
        return np.sum(energies)


    def calculate_Hij(self, i, j):
        """
        H[i,j] = <state[i]|H|state[j]>
        """
        energy = 0
        if i == j:
            energy += self.H0(self.states[i, :])
            print(energy)
            energy += self.elec_elec_interactions(i, i)
            print(energy)

        return energy
    def elec_elec_interactions(self, i, j):
        """
        state[i] = <e1|
        state[j] = |e2>
        """
        states = self.states
        energy = 0

        hpi = self.add_states(self.ground_state, self.states[i, :])
        hpj = self.add_states(self.ground_state, self.states[j, :])
        shi = hpi[0] // 2
        spi = hpi[1] // 2
        shj = hpj[0] // 2
        spj = hpj[1] // 2

        if hpj[0]%2 == hpi[1]%2:
            energy += self.V[spi, shj, shi, spj]
            energy -= self.V[spi, shj, spj, shi]
        else:
            energy += self.V[spi, shj, shi, spj]

        if i==0:
            if j==0:
                indices = np.array([k for k in range(self.nbasis) if states[i,k]])
                sindices = indices // 2
                print(indices)

                for k in range(self.nparticles):
                    for l in range(self.nparticles):
                        if k==l:
                            energy += 0
                        if indices[k]%2 == indices[l]%2:
                            energy += 0.5*self.V[sindices[k], sindices[l], sindices[k], sindices[l]] -\
                                      0.5*self.V[sindices[k], sindices[l], sindices[l], sindices[k]]
                        else:
                            energy += 0.5*self.V[sindices[k], sindices[l], sindices[k], sindices[l]]
            else:
                indices = np.array([k for k in range(self.nbasis) if states[j, k]])
                sindices = indices // 2
                hp = self.add_states(self.ground_state, states[j, :])
                hp_idx = np.array([k for k in range(self.nbasis) if hp[k]])
                s_h_idx = hp_idx[0] // 2
                s_p_idx = hp_idx[1] // 2
                for j in range(self.nparticles):
                    if hp_idx[0] == j:
                        energy += 0
                    elif hp_idx[0]%2 == j%2:
                        energy += self.V[s_h_idx, j//2, j//2, s_p_idx]
                        energy -= self.V[s_h_idx, j//2, s_p_idx, j//2]
                    else:
                        energy += self.V[s_h_idx, j//2, j//2, s_p_idx]


        elif i==j:
            indices = np.array([k for k in range(self.nbasis) if states[i,k]])
            sindices = indices // 2
            print(indices)
            hp = self.add_states(self.ground_state, states[i, :])
            print(hp)
            hp_idx = np.array([k for k in range(self.nbasis) if holeparticle[k]])
            s_h_idx = hp_idx[0] // 2
            s_p_idx = hp_idx[1] // 2
            energy += V[s_particle_index, s_hole_index, s_hole_index, sindices[-1]]
            energy -= V[sindices[-1], s_hole_index, sindices[-1], s_hole_index]
            for k in range(self.nparticles):
                for l in range(self.nparticles):
                    if k==l:
                        energy += 0
                    if indices[k]%2 == indices[l]%2:
                        energy += 0.5*self.V[sindices[k], sindices[l], sindices[k], sindices[l]] -\
                                  0.5*self.V[sindices[k], sindices[l], sindices[l], sindices[k]]
                    else:
                        energy += 0.5*self.V[sindices[k], sindices[l], sindices[k], sindices[l]]
        else:
            hpi = self.add_states(self.ground_state, states[i, :])
            hpj = self.add_states(self.ground_state, states[j, :])
            energy += 0

        return energy

    def e_e_int(self, i, j):
        # hole-particle states i, j
        hpi = self.add_states(self.ground_state, self.states[i, :])
        hpj = self.add_states(self.ground_state, self.states[j, :])
        shi = hpi[0] // 2
        spi = hpi[1] // 2
        shj = hpj[0] // 2
        spj = hpj[1] // 2

        if hpj[0]%2 == hpi[1]%2:
            energy += self.V[spi, shj, shi, spj]
            energy -= self.V[spi, shj, spj, shi]
        else:
            energy += self.V[spi, shj, shi, spj]
