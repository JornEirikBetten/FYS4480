import numpy as np



class RSPT:
    def __init__(self, nbasis, nparticles, g):
        self.g = g
        self.nparticles = nparticles
        self.nbasis = nbasis
        self.ground_state = np.array([1*(i<self.nparticles) for i in range(self.nbasis)], dtype=bool)
        self.mbstates = self.all_states()
        """
        for state in self.mbstates:
            print(f"Interaction between ground state, {self.ground_state} and")
            print(f"state, {self.ground_state}: {self.interaction(self.ground_state, state)}")
        """

    def info(self):
        print(f"Initiated with g={self.g} and ground state:")
        print(self.ground_state)
        print("Many-body states as rows:")
        print(self.mbstates)

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

        ph02 = self.ph(0, 2)
        ph13 = self.ph(1,3)
        twoptwoh = self.ground_state^ph02^ph13
        mbstates.append(twoptwoh)
        return np.array(mbstates)

    def interaction(self, state1, state2):
        interaction = -self.g/2*np.sum(state1==state2)*self.nparticles/self.nbasis
        return interaction

    def H0(self, state):
        energy = 0
        for i, occupied in enumerate(state):
            if occupied:
                energy += 2*i
        return energy

    def zeroth_order(self):
        energy = self.H0(self.ground_state)
        return energy

    def first_order(self):
        energy = self.interaction(self.ground_state, self.ground_state)
        return energy

    def second_order(self):
        energy = 0
        for i, state in enumerate(self.mbstates):
            if i==0:
                continue
            else:
                V0state = self.interaction(self.ground_state, state)
                Vstate0 = self.interaction(state, self.ground_state)
                denominator = self.H0(self.ground_state)-self.H0(state)
                energy += V0state*Vstate0/denominator

        #print(f"Interaction 4p4h state and ground state: {self.interaction(self.ground_state, np.array([False, False, True, True]))}")
        return energy

    def third_order(self):
        energy = 0
        for i, statei in enumerate(self.mbstates):
            if i==0:
                continue
            else:
                for j, statej in enumerate(self.mbstates):
                    if j==0:
                        continue
                    else:
                        V0i = self.interaction(self.ground_state, statei)
                        Vj0 = self.interaction(statej, self.ground_state)
                        E1 = self.first_order()*(i==j)
                        Vij = self.interaction(statei, statej)
                        denominator = (self.H0(self.ground_state)-self.H0(statei))*(self.H0(self.ground_state)-self.H0(statej))
                        energy += (V0i*(Vij - E1)*Vj0)/denominator
        return energy

    def fourth_order(self):
        energy = 0
        for i, statei in enumerate(self.mbstates):
            if i==0:
                continue
            else:
                for j, statej in enumerate(self.mbstates):
                    if j==0:
                        continue
                    else:
                        for k, statek in enumerate(self.mbstates):
                            if k==0:
                                continue
                            else:
                                E2 = self.second_order()
                                Wij = self.interaction(statei, statej)-self.first_order()*(i==j)
                                Wjk = self.interaction(statej, statek)-self.first_order()*(j==k)
                                V0i = self.interaction(self.ground_state, statei)
                                Vk0 = self.interaction(statek, self.ground_state)
                                diffi = self.H0(self.ground_state)-self.H0(statei)
                                diffj = self.H0(self.ground_state)-self.H0(statej)
                                diffk = self.H0(self.ground_state)-self.H0(statek)
                                denom = diffi*diffj*diffk
                                energy += V0i*Wij*Wjk*Vk0/denom
                energy -= E2*V0i*V0i/(diffi*diffi)

        return energy


    def energy_approximation(self, order):
        if order==0:
            energy = self.zeroth_order()
        elif order==1:
            energy = self.zeroth_order() + self.first_order()
        elif order==2:
            energy = self.zeroth_order() + self.first_order() +\
                     self.second_order()
        elif order==3:
            energy = self.zeroth_order() + self.first_order() +\
                     self.second_order() + self.third_order()
        else:
            energy = self.zeroth_order() + self.first_order() +\
                     self.second_order() + self.third_order() +\
                     self.fourth_order()
        return energy





if __name__=="__main__":
    rspt = RSPT(4, 2, -1)
    rspt.info()

    for i, state1 in enumerate(rspt.mbstates):
        print(f"Energy of state {i}: {rspt.H0(state1)}")
        for j, state2 in enumerate(rspt.mbstates):
            print(f"Interaction between state {i} and state {j}: {rspt.interaction(state1, state2)}")
