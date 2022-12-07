import numpy as np
import matplotlib.pyplot as plt
from src import *
import os

fig_path = os.getcwd() + "/SecondMidterm/latex/figures"

state = np.array([True, True, False, False])

gvals = np.linspace(-1, 1, 101)

FCIenergies = np.zeros((6, 101))
CIenergies = np.zeros((5, 101))
PMenergies = np.zeros(101)
zeroth = np.zeros(101)
first = np.zeros(101)
second = np.zeros(101)
third = np.zeros(101)
fourth = np.zeros(101)
for i, g in enumerate(gvals):
    rspt = RSPT(4, 2, g)
    PM = PairingModel(state, g)
    spenergies, basis_coeffs, PMenergies[i] = PM.solve_paired()
    zeroth[i] = rspt.energy_approximation(0)
    first[i] = rspt.energy_approximation(1)
    second[i] = rspt.energy_approximation(2)
    third[i] = rspt.energy_approximation(3)
    fourth[i] = rspt.energy_approximation(4)
    FCIsolver = FCI(len(state), np.sum(state), g, True)
    CIsolver = FCI(len(state), np.sum(state), g, False)
    FCIenergies[:, i], FCIcoeffs = FCIsolver.solve()
    CIenergies[:, i], CIcoeffs = CIsolver.solve()


print(f"Second[0] = {second[0]}")
#plt.plot(gvals, zeroth, label=r"$0^{\mathrm{th}}$ order", alpha=0.5)
#plt.plot(gvals, PMenergies, label="PM")
#plt.plot(gvals, first, label=r"$1^{\mathrm{st}}$ order", alpha=0.5)
#plt.plot(gvals, second, label=r"RSPT$^{(2)}$", alpha=0.5)
#plt.plot(gvals, third, label=r"RSPT$^{(3)}$", alpha=0.5)
#plt.plot(gvals, fourth, label=r"RSPT$^{(4)}$", alpha=0.5)
#plt.plot(gvals, FCIenergies[0, :], label="FCI", alpha=0.5)
#plt.plot(gvals, abs(second-CIenergies[0, :]), label=r'|CID-RSPT$^{(2)}$|')
plt.plot(gvals, fourth-FCIenergies[0, :], label=r"FCI-RSPT$^{(4)}$")
plt.xlabel(r"g [$\xi$]")
plt.ylabel(r"$\Delta E$ [$\xi$]")
plt.legend()
plt.savefig(fig_path + "/4th_FCI.pdf", format="pdf", bbox_inches="tight")
plt.show()
