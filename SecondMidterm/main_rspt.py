import numpy as np
import matplotlib.pyplot as plt
from src import *
import os

fig_path = os.getcwd() + "/SecondMidterm/latex/figures"

state = np.array([True, True, False, False])

gvals = np.linspace(-1, 1, 101)

FCIenergies = np.zeros((6, 101))
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
    FCIenergies[:, i], FCIcoeffs = FCIsolver.solve()


print(f"Second[0] = {second[0]}")
#plt.plot(gvals, zeroth, label=r"$0^{\mathrm{th}}$ order", alpha=0.5)
#plt.plot(gvals, PMenergies, label="PM")
plt.plot(gvals, first, label=r"$1^{\mathrm{st}}$ order", alpha=0.5)
plt.plot(gvals, second, label=r"$2^{\mathrm{nd}}$ order", alpha=0.5)
plt.plot(gvals, third, label=r"$3^{\mathrm{rd}}$ order", alpha=0.5)
plt.plot(gvals, fourth, label=r"$4^{\mathrm{th}}$ order", alpha=1.0)
plt.plot(gvals, FCIenergies[0, :], label="FCI", alpha=0.2)
plt.xlabel(r"g [$\xi$]")
plt.ylabel(r"Energy [$\xi$]")
plt.legend()
plt.savefig(fig_path + "/rspt.pdf", format="pdf", bbox_inches="tight")
plt.show()
