import numpy as np
import matplotlib.pyplot as plt
from src import *
import os

fig_path = os.getcwd() + "/SecondMidterm/latex/figures"

state = np.array([True, True, False, False])

gvals = np.linspace(-1, 1, 101)


HFenergies = np.zeros(101)
FCIenergies = np.zeros((6, 101))
CIDenergies = np.zeros((5, 101))
PMenergies = np.zeros(101)

for i, g in enumerate(gvals):
    PM = PairingModel(state, g)
    FCIsolver = FCI(len(state), np.sum(state), g, True)
    CIDsolver = FCI(len(state), np.sum(state), g, False)
    spenergies, basis_coeffs, PMenergies[i] = PM.solve_paired()
    #energies = HFsolver.HFsolve(10)
    FCIenergies[:, i], FCIcoeffs = FCIsolver.solve()
    CIDenergies[:, i], CIDcoeffs = CIDsolver.solve()
    hf = HartreeFock(4, 8, g, info=False)
    hfspenergies, HFenergies[i] = hf.run(3)




#plt.plot(gvals, PMenergies, label="PM")
plt.plot(gvals, CIDenergies[0, :], label="CID", alpha=0.5)
plt.plot(gvals, HFenergies, label="HF", alpha=0.5)
plt.plot(gvals, FCIenergies[0, :], label="FCI", alpha=0.5)
plt.xlabel(r"g [$\xi$]")
plt.ylabel(r"Ground state energy [$\xi$]")
plt.legend()
plt.savefig(fig_path + "/comparisons.pdf", format="pdf", bbox_inches="tight")
plt.show()


zeroth = np.zeros(101)
first = np.zeros(101)
second = np.zeros(101)
third = np.zeros(101)
fourth = np.zeros(101)
for i, g in enumerate(gvals):
    rspt = RSPT(2, 4, g)
    zeroth[i] = rspt.energy_approximation(0)
    first[i] = rspt.energy_approximation(1)
    second[i] = rspt.energy_approximation(2)
    third[i] = rspt.energy_approximation(3)
    fourth[i] = rspt.energy_approximation(4)

print(f"Second[0] = {second[0]}")
#plt.plot(gvals, zeroth, label=r"$0^{\mathrm{th}}$ order", alpha=0.5)
#plt.plot(gvals, first, label=r"$1^{\mathrm{st}}$ order", alpha=0.5)
plt.plot(gvals, second, label=r"$2^{\mathrm{nd}}$ order", alpha=0.5)
#plt.plot(gvals, third, label=r"$3^{\mathrm{rd}}$ order", alpha=0.5)
#plt.plot(gvals, fourth, label=r"$4^{\mathrm{th}}$ order", alpha=0.5)
#plt.plot(gvals, FCIenergies[0, :], label="FCI", alpha=0.5)
plt.xlabel(r"g [$\xi$]")
plt.ylabel(r"Energy [$\xi$]")
plt.legend()
plt.savefig(fig_path + "/rspt.pdf", format="pdf", bbox_inches="tight")
plt.show()
