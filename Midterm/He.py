from src import QuantumState, State, Reformatter
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def replace(V, val):
    """
    Formatting function.
    """
    Vcopy = V["V"].copy()
    Vreturn = V.copy()
    n = len(Vcopy)
    for i in range(n):
        Vcopy[i] = Vcopy[i].replace("Z", val)
        Vcopy[i] = Vcopy[i].replace("Sqrt[","sqrt(")
        Vcopy[i] = Vcopy[i].replace("]", ")")

    for i in range(n):
        Vreturn["V"] = Vcopy[i]

    return Vreturn





V = pd.read_csv(os.getcwd() + "/Midterm/V.csv")
VV = V["V"].copy()
VNew = V.copy()
"""
Formats the two-body matrix elements to
match specifics of helium
"""
for i in range(81):
    VV[i] = VV[i].replace("Z", "2")
    VV[i] = VV[i].replace("Sqrt[","sqrt(")
    VV[i] = VV[i].replace("]", ")")

for i in range(81):
    VNew["V"][i] = VV[i]

VHedata = VNew
reformatter = Reformatter(3, 6)
VHe = reformatter.get_values(VHedata)
"""
Makes a four-dim tensor that holds all antisymmetrized
elements of the two-body potential in our basis, but
with labeling 0,1,2,3,4,5
"""
VHe_AS = reformatter.make_antisymmetrized_elements(VHedata)

Z = 2

# Makes the ground state ansatz
nbasis = 6
nparticles = 2
ground_He_ansatz = np.zeros(6, dtype=bool)
for i in range(nparticles):
    ground_He_ansatz[i] = True

# Makes the many-body state from the ground state ansatz
MBState = State(ground_He_ansatz, Z, VHe, VHe_AS)


H, eigvals, eigvecs = MBState.solve()
H2, eigvals2, eigvecs2 = MBState.solve2()

print("Hamiltonian matrix: ")
print(H)
print("H2: ")
print(H2)
print("Eigenvalues 2: ")
print(eigvals2)
