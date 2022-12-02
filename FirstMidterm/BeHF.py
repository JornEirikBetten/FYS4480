from src import QuantumState, State, Reformatter, HF
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def replace(V, val):
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
for i in range(81):
    VV[i] = VV[i].replace("Z", "4")
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

"""
Sets up the ground state ansatz for the beryllium atom.
"""
Z = 4
nbasis = 6
nparticles = 4
ground_Be_ansatz = np.zeros(nbasis, dtype=bool)
for i in range(nparticles):
    ground_Be_ansatz[i] = True

hf = HF(ground_Be_ansatz, Z, VHe_AS)
hfenergies = hf.HFsolve(1000, 1e-10)
print("Final HF energies: ", hfenergies)
