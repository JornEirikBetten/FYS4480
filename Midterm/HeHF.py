from src import QuantumState, State, Hamiltonian, Reformatter, HF
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
    VV[i] = VV[i].replace("Z", "2")
    VV[i] = VV[i].replace("Sqrt[","sqrt(")
    VV[i] = VV[i].replace("]", ")")

for i in range(81):
    VNew["V"][i] = VV[i]

VHedata = VNew
reformatter = Reformatter(3, 6)
VHe = reformatter.get_values(VHedata)
VHe_AS = reformatter.make_antisymmetrized_elements(VHedata)
for i in range(2):
    ni = i // 2
    for j in range(2):
        nj = j // 2
        print(f"for i={i} and j={j}: ")
        print("VHE: ", VHe[ni, nj, ni, nj])
        print("VHe (AS): ", VHe[ni, nj, nj, ni])
        print("VHeAS: ", VHe_AS[i, j, i, j])
        print("VHeAS(AS): ", VHe_AS[i, j, j, i])



Z = 2
nbasis = 6
nparticles = 2
ground_He_ansatz = np.zeros(6, dtype=bool)
for i in range(nparticles):
    ground_He_ansatz[i] = True

hf = HF(ground_He_ansatz, Z, VHe_AS)
hfenergies = hf.HFsolve(10, 1e-8)
