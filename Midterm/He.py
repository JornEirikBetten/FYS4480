from src import QuantumState, State, Hamiltonian, Reformatter
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
print(VHe[0,0,0,0])
Z = 2
print(VHe)
nbasis = 6
nparticles = 2
ground_He_ansatz = np.zeros(6, dtype=bool)
for i in range(nparticles):
    ground_He_ansatz[i] = True

MBState = State(ground_He_ansatz, Z, VHe, VHe_AS)
print(MBState.H0(0))
print(MBState.HI(0,0))


H, eigvals, eigvecs = MBState.solve()
H2, eigvals2, eigvecs2 = MBState.solve2()

print("Hamiltonian matrix: ")
print(H)
print("Eigenvalues: ")
print(eigvals)
print("Eigenvectors: ")
print(eigvecs)
print("H2: ")
print(H2)
print("Eigenvalues 2: ")
print(eigvals2)
