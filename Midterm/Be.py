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





V = pd.read_csv(os.getcwd() + "/V.csv")
VV = V["V"].copy()
VNew = V.copy()
for i in range(81):
    VV[i] = VV[i].replace("Z", "4")
    VV[i] = VV[i].replace("Sqrt[","sqrt(")
    VV[i] = VV[i].replace("]", ")")

for i in range(81):
    VNew["V"][i] = VV[i]

VBedata = VNew
reformatter = Reformatter(3)
VBe = reformatter.get_values(VBedata)
print(VBe[0,0,0,0])
Z = 4
print(VBe)
nbasis = 6
nparticles = 4
ground_Be = np.zeros(6, dtype=bool)
for i in range(nparticles):
    ground_Be[i] = True

MBState = State(ground_Be, Z, VBe)
print(MBState.H0(0))
print(MBState.HI(0,0))
print(MBState.H0(0)+MBState.HI(0,0))


H, eigvals, eigvecs = MBState.solve()

print("Hamiltonian matrix: ")
print(H)
print("Eigenvalues: ")
print(eigvals)
print("Eigenvectors: ")
print(eigvecs)
