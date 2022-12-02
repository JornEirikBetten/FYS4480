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
for i in range(81):
    VV[i] = VV[i].replace("Z", "4")
    VV[i] = VV[i].replace("Sqrt[","sqrt(")
    VV[i] = VV[i].replace("]", ")")

for i in range(81):
    VNew["V"][i] = VV[i]

VBedata = VNew
reformatter = Reformatter(3, 6)
VBe = reformatter.get_values(VBedata)
"""
Makes a four-dim tensor that holds all antisymmetrized elements of
the two-body potential in our basis, but with labeling 0,1,2,3,4,5
"""
VBeAS = reformatter.make_antisymmetrized_elements(VBedata)

Z = 4
nbasis = 6
nparticles = 4
ground_Be = np.zeros(6, dtype=bool)
for i in range(nparticles):
    ground_Be[i] = True

MBState = State(ground_Be, Z, VBe, VBeAS)
print(MBState.H0(0))
print(MBState.HI(0,0))
print(MBState.H0(0)+MBState.HI(0,0))


H, eigvals, eigvecs = MBState.diagonalize()

print("Hamiltonian matrix: ")
print(H)
print("Eigenvalues: ")
print(eigvals)
print("Eigenvectors: ")
print(eigvecs)
