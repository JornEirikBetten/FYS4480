import numpy as np
import matplotlib.pyplot as plt
import os
"""
plotting parameters
"""

fontsize = "large"
params = {"font.family": "serif",
          "font.sans-serif": ["Computer Modern"],
          "axes.labelsize": fontsize,
          "legend.fontsize": fontsize,
          "xtick.labelsize": fontsize,
          "ytick.labelsize": fontsize,
          "legend.handlelength": 2
          }

plt.rcParams.update(params)

fig_path = os.getcwd() + "/SecondMidterm/latex/figures"

runs = 1001
gvals = np.linspace(-1, 1, runs)

eigenvalues = np.zeros((6, runs))
energies = np.zeros((6, runs))
for run, g in enumerate(gvals):
    HamMat = np.ones((6,6))*(-g/2)
    for i in range(6):
        HamMat[i, i] = -g
        for j in range(6):
            if (i+j)==5:
                HamMat[i,j] = 0
    HamMat[0, 0] += 2
    HamMat[1, 1] += 4
    HamMat[2, 2] += 6
    HamMat[3, 3] += 6
    HamMat[4, 4] += 8
    HamMat[5, 5] += 10
    print(HamMat)
    eigvals, eigvecs = np.linalg.eigh(HamMat)
    for i in range(6):
        eigenvalues[i, run] = eigvals[i]


fig = plt.figure(figsize=(8,8))
for i in range(6):
    plt.plot(gvals, eigenvalues[i, :], alpha=0.5, label=fr"$\epsilon_{i}$")
plt.xlabel(r"g $[\xi]$")
plt.ylabel(r"Energy $[\xi]$")
plt.legend()
plt.savefig(fig_path+"/fullCI.pdf", format="pdf", bbox_inches="tight")
#plt.show()



eigenvalues_approximation = np.zeros((5, runs))
energies = np.zeros((5, runs))
eigenvectors_approximation = np.zeros((5, 5, runs))
for run, g in enumerate(gvals):
    HamMat = np.ones((5,5))*(-g/2)
    for i in range(5):
        HamMat[i, i] = -g
        for j in range(5):
            if (i+j)==5:
                HamMat[i,j] = 0

    HamMat[0, 0] += 2
    HamMat[1, 1] += 4
    HamMat[2, 2] += 6
    HamMat[3, 3] += 6
    HamMat[4, 4] += 8

    eigvals, eigvecs = np.linalg.eigh(HamMat)
    for i in range(5):
        eigenvalues_approximation[i, run] = eigvals[i]
        eigenvectors_approximation[:, i, run] = eigvecs[:, i]


fig = plt.figure(figsize=(8,8))
for i in range(5):
    plt.plot(gvals, eigenvalues_approximation[i, :], alpha=0.5, label=fr"$\epsilon_{i+1}$")
plt.xlabel(r"g $[\xi]$")
plt.ylabel(r"Energy $[\xi]$")
plt.legend()
plt.savefig(fig_path+"/approximationCI.pdf", format="pdf", bbox_inches="tight")
#plt.show()

fig = plt.figure(figsize=(8,8))
for i in range(1):
    for coeff in range(0, 5, 1):
        plt.plot(gvals, eigenvectors_approximation[i, coeff, :], label=f"coeff basis func {coeff}")
plt.legend()
#plt.show()
fig = plt.figure(figsize=(8,8))
for i in range(1):
    plt.plot(gvals, eigenvalues[i, :], alpha=0.5, label=fr"$\epsilon_{i+1}$ FCI")
    plt.plot(gvals, eigenvalues_approximation[i, :], alpha=0.5, label=fr"$\epsilon_{i+1}$ Approximation")
plt.xlabel("g")
plt.ylabel("Energy")
plt.legend()
plt.savefig(fig_path +"/comparisonTrueApprox.pdf", format="pdf", bbox_inches="tight")
#plt.show()

fig = plt.figure(figsize=(8,8))
for i in range(1):
    plt.plot(gvals, eigenvalues_approximation[i,:]-eigenvalues[i, :], alpha=0.5, label=fr"$\Delta\epsilon_{i+1}$")
plt.xlabel(r"g $[\xi]$")
plt.ylabel(r"$\Delta E$ $[\xi]$")
plt.legend()
plt.savefig(fig_path +"/diffTrueApprox.pdf", format="pdf", bbox_inches="tight")
#plt.show()

print(f"GS energy g=1: {eigenvalues[0, -1]}")
