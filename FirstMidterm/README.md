# Midterm

## State
Class that takes in a ground state ansatz in a hydrogen-like basis set for single-particle functions, and returns all possible singly-excited states. It also is able to calculate what I call the CI1p1h energies of the system.

## QuantumState
Class that takes in a bit string and converts it into particles and holes from the particle-hole formalism.

## Reformatter
Class that reformats interactions.

## HF
Class that performs Hartree-Fock calculations given a ground state ansatz, and tabulated values for the electron-electron interactions.

### Helium calculations
CI1p1h: He.py command: python3 He.py \
HF: HeHF.py command: python3 HeHF.py \

### Beryllium calculations
CI1p1h: Be.py command: python3 Be.py \
HF: BeHF.py command: python3 BeHF.py \


### V.csv
Electron-electron interaction table, which needs some modifications before it can be
applied to the problems (see Reformatter, and He.py i.e.).
