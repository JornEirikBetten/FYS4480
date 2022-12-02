import numpy as np
import pandas as pd
from sympy import sympify


class Reformatter:
    def __init__(self, nstates, nbasis):
        self.nstates = nstates
        self.V = np.zeros((nstates, nstates, nstates, nstates))
        self.VAS = np.zeros((nbasis, nbasis, nbasis, nbasis))
        self.nbasis = nbasis

    def get_values(self, df):
        """
        Args:
            df pd.DataFrame
                updates values of self.V
        """
        for i in range(len(df["V"])):
            p = df["p"][i]-1
            q = df["q"][i]-1
            r = df["r"][i]-1
            s = df["s"][i]-1
            self.V[p, q, r, s] = float(sympify(df["V"][i]))
        return self.V

    def make_antisymmetrized_elements(self, df):
        """
        Args:
            df pd.DataFrame
                updates values of self.VAS
        """
        V = self.get_values(df)
        for i in range(self.nbasis):
            msi = (i%2 == 0)
            ni = i // 2
            for j in range(self.nbasis):
                msj = (j%2 == 0)
                nj = j // 2
                for k in range(self.nbasis):
                    msk = (k%2 == 0)
                    nk = k // 2
                    for l in range(self.nbasis):
                        msl = (l%2 == 0)
                        nl = l // 2
                        if (ni == nj and msi == msj):
                            self.VAS[i, j, k, l] += 0
                        elif (nk==nl and msk == msl):
                            self.VAS[i, j, k, l] += 0
                        else:
                            self.VAS[i, j, k, l] += V[ni, nj, nk, nl]*(msj==msl)*(msi==msk)
                            self.VAS[i, j, k, l] += -V[ni, nj, nl, nk]*(msj==msl)*(msi==msk)*(msj==msk)
        return self.VAS
