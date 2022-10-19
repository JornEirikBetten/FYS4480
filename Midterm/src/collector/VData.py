import numpy as np
import pandas as pd
from sympy import sympify


class Reformatter:
    def __init__(self, nstates):
        self.nstates = nstates
        self.V = np.zeros((nstates, nstates, nstates, nstates))

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
