# -*- coding: utf-8 -*-
"""HighD - init

Classifier Certainty Visualization.
"""
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import copy
import itertools
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

np.random.seed(0)

plt.style.use("illumina.mplstyle")

class HighD:
    def __init__(self, df, targets):
        # Select numerical and scale the data to [0, 1].
        self.df = df.select_dtypes(include=np.number)
        self.targets = np.array(targets)
        scaler = MinMaxScaler()
        scale = scaler.fit_transform(self.df)
        self.scaled = pd.DataFrame(scale)
        self.scaled.columns = self.df.columns

        # Create structure to store min and max values from scaling.
        min_max = scaler.inverse_transform([[0 for i in self.scaled.columns],
                                            [1 for i in self.scaled.columns]])
        min_max_vals = pd.DataFrame(index=["min", "max"])
        for i, col in enumerate(self.scaled.columns):
            min_max_vals[col] = [min_max[0][i], min_max[1][i]]


    def _vegas_mc_integrate(self, column, f, n=100, v=True):
        kernel = KernelDensity(0.02).fit(self.df)
        
        # Draw random sample from the sample space.
        D = pd.DataFrame(kernel.sample(n))
        D.columns = self.scaled.columns

        # Append the density of the prediction at a point.
        prediction = f(D)
        
        # Select only second index of prediction pairs.
        prediction = np.array([i[1] for i in prediction])
        
        D["prediction"] = prediction
        return D

    def scatter_plot_matrix(self):
        pd.plotting.scatter_matrix(self.scaled)
        plt.grid(b=None)
        plt.show()

        