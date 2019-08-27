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
        self.min_max_vals = pd.DataFrame(index=["min", "max"])
        for i, col in enumerate(self.scaled.columns):
            self.min_max_vals[col] = [min_max[0][i], min_max[1][i]]

    def density_estimate(self, f, n=100, resolution=25, k_dens=0.02):
        kernel = KernelDensity(k_dens).fit(self.scaled)

        # Draw random sample from the sample space and store.
        self.D = pd.DataFrame(kernel.sample(n))
        self.D.columns = self.scaled.columns
        prediction = f(self.D)
        self.D["prediction"] = prediction

        # Put values into bins.
        res_vals = np.linspace(-0.0, 1.0, resolution)
        self.D_bins = pd.DataFrame()
        for col in self.D.columns[:-1]:
            tmp = self.D[[col, "prediction"]]
            tmp["bin"] = pd.cut(tmp[col], bins=res_vals)
            tmp = tmp.sort_values(by="bin")
            tmp = tmp.groupby("bin").mean()
            # Select mid value of interval as index.
            self.D_bins[col] = tmp["prediction"]

        # Replace nan values with 0.5 as that is the completely uncertain
        # value.
        self.D_bins = self.D_bins.fillna(0.5)

    def scatter_plot_matrix(self):
        pd.plotting.scatter_matrix(self.scaled)
        plt.grid(b=None)
        plt.show()

    def density_scatter(self, col):
        self.D.plot.scatter(x="prediction", y=col)
        plt.title(col + " value and certainty classification")

    def vis_1d(self):
        # Shifting everything down 0.5 makes 0 the uncertain value.
        D_mid_bins = copy.deepcopy(self.D_bins)
        for col in D_mid_bins[:-1]:
            D_mid_bins[col] = D_mid_bins[col] - 0.5
        D_mid_bins.plot.bar(xlim=(-0.15, 1.15), ylim=(-0.6, 0.6),
                            title="Dimension certainty")
        plt.show()

    def vis_2d(self, title):
        # Some thanks to https://stackoverflow.com/questions/7941207/is-there-a-function-to-make-scatterplot-matrices-in-matplotlib
        np.random.seed(42)
        cols = self.D_bins.columns
        n_cols = len(self.D_bins.columns)
        fig, axes = plt.subplots(nrows=n_cols, ncols=n_cols,
                                 figsize=(n_cols * 4, n_cols * 4))

        # Hide all ticks and labels
        for ax in axes.flat:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

        res_sub = np.linspace(-0.0, 1.0, len(self.D_bins))
        Xm, Ym = np.meshgrid(res_sub, res_sub)

        # Plot the data.
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                Zm = [[(i * j)-0.5 for i in self.D_bins[cols[x]]] for
                      j in self.D_bins[cols[y]]]
                axes[x, y].contourf(Xm, Ym, Zm, levels=np.linspace(-0.5, 0.5,
                                    21), cmap="seismic")

        # Add bar charts as diagonal plots.
        for i, col in enumerate(cols):
            bar_vals = self.D_bins[col] - 0.5
            # Select mid values of intervals for x values.
            x = [i.mid for i in np.array(bar_vals.keys())]
            y = bar_vals.values
            axes[i, i].bar(x=x, height=y, width=1/len(x))
            axes[i, i].set_xlim(0.0, 1.0)
            axes[i, i].set_ylim(-0.5, 0.5)
            axes[i, i].grid(False)

        # Add X labels to all of bottom row.
        for i, _ in enumerate(cols):
            axes[n_cols-1, i].xaxis.set_visible(True)
            axes[n_cols-1, i].set_xlabel(cols[i] + " [" + str(self.min_max_vals[cols[i]]["min"]) + 
                                         ", " + str(self.min_max_vals[cols[i]]["max"]) + "]")

            axes[i, 0].yaxis.set_visible(True)
            axes[i, 0].set_ylabel(cols[i] + " [" + str(self.min_max_vals[cols[i]]["min"]) + 
                                            ", " + str(self.min_max_vals[cols[i]]["max"]) + "]")

        if title:
            fig.suptitle(title, size="26")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
