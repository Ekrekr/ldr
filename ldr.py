# -*- coding: utf-8 -*-
"""LDR - init

Classifier Certainty Visualization.
"""
import pandas as pd
import numpy as np
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import copy
import itertools
import typing
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from scipy import stats

np.random.seed(0)
# Disable warning where bin cut is set (false positive).
pd.options.mode.chained_assignment = None


class LDR:
    def __init__(self, df: pd.DataFrame, targets: pd.Series, problem_type,
                 pos_val: any=None, neg_val: any=None, sample_pos: bool=True,
                 sample_neg: bool=True):
        """
        Latent Dimensionality Reduction.

        Note: Currently only supports binary classification.

        args:
            df: The data (excluding targets).
            targets: The targets of the model.
            problem_type: 'class' for classification or 'reg' for regression.
            pos_val: Value of positives in targets.
            neg_val: Value of negatives in targets.
            sample_pos: Whether to include samples from positives sample space.
            sample_neg: Wheter to include samples from negative sample space.
        """
        self.pos_val = pos_val
        self.sample_pos = sample_pos
        self.sample_neg = sample_neg
        self.problem_type = problem_type

        # Select numerical and scale the data to [0, 1].
        self.df = df.select_dtypes(include=np.number)
        self.targets = np.array(targets)

        # Targets should be min-max scaled too if regression.
        if problem_type == "reg":
            self.df["target"] = self.targets

        scaler = MinMaxScaler()
        scale = scaler.fit_transform(self.df)
        self.scaled = pd.DataFrame(scale)
        self.scaled.columns = self.df.columns

        # Copy scaled to apply target normalization weightings to without
        # modifying original.
        self.scaled_w = copy.deepcopy(self.scaled)

        # Weighting of regression density in sample space should be done here,
        # if at all.

        # Weighted sample space needed if classification.
        if problem_type == "class":
            self.scaled_w["target"] = self.targets
            pos = self.scaled_w[self.scaled_w["target"] == pos_val]
            neg = self.scaled_w[self.scaled_w["target"] == neg_val]

            if sample_pos and sample_neg:
                if neg.shape[0] < pos.shape[0]:
                    # Duplicate negative values until equal amount as positive.
                    while neg.shape[0] < pos.shape[0]:
                        # Don't want to exceed post size
                        con = neg[:pos.shape[0] - neg.shape[0]] if pos.shape[
                            0] - neg.shape[0] < neg.shape[0] else neg
                        neg = pd.concat([neg, con])
                elif pos.shape[0] < neg.shape[0]:
                    # Duplicate positive values until equal amount as negative.
                    while pos.shape[0] < neg.shape[0]:
                        # Don't want to exceed post size
                        con = pos[:neg.shape[0] - pos.shape[0]] if neg.shape[
                            0] - pos.shape[0] < pos.shape[0] else pos
                        pos = pd.concat([pos, con])

                print("Class balance fixed, Negatives:", neg.shape[0],
                      ", Positives:", pos.shape[0])
                self.scaled_w = pd.concat([pos, neg])

            if sample_pos and not sample_neg:
                self.scaled_w = pos
            if not sample_pos and sample_neg:
                self.scaled_w = neg
            if not pos_val and not neg_val:
                raise Exception(
                    "At least one of sample_pos and sample_neg must be True"
                    "for classification problems.")

        self.targets_w = self.scaled_w["target"]
        self.scaled_w = self.scaled_w.drop(["target"], axis=1)

        # Create structure to store min and max values from scaling.
        min_max = scaler.inverse_transform([[0 for i in self.scaled.columns],
                                            [1 for i in self.scaled.columns]])
        self.min_max_vals = pd.DataFrame(index=["min", "max"])
        for i, col in enumerate(self.scaled.columns):
            self.min_max_vals[col] = [np.round(min_max[0][i], 2),
                                      np.round(min_max[1][i], 2)]

        # If regression, targets are in scaled; need to be removed.
        if problem_type == "reg":
            self.targets = self.scaled["target"]
            self.scaled = self.scaled.drop(["target"], axis=1)

    def density_estimate(self, f, n=50000, k_dens=0.02, n_bins=51):
        # n_bins <= 1/k_dens as that the bucket resolution should not exceed
        # that of the kernel density.
        self.n_bins = n_bins

        kernel = KernelDensity(k_dens).fit(self.scaled_w)

        # Draw random sample from the sample space and store, calculating
        # prediction.
        self.D = pd.DataFrame(kernel.sample(n))
        self.D.columns = self.scaled.columns
        self.D["prediction"] = f(self.D)

        # Add actual prediction, for colouring when plotted in contour.
        self.scaled_w_preds = f(self.scaled_w)

        self.select_bins(n_bins=n_bins)

    def select_bins(self, cols: list=None, n_bins=51):
        if not cols:
            cols = self.D.columns[:-1]
        self.n_bins = n_bins

        # Put values into bins.
        res_vals = np.linspace(0.0, 1.0, n_bins)
        self.D_bins = pd.DataFrame()
        for col in cols:
            tmp = self.D[[col, "prediction"]]
            tmp["bin"] = pd.cut(tmp[col], bins=res_vals)
            tmp = tmp.sort_values(by="bin")
            tmp = tmp.groupby("bin").mean()
            # Select mid value of interval as index.
            self.D_bins[col] = tmp["prediction"]

        # Replace nan values with 0.5 as that is the completely uncertain
        # value.
        self.D_bins = self.D_bins.fillna(0.5)

    # def select_2d_bins(self):

    def scatter_plot_matrix(self, cols: list=None):
        if not cols:
            to_plot = self.scaled
        else:
            to_plot = self.scaled[cols]
        pd.plotting.scatter_matrix(to_plot)
        plt.grid(False)
        plt.show()

    def density_scatter(self, col, figsize=(8, 4)):
        self.D.plot.scatter(x="prediction", y=col, figsize=figsize)
        plt.title(col + " value and certainty")

    def density_contour(self, col):
        self.D.plot.scatter(x="prediction", y=col)
        plt.title(col + " value and certainty")

    def vis_1d(self, figsize=(16, 8), title=None):
        # Shifting everything down 0.5 makes 0 the uncertain value.
        D_mid_bins = copy.deepcopy(self.D_bins)
        for col in D_mid_bins[:-1]:
            D_mid_bins[col] = D_mid_bins[col] - 0.5
        D_mid_bins.plot.bar(xlim=(-0.15, 1.15), ylim=(-0.6, 0.6),
                            title=title, figsize=figsize)
        plt.show()

    def _rescale(self, min_val, max_val, x):
        return (max_val - min_val) * x + min_val

    def _break_text(self, txt):
        ret = txt[:18]
        if len(txt) >= 18:
            ret += ("\n" + txt[18:36])
        if len(txt) >= 36:
            ret += ("\n" + txt[36:54])
        return ret

    def vis_1d_separate(self, title=None):
        rows = self.D_bins.columns
        n_rows = len(rows)
        colors = plt.get_cmap("Spectral")
        fig, axes = plt.subplots(nrows=n_rows, ncols=1,
                                 figsize=(8.0, n_rows * 4.0))
        for i, row in enumerate(rows):
            min_val = self.min_max_vals[row]["min"]
            max_val = self.min_max_vals[row]["max"]

            # Select mid values of intervals for x values.
            bar_vals = self.D_bins[row] - 0.5
            x = [self._rescale(min_val, max_val, i.mid)
                 for i in np.array(bar_vals.keys())]
            y = bar_vals.values
            c = [colors(i+0.5) for i in y]

            axes[i].bar(x=x, height=y,
                        width=(max_val - min_val) / len(x) * 1.05, color=c)
            axes[i].set_ylim(-0.5, 0.5)
            axes[i].set_yticks(np.arange(-0.5, 0.75, 0.25))
            axes[i].set_xlim(min_val, max_val)
            axes[i].set_xticks(np.linspace(min_val, max_val, 5))
            axes[i].grid(False)
            axes[i].set_ylabel(self._break_text(row))

        if title:
            fig.suptitle(title, size="26")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def vis_2d(self, title=None):
        """
        """
        np.random.seed(42)
        colors = plt.get_cmap("Spectral")
        cols = self.D_bins.columns
        n_cols = len(cols)
        fig, axes = plt.subplots(nrows=n_cols+1, ncols=n_cols+1,
                                 figsize=(n_cols * 5.0, n_cols * 4.5))

        # Hide all ticks and labels.
        for ax in axes.flat:
            pass
            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)
            # ax.set_xlim(0.0, 1.0)
            # ax.set_ylim(0.0, 1.0)

        # Calculate the general [0, 1] meshgrid for contours.
        res_sub = np.linspace(0.0, 1.0, self.n_bins - 1)

        # Find mid points for each bin, for selecting index of bin.
        mid_points = np.array([i.mid for i in self.D_bins.index])

        # Finds nearest point of a value in an array.
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        # Plot the data.
        # TODO: Optimise this to not repeat 2D contours.
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            # Don't want to plot outer columns.
            if i < n_cols and j < n_cols:
                for x, y in [(i, j), (j, i)]:
                    min_x_val = self.min_max_vals[cols[x]]["min"]
                    max_x_val = self.min_max_vals[cols[x]]["max"]
                    min_y_val = self.min_max_vals[cols[y]]["min"]
                    max_y_val = self.min_max_vals[cols[y]]["max"]
                    # Unscale limits for axis.
                    Xm, Ym = np.meshgrid(
                        np.linspace(min_x_val, max_x_val, self.n_bins - 1),
                        np.linspace(min_y_val, max_y_val, self.n_bins - 1))
                    # Create an entry matrix.
                    Zm = [[[] for i in res_sub] for j in res_sub]
                    # Go through each prediction based on KDE and put into
                    # nearest bin.
                    for d, d_val in self.D.iterrows():
                        col1_ind = int(find_nearest(mid_points, d_val[
                            cols[x]]) * (self.n_bins - 1) - 0.5)
                        col2_ind = int(find_nearest(mid_points, d_val[
                            cols[y]]) * (self.n_bins - 1) - 0.5)
                        Zm[col1_ind][col2_ind].append(d_val["prediction"])
                    # Take the mean of value, move down to mid at 0, make
                    # outliers 100 so not drawn.
                    Zm = [[np.mean(Zm[i][j]) - 0.5 if len(Zm[i][j]) > 0
                           else 100 for i in range(self.n_bins - 1)] for j in
                          range(self.n_bins - 1)]
                    axes[x, y].contourf(Xm, Ym, Zm, levels=np.linspace(
                        -0.5, 0.5, 41), cmap="Spectral")
                    axes[x, y].scatter(self.df[cols[x]],
                                       self.df[cols[y]], c="#000000",
                                       s=3, marker="o", alpha=0.75, zorder=1)
                    axes[x, y].set_xlim(min_x_val, max_x_val)
                    axes[x, y].set_ylim(min_y_val, max_y_val)
                    axes[x, y].set_xticks(np.linspace(min_x_val, max_x_val, 3))
                    axes[x, y].set_yticks(np.linspace(min_y_val, max_y_val, 3))
                    axes[x, y].grid(False)

        # Add bar charts as charts on bottom and right.
        for i, col in enumerate(cols):
            min_val = self.min_max_vals[col]["min"]
            max_val = self.min_max_vals[col]["max"]

            # Select mid values of intervals for x values.
            bar_vals = self.D_bins[col] - 0.5
            x = [self._rescale(min_val, max_val, i.mid)
                 for i in np.array(bar_vals.keys())]
            y = bar_vals.values
            c = [colors(i+0.5) for i in y]

            # Add density bar charts as bottom row.
            axes[i, n_cols].bar(x=x, height=y, color=c,
                                width=(max_val - min_val) / len(x) * 1.05)
            axes[i, n_cols].set_ylim(-0.5, 0.5)
            axes[i, n_cols].set_yticks(np.arange(-0.5, 0.75, 0.25))
            axes[i, n_cols].set_xlim(min_val, max_val)
            axes[i, n_cols].set_xticks(np.linspace(min_val, max_val, 3))
            axes[i, n_cols].grid(False)
            axes[i, n_cols].yaxis.tick_right()
            # axes[i, n_cols].yaxis.set_label_position("right")
            axes[i, n_cols].set_xlabel(self._break_text(col))

            # Add density bar charts as right row.
            axes[n_cols, i].barh(x, y, color=c,
                                 height=(max_val - min_val) / len(x) * 1.05)
            axes[n_cols, i].set_xlim(-0.5, 0.5)
            axes[n_cols, i].set_xticks(np.arange(-0.5, 0.75, 0.25))
            axes[n_cols, i].set_ylim(min_val, max_val)
            axes[n_cols, i].set_yticks(np.linspace(min_val, max_val, 3))
            axes[n_cols, i].grid(False)
            axes[n_cols, i].set_ylabel(self._break_text(col))

            # Add labels of variables with scaled interval in diagonal.
            # axes[i, i].annotate(self._break_text(col), (0.5, 0.5),
            #                     xycoords='axes fraction', ha='center',
            #                     va='center')
            # axes[i, i].xaxis.set_visible(False)
            # axes[i, i].yaxis.set_visible(False)
            axes[i, i].set_visible(False)
            # axes[i, i].annotate("Mean Certainty:\n" + str(np.round(np.mean(
            #     self.D_bins[col] - 0.5), 3)), (0.5, 0.5),
            #     xycoords='axes fraction', ha='center', va='center')
            axes[i, i].grid(False)

        # Remove diagram in bottom right.
        axes[n_cols, n_cols].set_visible(False)

        if title:
            fig.suptitle(title, size="26")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()