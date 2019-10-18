# -*- coding: utf-8 -*-
"""LDR - init

Classifier Certainty Visualization.
"""
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import copy
import typing
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)
# Disable warning where bin cut is set (false positive).
pd.options.mode.chained_assignment = None


class LDR:
    def __init__(self,
                 df: pd.DataFrame,
                 targets: pd.Series,
                 problem_type: str="class",
                 pos_val: any=None,
                 neg_val: any=None,
                 sample_pos: bool=True,
                 sample_neg: bool=True):
        """
        Latent Dimensionality Reduction.

        Note: Currently only supports binary classification.

        Args:
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

    def density_estimate(self,
                         f: any,
                         n: int=50000,
                         k_dens: float=0.02,
                         n_bins: int=51):
        """
        Draws samples from the kernel density of the sample space.

        Args:
            f: The model predictor, described as a function.
            n: The number of samples to draw.
            k_dens: The bandwidth of the KDE.
            n_bins: The resolution of the binning post sampling.
        """
        # n_bins + 1 <= 1/k_dens as that the bucket resolution should not
        # exceed that of the kernel density.
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

    def select_bins(self,
                    cols: typing.List[str]=None,
                    n_bins=51):
        """
        Groups samples into bins.

        Args:
            cols: If None then all columns, otherwise use subset specified.
            n_bins: The resolution of the bins (num of bins in unit interval).
        """
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

    def scatter_plot_matrix(self,
                            cols: list=None):
        """
        Creates scatterplot matrix of features.

        Args:
            cols: If None then all columns, otherwise subset specified.
        """
        if not cols:
            to_plot = self.scaled
        else:
            to_plot = self.scaled[cols]
        pd.plotting.scatter_matrix(to_plot)
        plt.grid(False)
        plt.show()

    def density_scatter(self,
                        col: str,
                        figsize: typing.Tuple[int, int]=(8, 4),
                        save=None,
                        title=None):
        """
        Plots provided model applied to samples of certainty.

        Args:
            col: Column to plot.
            figsize: Size of the figure (matplotlib).
            save: Whether to save the file.
            title: Title to give the figure, if not None.
        """
        self.D.plot.scatter(x="prediction", y=col, figsize=figsize,
                            s=1)
        if title:
            plt.title(title)
        if save:
            plt.savefig(save)
        plt.show()

    def density_contour(self,
                        col: str):
        """
        Plots the density contour of a column (feature).

        Args:
            col: The column to plot the contour of.
        """
        self.D.plot.scatter(x="prediction", y=col)
        plt.title(col + " value and certainty")

    def vis_1d(self,
               figsize: typing.Tuple[float, float]=(16, 8),
               title=None):
        """
        Visualizes effect of singular feature on the data.

        Args:
            figsize: Size of figure (matplotlib).
            title: Title to give the figure, if not None.
        """
        # Shifting everything down 0.5 makes 0 the uncertain value.
        D_mid_bins = copy.deepcopy(self.D_bins)
        for col in D_mid_bins[:-1]:
            D_mid_bins[col] = D_mid_bins[col] - 0.5
        D_mid_bins.plot.bar(xlim=(-0.15, 1.15), ylim=(-0.6, 0.6),
                            title=title, figsize=figsize)
        plt.show()

    def _rescale(self,
                 min_val: float,
                 max_val: float,
                 x: float):
        """
        Rescales variables back from unit interval.

        Args:
            min_val: Value that 0 would be converted from.
            max_val: Value that 1 would be converted from.
            x: The value to convert.
        """
        return (max_val - min_val) * x + min_val

    def _break_text(self,
                    txt: str):
        """
        Breaks text with new line every 18 chars. Max 3 lines.

        Args:
            txt: The text to break.
        """
        ret = txt[:18]
        if len(txt) >= 18:
            ret += ("\n" + txt[18:36])
        if len(txt) >= 36:
            ret += ("\n" + txt[36:54])
        return ret

    def vis_1d_separate(self,
                        title: str=None,
                        save: str=None):
        """
        Visualizes individual effect of all selected features on the data.

        Plots each axes in a roughly square format.

        Args:
            title: Title to give plot, if not None.
            save: Path to save file, if not None.
        """
        features = self.D_bins.columns
        n_features = len(features)
        n_rows = int(np.sqrt(n_features))
        n_cols = int(np.ceil(n_features / n_rows))
        colors = plt.get_cmap("Spectral")
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                                 figsize=(5.0 * n_cols, n_rows * 4.0 + 2.0))
        for i, feature in enumerate(features):
            row = int(i / n_rows)
            col = i % n_rows
            min_val = self.min_max_vals[feature]["min"]
            max_val = self.min_max_vals[feature]["max"]

            # Select mid values of intervals for x values.
            bar_vals = self.D_bins[feature] - 0.5
            x = [self._rescale(min_val, max_val, i.mid)
                 for i in np.array(bar_vals.keys())]
            y = bar_vals.values
            c = [colors(i+0.5) for i in y]

            axes[col, row].bar(x=x, height=y,
                               width=(max_val - min_val) / len(x) * 1.05,
                               color=c)
            axes[col, row].set_ylim(-0.5, 0.5)
            axes[col, row].set_yticks(np.arange(-0.5, 0.75, 0.25))
            axes[col, row].set_xlim(min_val, max_val)
            axes[col, row].set_xticks(np.linspace(min_val, max_val, 5))
            axes[col, row].grid(False)
            axes[col, row].set_xlabel(self._break_text(feature))

        # Hide axes which have nothing plotted.
        for i in range(n_features, n_rows * n_cols):
            row = int(i / n_rows)
            col = i % n_rows
            print(f"Hiding row: {row}, col: {col}")
            axes[col, row].set_visible(False)

        if title:
            fig.suptitle(title, size="26")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            plt.savefig(save)
        plt.show()

    def vis_2d(self,
               title: str=None,
               save: str=None,
               dots: bool=True):
        """
        Visualizes individual effects, 2D cross effects of selected features.

        Args:
            title: Title to give plot, if not None.
            save: Path to save file, if not None.
            dots: Whether to draw dots of VEGAS sampling.
        """
        np.random.seed(42)
        colors = plt.get_cmap("Spectral")
        cols = self.D_bins.columns
        n_cols = len(cols)
        fig, axes = plt.subplots(nrows=n_cols, ncols=n_cols,
                                 figsize=(n_cols * 5.0, n_cols * 4.5))

        # Calculate the general [0, 1] meshgrid for contours.
        res_sub = np.linspace(0.0, 1.0, self.n_bins - 1)

        # Find mid points for each bin, for selecting index of bin.
        mid_points = np.array([i.mid for i in self.D_bins.index])

        # Finds nearest point of a value in an array.
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        # Get coords for different sections of the triangle.
        top_left_co = [(n_cols - (x + 1), y + 1)
                       for x, y in zip(*np.tril_indices(n_cols - 1))]
        bot_right_co = [(x + 2, n_cols - y - 1)
                        for x, y in zip(*np.tril_indices(n_cols - 2))]
        left_row = [(0, i + 1) for i in range(n_cols - 1)]
        top_row = [(i + 1, 0) for i in range(n_cols - 1)]
        print("TL:", top_left_co)
        print("BR:", bot_right_co)
        print("LR:", left_row)
        print("TR:", top_row)

        # Plot contours in top right triangle coordinates.
        for x, y in top_left_co:
            print(x, y)
            x_col = cols[x - 1]
            y_col = cols[n_cols - y]
            print(x_col, y_col)
            min_x_val = self.min_max_vals[x_col]["min"]
            max_x_val = self.min_max_vals[x_col]["max"]
            min_y_val = self.min_max_vals[y_col]["min"]
            max_y_val = self.min_max_vals[y_col]["max"]
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
                    x_col]) * (self.n_bins - 1) - 0.5)
                col2_ind = int(find_nearest(mid_points, d_val[
                    y_col]) * (self.n_bins - 1) - 0.5)
                Zm[col1_ind][col2_ind].append(d_val["prediction"])
            # Take the mean of value, move down to mid at 0, make
            # outliers 100 so not drawn.
            Zm = [[np.mean(Zm[i][j]) - 0.5 if len(Zm[i][j]) > 0
                  else 100 for i in range(self.n_bins - 1)] for j in
                  range(self.n_bins - 1)]
            axes[x, y].contourf(Xm, Ym, Zm, levels=np.linspace(
                -0.5, 0.5, 41), cmap="Spectral")
            if dots:
                axes[x, y].scatter(self.df[x_col],
                                   self.df[y_col], c="#000000",
                                   s=3, marker="o", alpha=0.2, zorder=1)
            axes[x, y].set_xlim(min_x_val, max_x_val)
            axes[x, y].set_ylim(min_y_val, max_y_val)
            axes[x, y].set_xticks(np.linspace(min_x_val, max_x_val, 3))
            axes[x, y].set_yticks(np.linspace(min_y_val, max_y_val, 3))
            axes[x, y].yaxis.tick_right()
            axes[x, y].yaxis.set_label_position("right")
            axes[x, y].grid(False)

        # Add bar charts as charts on top row.
        for x, y in top_row:
            print(x, y)
            col = cols[x - 1]
            print(col)
            min_val = self.min_max_vals[col]["min"]
            max_val = self.min_max_vals[col]["max"]

            # Select mid values of intervals for x values.
            bar_vals = self.D_bins[col] - 0.5
            x_vals = [self._rescale(min_val, max_val, i.mid)
                      for i in np.array(bar_vals.keys())]
            y_vals = bar_vals.values
            c = [colors(i+0.5) for i in y_vals]

            axes[x, y].bar(x=x_vals, height=y_vals, color=c,
                           width=(max_val - min_val) / len(x_vals) * 1.05)
            axes[x, y].set_ylim(-0.5, 0.5)
            axes[x, y].set_yticks(np.arange(-0.5, 0.75, 0.25))
            axes[x, y].set_xlim(min_val, max_val)
            axes[x, y].set_xticks(np.linspace(min_val, max_val, 3))
            axes[x, y].grid(False)
            # axes[i, y].yaxis.set_label_position("right")
            axes[x, y].set_xlabel(self._break_text(col), fontsize=18)

        # Add bar charts on left row.
        for x, y in left_row:
            print(x, y)
            col = cols[n_cols - y]
            print(col)
            min_val = self.min_max_vals[col]["min"]
            max_val = self.min_max_vals[col]["max"]

            # Select mid values of intervals for x values.
            bar_vals = self.D_bins[col] - 0.5
            x_vals = [self._rescale(min_val, max_val, i.mid)
                      for i in np.array(bar_vals.keys())]
            y_vals = bar_vals.values
            c = [colors(i+0.5) for i in y_vals]

            # Add density bar charts as right row.
            axes[x, y].barh(x_vals, y_vals, color=c,
                            height=(max_val - min_val) / len(x_vals) * 1.05)
            axes[x, y].set_xlim(-0.5, 0.5)
            axes[x, y].set_xticks(np.arange(-0.5, 0.75, 0.25))
            axes[x, y].set_ylim(min_val, max_val)
            axes[x, y].set_yticks(np.linspace(min_val, max_val, 3))
            axes[x, y].yaxis.tick_right()
            axes[x, y].yaxis.set_label_position("right")
            axes[x, y].grid(False)
            axes[x, y].set_ylabel(self._break_text(col), fontsize=18)

        # Remove subplot in bottom right triangle.
        for x, y in bot_right_co:
            axes[x, y].set_visible(False)

        # Remove subplot in top left.
        axes[0, 0].set_visible(False)

        if title:
            fig.suptitle(title, size="26")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            plt.savefig(save)

        plt.show()
