# -*- coding: utf-8 -*-
"""LDR - init

Classifier Certainty Visualization.
"""
import copy
import typing
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from . import utils


class ProblemType:
    """
    Enum equivalent for describing problem type.

    Starts at 1 so there is no temptation to use boolean logic.
    """
    classification = 1
    regression = 2


def set_style():
    """
    Sets matplotlib style to my ekrekr sheet.
    """
    plt.style.use("ekrekr.mplstyle")


class LDR:
    def __init__(self,
                 df: pd.DataFrame,
                 targets: pd.Series,
                 class_order: typing.List[any] = None,
                 verbose: bool = True,
                 seed: int = None):
        """
        Latent Dimensionality Reduction.

        Prepares data for sampling.

        LDR automatically detects whether it is a classification or a
        regression problem.

        Note that data is min-max scaled.

        n_bins + 1 should be less than 1/k_dens as the bucket resolution should
        not exceed that of the kernel density.

        LDR is applied to the entire dataset. How the classifier is trained is
        done at the discretion of the user.

        Args:
            df: The data (excluding targets).
            targets: The targets of the model.
            class_order: Order to display density. Only valid if class problem.
            self.verbose: Whether to print anything out.
            seed: Seed to set for numpy, if not none.
        """
        if seed:
            np.random.seed(0)

        self.df = df
        self.targets = np.array(targets)
        # The list of unique values is the order of classes if not specified.
        self.class_order = np.array(class_order or list(set(self.targets)))
        self.verbose = verbose

        # These are variables that get assigned during density estimation.
        self.func = None
        self.n_samples = None
        self.k_dens = None
        self.n_bins = None
        self.samples = None
        self.class_bins = None
        self.feature_bins = None

        self.problem_type = self._determine_problem_type()
        self._check_class_order()
        self._check_data_numerical()

        scaler = MinMaxScaler()
        scale = scaler.fit_transform(self.df)
        self.scaled = pd.DataFrame(scale)

        # Reshuffle the scaled columns to be same as input; nicer for user.
        self.scaled.columns = self.df.columns

        # Create structure to store min and max values from scaling. This is
        # useful for plotting.
        min_max = scaler.inverse_transform([[0 for i in self.scaled.columns],
                                            [1 for i in self.scaled.columns]])
        self.min_max_vals = pd.DataFrame(index=["min", "max"])
        for i, col in enumerate(self.scaled.columns):
            self.min_max_vals[col] = [np.round(min_max[0][i], 2),
                                      np.round(min_max[1][i], 2)]

        # If regression, targets are in scaled; then need to be removed.
        if self.problem_type == ProblemType.regression:
            self.targets = self.scaled["target"]
            self.scaled = self.scaled.drop(["target"], axis=1)

    def _print(self, msg: str):
        """
        Prints if verbose is flagged, otherwise ignores.

        Args:
            msg: What to print.
        """
        if self.verbose:
            print(msg)

    def _determine_problem_type(self) -> ProblemType:
        """
        Determines the problem type, either classification or regression.

        Returns:
            The type of problem.
        """
        if utils.is_np_numerical(self.targets):
            if self.verbose:
                self._print("Targets indicate a regression problem.")
            return ProblemType.regression
        if self.verbose:
            self._print("Targets indicate a classification problem.")
        return ProblemType.classification

    def _check_class_order(self):
        """
        Checks that class order is valid. If not, raises exception.

        This check includes:

        * Set of targets is equivalent to set of class orders.
        """
        if set(self.targets) != set(self.class_order):
            raise Exception(f"class_order does not contain same unique"
                            f"elements as targets.")

    def _check_data_numerical(self):
        """
        Checks that input data is numerical. If not, converts and warns user.
        """
        columns = self.df.columns
        self.df = self.df.select_dtypes(include=np.number)
        if set(self.df.columns) != set(columns):
            removed_cols = [i for i in columns if i not in self.df.columns]
            print(f"Data not numerical. Removed: {removed_cols}.")

    # def _weight_samples(self):
    #     """
    #     Equally weights samples. Can improve visualizations.
    #     """
    #     # Copy scaled to apply target normalization weightings to without
    #     # modifying original.
    #     self.scaled_w = copy.deepcopy(self.scaled)

    #     self.targets_w = self.scaled_w["target"]
    #     self.scaled_w = self.scaled_w.drop(["target"], axis=1)

    #     self.scaled_w["target"] = self.targets
    #     pos = self.scaled_w[self.scaled_w["target"] == pos_val]
    #     neg = self.scaled_w[self.scaled_w["target"] == neg_val]

    #     if sample_pos and sample_neg:
    #         if neg.shape[0] < pos.shape[0]:
    #             # Duplicate negative values until equal amount as positive.
    #             while neg.shape[0] < pos.shape[0]:
    #                 # Don't want to exceed post size
    #                 con = neg[:pos.shape[0] - neg.shape[0]] if pos.shape[
    #                     0] - neg.shape[0] < neg.shape[0] else neg
    #                 neg = pd.concat([neg, con])
    #         elif pos.shape[0] < neg.shape[0]:
    #             # Duplicate positive values until equal amount as negative.
    #             while pos.shape[0] < neg.shape[0]:
    #                 # Don't want to exceed post size
    #                 con = pos[:neg.shape[0] - pos.shape[0]] if neg.shape[
    #                     0] - pos.shape[0] < pos.shape[0] else pos
    #                 pos = pd.concat([pos, con])

    #         print("Class balance fixed, Negatives:", neg.shape[0],
    #                 ", Positives:", pos.shape[0])
    #         self.scaled_w = pd.concat([pos, neg])

    #     if sample_pos and not sample_neg:
    #         self.scaled_w = pos
    #     if not sample_pos and sample_neg:
    #         self.scaled_w = neg
    #     if not pos_val and not neg_val:
    #         raise Exception(
    #             "At least one of sample_pos and sample_neg must be True"
    #             "for classification problems.")

    def density_estimate(self,
                         func: any,
                         classes: list,
                         n_samples: int = 50000,
                         k_dens: float = 0.02,
                         n_bins: int = 51):
        """
        Draws samples from the kernel density of the sample space, then bin.

        Bin here means put into categories depending on values.

        self.D is the set of samples.

        Func should return a matrix of certainties, even if one hot encoded.
        For example:
        [[0.1, 0.1, 0.8],
        [0.4, 0.3, 0.3],
        ...
        [0.0, 0.1, 0.9]]
        Each certainty here should correspond to the class described in
        `classes`, so if `classes[0] == 'queequeg'`, then the certainty of the
        first item of being queequeg is 0.1.

        Args:
            func: The model predictor, described as a function.
            classes: Classes corresponding to the func.
            n_samples: The number of samples to draw.
            k_dens: The bandwidth of the KDE.
            n_bins: The resolution of the binning, post sampling.
        """
        self.func = func
        self.n_samples = n_samples
        self.k_dens = k_dens
        self.n_bins = n_bins

        kernel = KernelDensity(self.k_dens).fit(self.scaled)

        # Draw random sample from the sample space and store, calculating
        # prediction.
        self.samples = pd.DataFrame(kernel.sample(self.n_samples))
        self.samples.columns = self.scaled.columns
        predictions = self.func(self.samples)

        self.class_bins = {}
        for i_class, v_class in enumerate(classes):
            class_preds = np.array([i[i_class] for i in predictions])
            self.class_bins[v_class] = self._bin_values(class_preds)

        # Feature bins make plotting easier in some cases.
        # Instead of having a dict of classes with DataFrames of features,
        # it is a dict of features with DataFrames of classes.
        self.feature_bins = {}
        for key, v_class in self.class_bins.items():
            for feature in v_class.columns:
                if feature not in self.feature_bins:
                    self.feature_bins[feature] = {}
                self.feature_bins[feature][key] = v_class[feature]
        for key in self.feature_bins:
            self.feature_bins[key] = pd.DataFrame(self.feature_bins[key])
        

    def _bin_values(self,
                    predictions: np.array) -> pd.DataFrame:
        """
        Puts values into bins for prediction certainty of a single feature.

        Extensive used of pandas/numpy here takes advantage of their Cython
        optimization, dramatically reducing run time.

        Args:
            predictions: Prediction certainty of the feature being binned.

        Returns:
            Binned samples.
        """
        # Select colums other than prediction for binning.
        cols = self.samples.columns

        # Inbuilt pandas functionality useful for binning, so temporarily asign
        # column.
        self.samples["prediction"] = predictions

        # Binning between 0.0 and 1.0 means between the minimum and maximum
        # values of the data.
        res_vals = np.linspace(0.0, 1.0, self.n_bins)

        class_bins = pd.DataFrame()
        for col in cols:
            tmp = self.samples[[col, "prediction"]]

            # Disable warning here for "A value is trying to be set on a copy
            # of a slice from a DataFrame." as it is a false positive; slice is
            # not a copy.
            pd.options.mode.chained_assignment = None
            tmp["bin"] = pd.cut(tmp[col], bins=res_vals)

            tmp = tmp.sort_values(by="bin")
            tmp = tmp.groupby("bin").mean()

            # Select mid value of interval as index.
            class_bins[col] = tmp["prediction"]

        # Replace nan values with 0 as that is the completely uncertain
        # value. Nan values occur where no values are present, as they are not
        # in the sample space.
        class_bins = class_bins.fillna(0.0)

        # Remove additional prediction column.
        self.samples = self.samples.drop(["prediction"], axis=1)

        return class_bins

    def certainty_plots(self):
        """
        Plots 1D certainty plots for different features.

        For each feature a line graph is drawn, and for each graph each of the
        different class certainties across the data is drawn.
        """
        # print("bins:", self.feature_bins)
        for feature in self.feature_bins.keys():
            lines = self.feature_bins[feature].plot.line()
            plt.show()


    # def density_scatter(self,
    #                     col: str,
    #                     figsize: typing.Tuple[int, int]=(8, 4),
    #                     save=None,
    #                     title=None):
    #     """
    #     Plots provided model applied to samples of certainty.

    #     Args:
    #         col: Column to plot.
    #         figsize: Size of the figure (matplotlib).
    #         save: Whether to save the file.
    #         title: Title to give the figure, if not None.
    #     """
    #     self.D.plot.scatter(x="prediction", y=col, figsize=figsize,
    #                         s=1)
    #     if title:
    #         plt.title(title)
    #     if save:
    #         plt.savefig(save)
    #     plt.show()


    # # def select_2d_bins(self):

    # def scatter_plot_matrix(self,
    #                         cols: list=None):
    #     """
    #     Creates scatterplot matrix of features.

    #     Args:
    #         cols: If None then all columns, otherwise subset specified.
    #     """
    #     if not cols:
    #         to_plot = self.scaled
    #     else:
    #         to_plot = self.scaled[cols]
    #     pd.plotting.scatter_matrix(to_plot)
    #     plt.grid(False)
    #     plt.show()

    # def density_contour(self,
    #                     col: str):
    #     """
    #     Plots the density contour of a column (feature).

    #     Args:
    #         col: The column to plot the contour of.
    #     """
    #     self.D.plot.scatter(x="prediction", y=col)
    #     plt.title(col + " value and certainty")

    # def vis_1d(self,
    #            figsize: typing.Tuple[float, float]=(16, 8),
    #            title=None):
    #     """
    #     Visualizes effect of singular feature on the data.

    #     Args:
    #         figsize: Size of figure (matplotlib).
    #         title: Title to give the figure, if not None.
    #     """
    #     # Shifting everything down 0.5 makes 0 the uncertain value.
    #     D_mid_bins = copy.deepcopy(self.D_bins)
    #     for col in D_mid_bins[:-1]:
    #         D_mid_bins[col] = D_mid_bins[col] - 0.5
    #     D_mid_bins.plot.bar(xlim=(-0.15, 1.15), ylim=(-0.6, 0.6),
    #                         title=title, figsize=figsize)
    #     plt.show()

    # def _rescale(self,
    #              min_val: float,
    #              max_val: float,
    #              x: float):
    #     """
    #     Rescales variables back from unit interval.

    #     Args:
    #         min_val: Value that 0 would be converted from.
    #         max_val: Value that 1 would be converted from.
    #         x: The value to convert.
    #     """
    #     return (max_val - min_val) * x + min_val

    # def _break_text(self,
    #                 txt: str):
    #     """
    #     Breaks text with new line every 18 chars. Max 3 lines.

    #     Args:
    #         txt: The text to break.
    #     """
    #     ret = txt[:18]
    #     if len(txt) >= 18:
    #         ret += ("\n" + txt[18:36])
    #     if len(txt) >= 36:
    #         ret += ("\n" + txt[36:54])
    #     return ret

    # def vis_1d_separate(self,
    #                     title: str=None,
    #                     save: str=None):
    #     """
    #     Visualizes individual effect of all selected features on the data.

    #     Plots each axes in a roughly square format.

    #     Args:
    #         title: Title to give plot, if not None.
    #         save: Path to save file, if not None.
    #     """
    #     features = self.D_bins.columns
    #     n_features = len(features)
    #     n_rows = int(np.sqrt(n_features))
    #     n_cols = int(np.ceil(n_features / n_rows))
    #     colors = plt.get_cmap("Spectral")
    #     fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
    #                              figsize=(5.0 * n_cols, n_rows * 4.0 + 2.0))
    #     for i, feature in enumerate(features):
    #         row = int(i / n_rows)
    #         col = i % n_rows
    #         min_val = self.min_max_vals[feature]["min"]
    #         max_val = self.min_max_vals[feature]["max"]

    #         # Select mid values of intervals for x values.
    #         bar_vals = self.D_bins[feature] - 0.5
    #         x = [self._rescale(min_val, max_val, i.mid)
    #              for i in np.array(bar_vals.keys())]
    #         y = bar_vals.values
    #         c = [colors(i+0.5) for i in y]

    #         axes[col, row].bar(x=x, height=y,
    #                            width=(max_val - min_val) / len(x) * 1.05,
    #                            color=c)
    #         axes[col, row].set_ylim(-0.5, 0.5)
    #         axes[col, row].set_yticks(np.arange(-0.5, 0.75, 0.25))
    #         axes[col, row].set_xlim(min_val, max_val)
    #         axes[col, row].set_xticks(np.linspace(min_val, max_val, 5))
    #         axes[col, row].grid(False)
    #         axes[col, row].set_xlabel(self._break_text(feature))

    #     # Hide axes which have nothing plotted.
    #     for i in range(n_features, n_rows * n_cols):
    #         row = int(i / n_rows)
    #         col = i % n_rows
    #         print(f"Hiding row: {row}, col: {col}")
    #         axes[col, row].set_visible(False)

    #     if title:
    #         fig.suptitle(title, size="26")
    #     fig.tight_layout(rect=[0, 0, 1, 0.96])

    #     if save:
    #         plt.savefig(save)
    #     plt.show()

    # def vis_2d(self,
    #            title: str=None,
    #            save: str=None,
    #            dots: bool=True):
    #     """
    #     Visualizes individual effects, 2D cross effects of selected features.

    #     Args:
    #         title: Title to give plot, if not None.
    #         save: Path to save file, if not None.
    #         dots: Whether to draw dots of VEGAS sampling.
    #     """
    #     np.random.seed(42)
    #     colors = plt.get_cmap("Spectral")
    #     cols = self.D_bins.columns
    #     n_cols = len(cols)
    #     fig, axes = plt.subplots(nrows=n_cols, ncols=n_cols,
    #                              figsize=(n_cols * 5.0, n_cols * 4.5))

    #     # Calculate the general [0, 1] meshgrid for contours.
    #     res_sub = np.linspace(0.0, 1.0, self.n_bins - 1)

    #     # Find mid points for each bin, for selecting index of bin.
    #     mid_points = np.array([i.mid for i in self.D_bins.index])

    #     # Finds nearest point of a value in an array.
    #     def find_nearest(array, value):
    #         array = np.asarray(array)
    #         idx = (np.abs(array - value)).argmin()
    #         return array[idx]

    #     # Get coords for different sections of the triangle.
    #     top_left_co = [(n_cols - (x + 1), y + 1)
    #                    for x, y in zip(*np.tril_indices(n_cols - 1))]
    #     bot_right_co = [(x + 2, n_cols - y - 1)
    #                     for x, y in zip(*np.tril_indices(n_cols - 2))]
    #     left_row = [(0, i + 1) for i in range(n_cols - 1)]
    #     top_row = [(i + 1, 0) for i in range(n_cols - 1)]
    #     print("TL:", top_left_co)
    #     print("BR:", bot_right_co)
    #     print("LR:", left_row)
    #     print("TR:", top_row)

    #     # Plot contours in top right triangle coordinates.
    #     for x, y in top_left_co:
    #         print(x, y)
    #         x_col = cols[x - 1]
    #         y_col = cols[n_cols - y]
    #         print(x_col, y_col)
    #         min_x_val = self.min_max_vals[x_col]["min"]
    #         max_x_val = self.min_max_vals[x_col]["max"]
    #         min_y_val = self.min_max_vals[y_col]["min"]
    #         max_y_val = self.min_max_vals[y_col]["max"]
    #         # Unscale limits for axis.
    #         Xm, Ym = np.meshgrid(
    #             np.linspace(min_x_val, max_x_val, self.n_bins - 1),
    #             np.linspace(min_y_val, max_y_val, self.n_bins - 1))
    #         # Create an entry matrix.
    #         Zm = [[[] for i in res_sub] for j in res_sub]
    #         # Go through each prediction based on KDE and put into
    #         # nearest bin.
    #         for d, d_val in self.D.iterrows():
    #             col1_ind = int(find_nearest(mid_points, d_val[
    #                 x_col]) * (self.n_bins - 1) - 0.5)
    #             col2_ind = int(find_nearest(mid_points, d_val[
    #                 y_col]) * (self.n_bins - 1) - 0.5)
    #             Zm[col1_ind][col2_ind].append(d_val["prediction"])
    #         # Take the mean of value, move down to mid at 0, make
    #         # outliers 100 so not drawn.
    #         Zm = [[np.mean(Zm[i][j]) - 0.5 if len(Zm[i][j]) > 0
    #               else 100 for i in range(self.n_bins - 1)] for j in
    #               range(self.n_bins - 1)]
    #         axes[x, y].contourf(Xm, Ym, Zm, levels=np.linspace(
    #             -0.5, 0.5, 41), cmap="Spectral")
    #         if dots:
    #             axes[x, y].scatter(self.df[x_col],
    #                                self.df[y_col], c="#000000",
    #                                s=3, marker="o", alpha=0.2, zorder=1)
    #         axes[x, y].set_xlim(min_x_val, max_x_val)
    #         axes[x, y].set_ylim(min_y_val, max_y_val)
    #         axes[x, y].set_xticks(np.linspace(min_x_val, max_x_val, 3))
    #         axes[x, y].set_yticks(np.linspace(min_y_val, max_y_val, 3))
    #         axes[x, y].yaxis.tick_right()
    #         axes[x, y].yaxis.set_label_position("right")
    #         axes[x, y].grid(False)

    #     # Add bar charts as charts on top row.
    #     for x, y in top_row:
    #         print(x, y)
    #         col = cols[x - 1]
    #         print(col)
    #         min_val = self.min_max_vals[col]["min"]
    #         max_val = self.min_max_vals[col]["max"]

    #         # Select mid values of intervals for x values.
    #         bar_vals = self.D_bins[col] - 0.5
    #         x_vals = [self._rescale(min_val, max_val, i.mid)
    #                   for i in np.array(bar_vals.keys())]
    #         y_vals = bar_vals.values
    #         c = [colors(i+0.5) for i in y_vals]

    #         axes[x, y].bar(x=x_vals, height=y_vals, color=c,
    #                        width=(max_val - min_val) / len(x_vals) * 1.05)
    #         axes[x, y].set_ylim(-0.5, 0.5)
    #         axes[x, y].set_yticks(np.arange(-0.5, 0.75, 0.25))
    #         axes[x, y].set_xlim(min_val, max_val)
    #         axes[x, y].set_xticks(np.linspace(min_val, max_val, 3))
    #         axes[x, y].grid(False)
    #         # axes[i, y].yaxis.set_label_position("right")
    #         axes[x, y].set_xlabel(self._break_text(col), fontsize=18)

    #     # Add bar charts on left row.
    #     for x, y in left_row:
    #         print(x, y)
    #         col = cols[n_cols - y]
    #         print(col)
    #         min_val = self.min_max_vals[col]["min"]
    #         max_val = self.min_max_vals[col]["max"]

    #         # Select mid values of intervals for x values.
    #         bar_vals = self.D_bins[col] - 0.5
    #         x_vals = [self._rescale(min_val, max_val, i.mid)
    #                   for i in np.array(bar_vals.keys())]
    #         y_vals = bar_vals.values
    #         c = [colors(i+0.5) for i in y_vals]

    #         # Add density bar charts as right row.
    #         axes[x, y].barh(x_vals, y_vals, color=c,
    #                         height=(max_val - min_val) / len(x_vals) * 1.05)
    #         axes[x, y].set_xlim(-0.5, 0.5)
    #         axes[x, y].set_xticks(np.arange(-0.5, 0.75, 0.25))
    #         axes[x, y].set_ylim(min_val, max_val)
    #         axes[x, y].set_yticks(np.linspace(min_val, max_val, 3))
    #         axes[x, y].yaxis.tick_right()
    #         axes[x, y].yaxis.set_label_position("right")
    #         axes[x, y].grid(False)
    #         axes[x, y].set_ylabel(self._break_text(col), fontsize=18)

    #     # Remove subplot in bottom right triangle.
    #     for x, y in bot_right_co:
    #         axes[x, y].set_visible(False)

    #     # Remove subplot in top left.
    #     axes[0, 0].set_visible(False)

    #     if title:
    #         fig.suptitle(title, size="26")
    #     fig.tight_layout(rect=[0, 0, 1, 0.96])

    #     if save:
    #         plt.savefig(save)

    #     plt.show()