# -*- coding: utf-8 -*-
"""LDR - init

Generic Model Visualization.
"""
import os
import copy
import typing
import sklearn
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal

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
    path = os.path.join(os.path.dirname(__file__), "resources/ekrekr.mplstyle")
    plt.style.use(path)


class LDR:
    """
    Latent Dimensonality Reduction.
    """

    def __init__(self,
                 data_df: pd.DataFrame,
                 targets: pd.Series,
                 class_order: typing.List[any] = None,
                 verbose: bool = True,
                 seed: int = 0):
        """
        Prepares data for sampling; not done until `density_estimate` called.

        LDR automatically detects whether it is a classification or a
        regression problem.

        Note that data is min-max scaled.

        n_bins + 1 should be less than 1/k_dens as the bucket resolution should
        not exceed that of the kernel density.

        LDR is applied to the entire dataset. How the classifier is trained is
        done at the discretion of the user.

        Args:
            data_df: The data (excluding targets).
            targets: The targets of the model.
            class_order: Order to display density. Only valid if class problem.
            self.verbose: Whether to print anything out.
            seed: Seed to set for numpy, if not none.
        """
        if seed:
            np.random.seed(seed)

        set_style()

        self.data_df = data_df
        self.targets = np.array(targets)
        # The list of unique values is the order of classes if not specified.
        self.class_order = np.array(class_order or list(set(self.targets)))
        self.verbose = verbose
        self.colors = utils.gen_colors(len(self.class_order))

        # These are variables that get assigned during density estimation.
        self.func = None
        self.n_samples = None
        self.k_dens = None
        self.n_bins = None
        self.samples = None
        self.sample_predictions = None
        self.classifications = None
        self.class_bins = None
        self.feature_bins = None
        self.intervals = None
        self.density_estimate_run = False

        self.problem_type = self._determine_problem_type()
        self._check_class_order()
        self._check_data_numerical()

        scaler = MinMaxScaler()
        scale = scaler.fit_transform(self.data_df)
        self.scaled = pd.DataFrame(scale)

        # Reshuffle the scaled columns to be same as input; nicer for user.
        self.scaled.columns = self.data_df.columns

        # Create structure to store min and max values from scaling. This is
        # useful for plotting.
        min_max = scaler.inverse_transform([[0 for i in self.scaled.columns],
                                            [1 for i in self.scaled.columns]])
        self.min_max = pd.DataFrame(index=["min", "max"])
        for i, col in enumerate(self.scaled.columns):
            self.min_max[col] = [np.round(min_max[0][i], 2),
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
            self._print("Targets indicate a regression problem.")
            return ProblemType.regression
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
        columns = self.data_df.columns
        self.data_df = self.data_df.select_dtypes(include=np.number)
        if set(self.data_df.columns) != set(columns):
            removed_cols = [
                i for i in columns if i not in self.data_df.columns]
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

    def _check_density_estimate_run(self):
        """
        Checks that density estimate has been run, otherwise raises exception.
        """
        if self.density_estimate_run is False:
            raise Exception(f"Function called that requires desnity estimate, "
                            f"but has not been run.")

    def density_estimate(self,
                         func: any,
                         classifications: list,
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
        self.classifications = classifications
        self.n_samples = n_samples
        self.k_dens = k_dens
        self.n_bins = n_bins

        kernel = KernelDensity(self.k_dens).fit(self.scaled)

        # Draw random sample from the sample space and store, calculating
        # prediction.
        self.samples = pd.DataFrame(kernel.sample(self.n_samples))
        self.samples.columns = self.scaled.columns
        self.sample_predictions = self.func(self.samples)

        self.class_bins = {}
        for i_class, v_class in enumerate(classifications):
            class_preds = np.array(
                [i[i_class] for i in self.sample_predictions])
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

        self.density_estimate_run = True

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

            # Useful to store intervals.
            self.intervals = tmp.index

        # Replace nan values with 0 as that is the completely uncertain
        # value. Nan values occur where no values are present, as they are not
        # in the sample space.
        class_bins = class_bins.fillna(0.0)

        # Remove additional prediction column, after storing.
        self.samples = self.samples.drop(["prediction"], axis=1)

        return class_bins

    def _rescale(self,
                 min_val: float,
                 max_val: float,
                 val: float):
        """
        Rescales variables back from unit interval.

        Args:
            min_val: Value that 0 would be converted from.
            max_val: Value that 1 would be converted from.
            val: The value to convert.
        """
        return (max_val - min_val) * val + min_val

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

    def _draw_1d_subplot(self,
                         axes: plt.subplot,
                         col: int,
                         row: int,
                         feature: str,
                         classes: list,
                         invert: bool = False):
        """
        Draws 1d graph for a specified feature.

        Args:
            axes: Axes to draw to.
            col: Subplot column.
            row: Subplot row.
            feature: Subplot feature to draw.
            classes: Labels to give as classes to plot.
            invert: If true, draw vertically rather than horizontally.
        """
        min_val = self.min_max[feature]["min"]
        max_val = self.min_max[feature]["max"]

        # Select mid values of intervals for x values on plot.
        inter_vals = [self._rescale(min_val, max_val, i.mid)
                      for i in np.array(self.intervals)]

        if not invert:
            j = 0
            for _, v_class in self.feature_bins[feature].iteritems():
                axes[col, row].plot(inter_vals, v_class.values,
                                    color=self.colors[j])
                j += 1
            axes[col, row].set_ylim(0.0, 1.0)
            # Overshooting the arange causes 1.0 to be visible.
            axes[col, row].set_yticks(np.arange(0.0, 1.25, 0.25))
            axes[col, row].set_xlim(min_val, max_val)
            axes[col, row].set_xticks(np.linspace(min_val, max_val, 5))
            axes[col, row].grid(True)
            axes[col, row].set_xlabel(self._break_text(feature))
            axes[col, row].legend(labels=classes, title="Classes",
                                  loc="upper right")
        else:
            # Draws graph vertically, which is what `invert` means here.
            j = 0
            for _, v_class in self.feature_bins[feature].iteritems():
                axes[col, row].plot(v_class.values, inter_vals,
                                    color=self.colors[j])
                j += 1
            axes[col, row].set_xlim(0.0, 1.0)
            # Overshooting the arange causes 1.0 to be visible.
            axes[col, row].set_xticks(np.arange(0.0, 1.25, 0.25))
            axes[col, row].set_ylim(min_val, max_val)
            axes[col, row].set_yticks(np.linspace(min_val, max_val, 5))
            axes[col, row].grid(True)
            axes[col, row].set_ylabel(self._break_text(feature))
            axes[col, row].legend(labels=classes, title="Classes",
                                  loc="upper right")
            axes[col, row].invert_yaxis()

    def vis_1d_certainty(self,
                         title: str = None,
                         save: str = None,
                         show: bool = False):
        """
        Plots 1D certainty plots for different features.

        For each feature a line graph is drawn, and for each graph each of the
        different class certainties across the data is drawn.

        Density estimate must have been run beforehand.

        Args:
            title: Suptitle to show, if not None.
            save: Path to save file as, if not None.
            show: Calls plt.show() if True.
        """
        self._check_density_estimate_run()

        features = list(self.feature_bins.keys())
        n_features = len(features)
        classes = self.feature_bins[features[0]].columns
        n_rows = int(np.sqrt(n_features))
        n_cols = int(np.ceil(n_features / n_rows))

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(
            n_cols * 5.0, n_rows * 4.0 + 2.0))

        for i, feature in enumerate(features):
            row, col = int(i / n_rows), i % n_rows
            self._draw_1d_subplot(axes, col, row, feature, classes)

        # Hide axes which have nothing plotted.
        for i in range(n_features, n_rows * n_cols):
            row, col = int(i / n_rows), i % n_rows
            axes[col, row].set_visible(False)

        if title:
            fig.suptitle(title, size="26")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            plt.savefig(save)

        if show:
            plt.show()

    def _make_scientific(self,
                         arr: np.array):
        """
        Turns numpy array numbers into scientific format.

        Args:
            arr: Array to convert.

        Returns:
            Array of strings in scientific number format.
        """
        return ["%.2E" % Decimal(i) for i in arr]

    def _bin_2d_values(self,
                       x_col: str,
                       y_col: str):
        """
        Bins VEGAS samples for Monte Carlo estimation for two features.

        Args:
            x_col: A column to compare.
            y_col: Other column to compare.

        Returns:
            Tuple of meshgrid X, Y, and mean color at Z.
        """
        # This is the resolution of the 2d mesh.
        res_sub = np.linspace(0.0, 1.0, self.n_bins - 1)

        # Find mid points for each bin, for selecting index of bin.
        mid_points = np.array([i.mid for i in self.intervals])

        min_max = self.min_max
        min_x, max_x = min_max[x_col]["min"], min_max[x_col]["max"]
        min_y, max_y = min_max[y_col]["min"], min_max[y_col]["max"]

        # Go through each prediction and put into nearest bin for Z values.
        z2d = [[[] for i in res_sub] for j in res_sub]
        for i_sample, sample in self.samples.iterrows():
            col1_ind = int(utils.find_nearest(mid_points, sample[
                x_col]) * (self.n_bins - 1) - 0.5)
            col2_ind = int(utils.find_nearest(mid_points, sample[
                y_col]) * (self.n_bins - 1) - 0.5)
            z2d[col1_ind][col2_ind].append(self.sample_predictions[i_sample])

        iters = range(self.n_bins - 1)
        z2d = np.array([[utils.reduce_colors(
            z2d[i][j], self.colors) for i in iters] for j in iters])

        # print("z2d[0] after:", z2d[0])

        return z2d, min_x, max_x, min_y, max_y

    def _draw_2d_subplot(self,
                         axes: plt.subplot,
                         col: int,
                         row: int,
                         x_feature: str,
                         y_feature: str,
                         classes: list):
        """
        Draws 2d subplot.
        """
        z2d, min_x, max_x, min_y, max_y = self._bin_2d_values(x_feature,
                                                              y_feature)

        axes[col, row].imshow(z2d, extent=(0.0, 1.0, 1.0, 0.0))
        axes[col, row].scatter(self.scaled[x_feature],
                               self.scaled[y_feature], c="#000000",
                               s=3, marker="o", alpha=0.2, zorder=1)
        axes[col, row].set_xlim(0.0, 1.0)
        axes[col, row].set_xticklabels(
            self._make_scientific(np.linspace(min_x, max_x, 5)))
        axes[col, row].set_ylim(0.0, 1.0)
        axes[col, row].set_yticklabels(
            self._make_scientific(np.linspace(min_y, max_y, 5)))
        axes[col, row].yaxis.tick_right()
        axes[col, row].yaxis.set_label_position("right")
        axes[col, row].grid(False)

    def _select_feature_subset(self,
                               features: typing.List[str]):
        """
        Selects a feature subset if requested, otherwise returns all features.

        Raises an exception if requested feature in subset not present in
        features during class initialization.

        Args:
            features: Feature subset if not None, otherwise all features.
        """
        all_features = list(self.feature_bins.keys())

        if not features:
            return all_features

        if any([f not in all_features for f in features]):
            raise Exception(f"Feature subset requested, but subset contains a"
                            f"feature not in original feature set.\n"
                            f"All features: {all_features}\n"
                            f"Requested features: {features}")

        return features

    def vis_2d_certainty(self,
                         title: str = None,
                         save: str = None,
                         features: typing.List[str] = None,
                         show: bool = False):
        """
        Visualizes individual effects, 2D cross effects of selected features.

        The structure of the graphs drawn dynamically scales to follow a
        particular layout. Consider features w, x, y, and z. What would be
        drawn would look like:

        ```
        (  ) (z ) (y ) (x )
        (w ) (wz) (wy) (wx)
        (x ) (xz) (xy) (  )
        (y ) (zy) (  ) (  )
        ```

        * A singular feature (e.g. `(x )`) indicates a line graph showing the
          certainty of each class when varying the value of the feature.

        * A pair of features (e.g. `(xy)`) indicates the cross variation of
          certainty when across two features.

        * An empty feature (e.g. `(  )`) indicates nothing drawn.

        Args:
            title: Suptitle to show, if not None.
            save: Path to save file as, if not None.
            features: If set, only draws features specified.
            show: Calls plt.show() if True.
        """
        features = self._select_feature_subset(features)
        classes = self.feature_bins[features[0]].columns
        n_features = len(features)
        fig, axes = plt.subplots(nrows=n_features, ncols=n_features, figsize=(
            n_features * 5.0, n_features * 4.0 + 2.0))

        # Get coords for different sections of the triangle.
        top_left_co = [(n_features - (x + 1), y + 1)
                       for x, y in zip(*np.tril_indices(n_features - 1))]
        bot_right_co = [(x + 2, n_features - y - 1)
                        for x, y in zip(*np.tril_indices(n_features - 2))]
        left_row = [(0, i + 1) for i in range(n_features - 1)]
        top_row = [(i + 1, 0) for i in range(n_features - 1)]

        # Plot contours in top left triangle coordinates.
        for col, row in top_left_co:
            x_feature = features[col - 1]
            y_feature = features[n_features - row]
            self._draw_2d_subplot(axes, col, row, x_feature, y_feature,
                                  classes)

        # Add bar charts as charts on top row.
        for col, row in top_row:
            feature = features[col - 1]
            self._draw_1d_subplot(axes, col, row, feature, classes)

        for col, row in left_row:
            feature = features[n_features - row]
            self._draw_1d_subplot(axes, col, row, feature,
                                  classes, invert=True)

    #     # Add bar charts on left row.
    #     for x, y in left_row:
    #         print(x, y)
    #         col = cols[n_cols - y]
    #         print(col)
    #         min_val = self.min_max[col]["min"]
    #         max_val = self.min_max[col]["max"]

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

        if show:
            plt.show()
