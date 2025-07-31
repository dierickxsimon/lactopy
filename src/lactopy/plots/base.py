import inspect

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from lactopy.lactate_models.base import BaseModel


class Plot:
    """
    Base class for all plots.
    """

    _title = "Lactate Model Plot"

    def __init__(self, context: "BaseModel"):
        self.base_lactate_model = context

    def __call__(self):
        return self.plot_fit()

    def plot_fit(self):
        """
        Plot the model.
        """
        X = np.linspace(
            self.base_lactate_model.X.min(), self.base_lactate_model.X.max(), 100
        )
        plt.figure(figsize=(10, 6))
        plt.scatter(
            (
                self.base_lactate_model.X_raw_for_plot
                if hasattr(self.base_lactate_model, "X_raw_for_plot")
                else self.base_lactate_model.X
            ),
            (
                self.base_lactate_model.y_raw_for_plot
                if hasattr(self.base_lactate_model, "y_raw_for_plot")
                else self.base_lactate_model.y
            ),
            color="blue",
            label="Data",
        )
        plt.plot(
            X, self.base_lactate_model.model.predict(X), color="red", label="Model"
        )
        plt.title(self.__class__._title)
        plt.xlabel("Intensity")
        plt.ylabel("Lactate")
        plt.legend()
        return plt.gca()

    @classmethod
    def add_threshold_lines(self, ax: plt.Axes, threshold: float, color: str = "black"):
        # TODO: create a seprate builder class for buidling plots (even building )
        """
        Add threshold lines to the plot.

        Args:
            threshold (float): The threshold value.
            color (str): Color of the threshold line.
        """
        ax.axvline(x=threshold, color=color, linestyle="--", label="Threshold")
        ax.legend()
        return ax

    @classmethod
    def add_line(cls, ax: plt.Axes, x: ArrayLike, y: ArrayLike, color: str = "blue"):
        """
        Add a straight line to the plot.

        Args:
            ax (plt.Axes): The axes to plot on.
            x (array-like): The x-coordinates of the line.
            y (array-like): The y-coordinates of the line.
            color (str): Color of the line.
        """
        ax.plot(x, y, color=color, label="Line")
        return ax

    def plot_predictions(self, X=None):
        """
        Plot the model predictions.
        """
        predict_x = (
            self.base_lactate_model.predict(X)
            if "X" in inspect.getfullargspec(self.base_lactate_model.predict).args
            else self.base_lactate_model.predict()
        )
        ax = self.plot_fit()
        Plot.add_threshold_lines(ax, predict_x)
        return ax
