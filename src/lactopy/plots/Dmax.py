import matplotlib.pyplot as plt
import numpy as np
from lactopy.plots.base import Plot


class DmaxPlot(Plot):
    """
    Plot for Dmax model.
    """

    def __init__(self, base_lactate_dmax_model):
        super().__init__(base_lactate_dmax_model)

    def plot_fit(self):
        """
        Plot the Dmax model fit.
        """
        ax = super().plot_fit()
        ax.set_title("Dmax Model Fit")
        return ax

    def plot_predictions(self):
        """
        Plot the Dmax model predictions.
        """
        self.plot_fit()
        plt.axvline(
            self.base_lactate_model.predict(),
            color="black",
            label="Predictions",
            linestyle="--",
        )
        X = np.linspace(
            self.base_lactate_model.X.min(),
            self.base_lactate_model.X.max(),
            100,
        )
        Y = np.linspace(
            self.base_lactate_model.y.min(),
            self.base_lactate_model.y.max(),
            100,
        )
        plt.plot(
            X,
            Y,
            color="blue",
            label="Data",
        )

        return plt.gca()
