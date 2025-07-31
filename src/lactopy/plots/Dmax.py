import numpy as np
from lactopy.plots.base import Plot


class DmaxPlot(Plot):
    """
    Plot for Dmax model.
    """

    _title = "Dmax Model Fit"

    def plot_predictions(self):
        """
        Plot the Dmax model predictions.
        """
        plot = super().plot_predictions()
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
        plot = Plot.add_line(plot, X, Y, color="blue")

        return plot
