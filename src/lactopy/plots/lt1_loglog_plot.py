import matplotlib.pyplot as plt
import numpy as np
from lactopy.plots.base import Plot


class LT1_loglog_Plot(Plot):
    """
    Plot for lt1_loglog model.
    """

    def __init__(self, base_lactate_LT1_si_model):
        super().__init__(base_lactate_LT1_si_model)

    def plot_fit(self):
        """
        Plot the lt1_loglog model fit.
        """
        ax = super().plot_fit()
        ax.set_title("LT1 Log-Log Model Fit")
        return ax

    def plot_predictions(self):
        """
        Plot the lt1_loglog predictions.
        """
        self.plot_fit()
        plt.axvline(
            self.base_lactate_model.predict(),
            color="black",
            label="Predictions",
            linestyle="--",
        )

        return plt.gca()
