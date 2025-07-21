import matplotlib.pyplot as plt
import numpy as np
from lactopy.plots.base import Plot


class LT2_breakpoint_Plot(Plot):
    """
    Plot for LT2_breakpoint model.
    """

    def __init__(self, base_lactate_LT1_si_model):
        super().__init__(base_lactate_LT1_si_model)

    def plot_fit(self):
        """
        Plot the LT2_breakpoint model fit.
        """
        ax = super().plot_fit()
        ax.set_title("LT2 Breakpoint Model Fit")
        return ax

    def plot_predictions(self):
        """
        Plot the LT2 model predictions.
        """
        self.plot_fit()
        lt2 = self.base_lactate_model.predict()

        plt.axvline(
            lt2,
            color="black",
            label="lt2",
            linestyle="--",
        )

        return plt.gca()
