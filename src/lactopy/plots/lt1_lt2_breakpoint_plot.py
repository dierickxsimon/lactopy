import matplotlib.pyplot as plt
import numpy as np
from lactopy.plots.base import Plot


class LT1_LT2_breakpoint_Plot(Plot):
    """
    Plot for LT1_LT2 breakpoint model.
    """

    def __init__(self, base_lactate_LT1_LT2_breakpoint_model):
        super().__init__(base_lactate_LT1_LT2_breakpoint_model)

    def plot_fit(self):
        """
        Plot the LT1_LT2_breakpoint model fit.
        """
        ax = super().plot_fit()
        ax.set_title("LT1-LT2 Breakpoint Model Fit")
        return ax

    def plot_predictions(self):
        """
        Plot the LT1 and LT2 model predictions.
        """
        self.plot_fit()
        lt_array = self.base_lactate_model.predict()
        lt1 = lt_array[0]
        lt2 = lt_array[1]
        
        plt.axvline(
            lt1,
            color="black",
            label="lt1",
            linestyle="--",
        )
        
        plt.axvline(
            lt2,
            color="red",
            label="lt2",
            linestyle="--",
        )

        return plt.gca()
