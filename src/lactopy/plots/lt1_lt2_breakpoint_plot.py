from lactopy.plots.base import Plot


class LT1_LT2_breakpoint_Plot(Plot):
    """
    Plot for LT1_LT2 breakpoint model.
    """

    _title = "LT1-LT2 Breakpoint Plot"

    def plot_predictions(self):
        """
        Plot the LT1 and LT2 model predictions.
        """
        ax = self.plot_fit()
        result = self.base_lactate_model.predict()
        ax = Plot.add_threshold_lines(ax, result.lt1, color="black")
        ax = Plot.add_threshold_lines(ax, result.lt2, color="red")
        return ax
