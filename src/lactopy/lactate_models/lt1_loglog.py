from numpy.typing import ArrayLike
import numpy as np
import piecewise_regression

from lactopy.base import BaseModel
from lactopy.plots.lt1_loglog_plot import LT1_loglog_Plot


class LT1_loglog(BaseModel):
    """
    Log-Log model for lactate threshold estimation.

    Log-Log method is used to estimate the first lactate threshold by plotting
    the lactate response against intensity on a logaritmic scale.

    Plot is divided into 2 segments --> segmented regression identifies breakpoint

    Attributes:
        plot (lt1_loglog_plot): For visualizing the model..
    """

    def __init__(self):
        """
        Initializes the lt1_loglog model and its associated plot.

        """
        super().__init__()
        self.plot = LT1_loglog_Plot(self)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        **kwargs,
    ):
        """
        Fits the lt1_loglog model to the given data.

        Args:
            X (ArrayLike): The independent variable (e.g., intensity).
            y (ArrayLike): The dependent variable (e.g., lactate concentration).
            si: the standard increment, defaults to 0.5

            method (str):
                Method to use for fitting the model. Options are:

                - `"3th_poly"`: 3rd degree polynomial
                - `"4th_poly"`: 4th degree polynomial
                - `"spline"`: Cubic Spline

                Defaults to "4th_poly".

        Returns:
            self: Fitted lt1_loglog model instance.
        """

        y_log = np.log(y)
        x_log = np.log(X)

        super().fit(x_log, y_log, **kwargs)
        return self

    def predict(self) -> float:
        """
        Predicts intensity for the lt1_loglog method.

        Returns:
            float: Predicted intensity.

        """

        pw_fit = piecewise_regression.Fit(self.X, self.y, n_breakpoints=1)
        pw_results = pw_fit.get_results()

        if "breakpoint1" not in pw_results.get("estimates", {}):
            raise ValueError(
                f"'breakpoint1' not found in regression results."
                f"It is recommended to use a different treshold estimation method."
                f"Piecewise regression may have failed. Result: {pw_results}"
            )
        else:
            predicted_x = pw_results["estimates"]["breakpoint1"]["estimate"]

        return predicted_x
