from numpy.typing import ArrayLike
import numpy as np
import copy
import piecewise_regression

from lactopy.base import BaseModel
from lactopy.plots.lt1_loglog_plot import LT1_loglog_Plot


class LT1_loglog(BaseModel):
    """
    Log-Log model for lactate threshold estimation.

    Log-Log method is used to estimate the first lactate threshold by plotting
    the lactate response against intensity on a logaritmic scale.
    
    Plot is divided into 2 segments --> segmented regression identifies brekpoint

    This is computed by using a standard value, preferably equlual to the lowest 
    detectable change in lactate concentration of the measurement tool

    Attributes:
        plot (lt1_si_plot): For visualizing the model..
    """

    def __init__(self):
        """
        Initializes the LT1_si model and its associated plot.

        """
        super().__init__()
        self.plot = LT1_loglog_Plot(self)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        si: float = 0.5,
        **kwargs,
    ):
        """
        Fits the Standard Increment model to the given data.

        Args:
            X (ArrayLike): The independent variable (e.g., intensity).
            y (ArrayLike): The dependent variable (e.g., lactate concentration).
            si: the standard increment, defaults to 0.5
            
            threshold_above_baseline (float, optional): Threshold above baseline
            for filtering in the "modified" implementation. Defaults to 0.5.

            method (str):
                Method to use for fitting the model. Options are:

                - `"3th_poly"`: 3rd degree polynomial
                - `"4th_poly"`: 4th degree polynomial
                - `"spline"`: Cubic Spline

                Defaults to "4th_poly".

        Returns:
            self: Fitted Dmax model instance.
        """
        self._si = si

        # I have no clue if this is the best options
        # feels wrong to me
        self.X_raw_for_plot = X
        self.y_raw_for_plot = y
        
        # Ensure working on copies
        self.X = np.asarray(copy.deepcopy(X))
        self.y = np.asarray(copy.deepcopy(y))
        
        super().fit(X, y, **kwargs)
        return self

    def predict(self) -> float:
        """
        Predicts intensity for the LT1_si method.

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
