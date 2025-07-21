from numpy.typing import ArrayLike
import numpy as np
import copy
import piecewise_regression

from lactopy.base import BaseModel
from lactopy.plots.lt1_lt2_breakpoint_plot import LT1_LT2_breakpoint_Plot


class LT1_LT2_breakpoint(BaseModel):
    """
    Determining both LT1 and LT2 based on the breakpoints in the lactate curve
    Piecewie regression module is used to identify the breakpoints
    
    Plot is divided into 3 segments --> segmented regression identifies breakpoint


    Attributes:
        plot (lt1_lt2_breakpoint_plot): For visualizing the model.
    """

    def __init__(self):
        """
        Initializes the LT1_LT2_breakpoint model and its associated plot.
        """
        super().__init__()
        self.plot = LT1_LT2_breakpoint_Plot(self)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        si: float = 0.5,
        **kwargs,
    ):
        """
        Fits the LT1_LT2_breakpoint model to the given data.

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
        Predicts intensity for LT1 and LT2 method. 

        Returns:
            float: Predicted intensity.
            
        """
        
        pw_fit = piecewise_regression.Fit(self.X, self.y, n_breakpoints=2)
        pw_results = pw_fit.get_results()
        
        if "breakpoint1" not in pw_results.get("estimates", {}):
            raise ValueError(
                f"'breakpoint1' not found in regression results."
                f"It is recommended to use a different treshold estimation method."
                f"Piecewise regression may have failed. Result: {pw_results}"
            )
        else:
            predicted_lt1 = pw_results["estimates"]["breakpoint1"]["estimate"]
            predicted_lt2 = pw_results["estimates"]["breakpoint2"]["estimate"]

        return predicted_lt1, predicted_lt2
