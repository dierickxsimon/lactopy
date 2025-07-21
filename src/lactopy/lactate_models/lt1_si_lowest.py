from numpy.typing import ArrayLike
import numpy as np
import copy


from lactopy.base import BaseModel
from lactopy.plots.lt1_si_lowest_plot import LT1_si_lowest_Plot


class LT1_si_lowest(BaseModel):
    """
    Standard increment model for lactate threshold estimation.

    The Standard increment method is used to estimate the first lactate threshold by identifying
    the point in which the first meaningfull lactate increase occurs.
    In LT1_si_lowest the lowest measurment of lactate concentration is used as reference point.
    This means that that the reference measuement is not necessarily equal to the first measurement.

    This is computed by using a standard value, preferably equlual to the lowest 
    detectable change in lactate concentration of the measurement tool

    Attributes:
        plot (lt1_si_lowest_plot): For visualizing the model.
    """

    def __init__(self):
        """
        Initializes the LT1_si_lowest model and its associated plot.

        """
        super().__init__()
        self.plot = LT1_si_lowest_Plot(self)

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
        X = np.asarray(copy.deepcopy(X))
        y = np.asarray(copy.deepcopy(y))

        if si > 0 :
            X, y = copy.deepcopy(X), copy.deepcopy(y)
            min_y_idx = np.argmin(y) # get the index of the lowest y
            self.min_y_idx = min_y_idx
            self.x_at_min_y = X[min_y_idx]
            self.y_lt1 = y[self.min_y_idx] + si
        else:
            raise ValueError(f"Impossible lactate change to find LT1: {si}")

        # Use only values AFTER x_at_min_y for model fitting
        valid_mask = X >= self.x_at_min_y
        if not np.any(valid_mask):
            raise ValueError("No X values greater than X at minimum Y. Cannot fit model.")

        self.X = X[valid_mask]
        self.y = y[valid_mask]

        super().fit(self.X, self.y, **kwargs)
        return self

    def predict(self) -> float:
        """
        Predicts intensity (X) at LT1 based on y_lt1.
        """
        prediction = self.model.predict_inverse(self.y_lt1)

        if prediction <= self.x_at_min_y:
            raise ValueError("Predicted LT1 intensity is not greater than baseline.")

        return prediction
