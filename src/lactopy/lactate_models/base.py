from sklearn.base import BaseEstimator, RegressorMixin
from scipy.interpolate import UnivariateSpline
import numpy as np

from numpy.typing import ArrayLike
from typing import Union

from lactopy.plots.base import Plot


class BaseModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.plot: Plot = Plot(self)
        self.model = None

    def validate_lactate_test(self, X: ArrayLike, y: ArrayLike):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if len(X) == 0:
            raise ValueError("X and y must not be empty.")
        if not all(isinstance(x, (int, float)) for x in X):
            raise ValueError("All elements in X must be numeric.")
        if not all(isinstance(y_val, (int, float)) for y_val in y):
            raise ValueError("All elements in y must be numeric.")

    def fit(self, X: ArrayLike, y: ArrayLike, method="3th_poly"):
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Intensity data.
        y : array-like, shape (n_samples,)
            lactate values.

        Returns
        -------
        self : object
            Fitted model.
        """
        self.validate_lactate_test(X, y)
        X = np.array(X)
        y = np.array(y)
        match method:
            case "3th_poly":
                # Fit a 3rd degree polynomial
                self.model = np.polyfit(X, y, 3)
            case "2th_poly":
                # Fit a 2nd degree polynomial
                self.model = np.polyfit(X, y, 2)
            case "spline":
                # Fit a spline
                self.model = UnivariateSpline(X, y)
            case _:
                raise ValueError(f"Unknown method: {method}")

        return self

    def predict(self, X: Union[ArrayLike, float, int]):
        raise NotImplementedError("Subclasses should implement this method.")
