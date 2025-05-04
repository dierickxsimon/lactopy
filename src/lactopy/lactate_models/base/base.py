from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

from numpy.typing import ArrayLike
from typing import Union

from lactopy.lactate_models.base.adaptors import PolyAdaptor, CubicAdaptor


class BaseModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = None

    def validate_lactate_test(self, X: ArrayLike, y: ArrayLike):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if len(X) == 0:
            raise ValueError("X and y must not be empty.")

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
        self.X = np.array(X)
        self.y = np.array(y)
        match method:
            case "3th_poly":
                # Fit a 3rd degree polynomial
                self.model = PolyAdaptor().fit(self.X, self.y, degree=3)
            case "4th_poly":
                # Fit a 4th degree polynomial
                self.model = PolyAdaptor().fit(self.X, self.y, degree=4)
            case "spline":
                # Fit a spline
                self.model = CubicAdaptor().fit(self.X, self.y)
            case _:
                raise ValueError(f"Unknown method: {method}")

        return self

    def predict(self, X: Union[ArrayLike, float, int]):
        raise NotImplementedError("Subclasses should implement this method.")
