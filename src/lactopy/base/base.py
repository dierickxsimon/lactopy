from numpy.typing import ArrayLike
from typing import Union, Self

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


from lactopy.base.adaptors import PolyAdaptor, CubicAdaptor, ExpAdaptor
from lactopy.plots.base import Plot


class BaseModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        """
        Atributes:

            model (object):
                Model object.
            plot (object):
                Plot object.

        """
        self.model = None
        self.plot = Plot(self)

    def _validate_lactate_test(self, X: ArrayLike, y: ArrayLike) -> None:
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if len(X) == 0:
            raise ValueError("X and y must not be empty.")

    def fit(self, X: ArrayLike, y: ArrayLike, method: str = "4th_poly") -> Self:
        """
        Fit the model to the training data.

        Args:
            X (array-like):
                Intensity data.
            y (array-like):
                lactate values.
            method (str):
                Method to use for fitting the model. Options are:

                - `"3th_poly"`: 3rd degree polynomial
                - `"4th_poly"`: 4th degree polynomial
                - `"spline"`: Cubic Spline

        Returns:
            self (object):
                Fitted model.
        """
        self._validate_lactate_test(X, y)
        self.X = np.array(X)
        self.y = np.array(y)
        match method:
            case "3th_poly":
                self.model = PolyAdaptor().fit(self.X, self.y, degree=3)
            case "4th_poly":
                self.model = PolyAdaptor().fit(self.X, self.y, degree=4)
            case "spline":
                self.model = CubicAdaptor().fit(self.X, self.y)
            case "exp":
                self.model = ExpAdaptor().fit(self.X, self.y)
            case _:
                raise ValueError(f"Unknown method: {method}")
        return self

    def predict(self, X: Union[ArrayLike, float, int]):
        raise NotImplementedError("Subclasses should implement this method.")
