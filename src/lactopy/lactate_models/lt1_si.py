from numpy.typing import ArrayLike


from lactopy.base import BaseModel
from lactopy.plots.lt1_si_plot import LT1_si_Plot


class LT1_si(BaseModel):
    """
    Standard increment model for lactate threshold estimation.

    The Standard increment method is used to estimate the first lactate
    threshold by identifying
    the point in which the first meaningfull lactate increase occurs.

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
        self.plot = LT1_si_Plot(self)

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

        if si > 0:
            y_lt1 = y[0] + si
            self.y_lt1 = y_lt1
        else:
            raise ValueError(f"Impossible lactate change to find LT1: {si}")

        self.y = y

        super().fit(X, y, **kwargs)
        return self

    def predict(self) -> float:
        """
        Predicts intensity for the LT1_si method.

        Returns:
            float: Predicted intensity.
        """
        return self.model.predict_inverse(self.y_lt1)
