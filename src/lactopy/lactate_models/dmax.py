import numpy as np
from numpy.typing import ArrayLike


from lactopy.base import BaseModel
from lactopy.plots.Dmax import DmaxPlot


class Dmax(BaseModel):
    """
    Dmax model for lactate threshold estimation.

    The Dmax method is used to estimate the lactate threshold by identifying
    the point of maximal perpendicular distance from a straight line connecting
    the first and last points of the lactate curve.

    This is computed by finding the point where the derivative of the lactate
    curve is equal to the average rate of change between the first and last points.

    Attributes:
        plot (DmaxPlot): An instance of the DmaxPlot class for visualizing the model.
    """

    def __init__(self):
        """
        Initializes the Dmax model and its associated plot.

        """
        super().__init__()
        self.plot = DmaxPlot(self)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        impl: str = "normal",
        threshold_above_baseline=0.4,
        **kwargs,
    ):
        """
        Fits the Dmax model to the given data.

        Args:
            X (ArrayLike): The independent variable (e.g., intensity).
            y (ArrayLike): The dependent variable (e.g., lactate concentration).
            impl (str, optional): The implementation type. Options are "normal" or
            "modified".
                Defaults to "normal".
            threshold_above_baseline (float, optional): Threshold above baseline
            for filtering
                in the "modified" implementation. Defaults to 0.4.

            method (str):
                Method to use for fitting the model. Options are:

                - `"3th_poly"`: 3rd degree polynomial
                - `"4th_poly"`: 4th degree polynomial
                - `"spline"`: Cubic Spline

                Defaults to "4th_poly".

        Returns:
            self: Fitted Dmax model instance.
        """
        self._impl = impl
        mask = None

        match impl:
            case "normal":
                pass
            case "modified":
                X, y = np.array(X), np.array(y)
                mask = X > X[0] + threshold_above_baseline
            case _:
                raise ValueError(f"Unknown implementation: {impl}")

        super().fit(X, y, _mask=mask, **kwargs)
        self.dxdt_model = self.model.dxdt()
        return self

    def predict(self) -> float:
        """
        Predicts intensity for the Dmax method.

        Returns:
            float: Predicted intensity.
        """
        dxdt_first_last_value = (self.y[-1] - self.y[0]) / (self.X[-1] - self.X[0])
        return self.dxdt_model.predict_inverse(dxdt_first_last_value)
