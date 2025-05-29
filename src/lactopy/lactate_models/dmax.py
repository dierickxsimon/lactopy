from numpy.typing import ArrayLike
import copy


from lactopy.base import BaseModel
from lactopy.plots.Dmax import DmaxPlot


class Dmax(BaseModel):
    """
    Dmax model for lactate threshold estimation.
    """

    def __init__(self):
        super().__init__()
        self.plot = DmaxPlot(self)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        impl: str = "normal",
        threshold_above_baseline=0.4,
        *args,
        **kwargs,
    ):
        self._impl = impl

        # I have no clue if this is the best options
        # feels wrong to me
        self.X_raw_for_plot = X
        self.y_raw_for_plot = y

        match impl:
            case "normal":
                pass
            case "modified":
                X, y = copy.deepcopy(X), copy.deepcopy(y)
                filter = X > X[0] + threshold_above_baseline
                X = X[filter]
                y = y[filter]
            case _:
                raise ValueError(f"Unknown implementation: {impl}")

        rv = super().fit(X, y, *args, **kwargs)
        self.dxdt_model = self.model.dxdt()
        return rv

    def predict(self) -> float:
        """
        Predicts intensity for the Dmax method.

        Returns:
            float:
                Predicted intensity.
        """
        dxdt_first_last_value = (self.y[-1] - self.y[0]) / (self.X[-1] - self.X[0])
        return self.dxdt_model.predict_inverse(dxdt_first_last_value)
