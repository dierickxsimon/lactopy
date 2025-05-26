from lactopy.base import BaseModel
from lactopy.plots.Dmax import DmaxPlot


class Dmax(BaseModel):
    """
    Dmax model for lactate threshold estimation.
    """

    def __init__(self):
        super().__init__()
        self.plot = DmaxPlot(self)

    def fit(self, *args, **kwargs):
        rv = super().fit(*args, **kwargs)
        self.dxdt_model = self.model.dxdt()
        return rv

    def predict(self) -> float:
        """
        Predicts intensity for the Dmax method.

        Returns:
            float:
                Predicted intensity.
        """

        dxdt_first_last_value = (self.y.max() - self.y.min()) / (
            self.X.max() - self.X.min()
        )
        return self.dxdt_model.predict_inverse(dxdt_first_last_value)
