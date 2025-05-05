from lactopy.lactate_models.base import BaseModel


class Dmax(BaseModel):
    """
    Dmax model for lactate threshold estimation.
    """

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
        return self.model.dxdt().predict_inverse(dxdt_first_last_value)
