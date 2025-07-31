from lactopy.base.base import BaseModel


class OBLA(BaseModel):
    """
    OBLA model for lactate threshold prediction.
    This model uses a 3rd degree polynomial to fit the lactate data.
    """

    def predict(self, X: float) -> float:
        """
        Predicts intensity at a given lactate value.

        Args:
            lactate_value (float):
                Lactate value to predict intensity for.

        Returns:
            float:
                Predicted intensity.
        """
        return self.model.predict_inverse(X)
