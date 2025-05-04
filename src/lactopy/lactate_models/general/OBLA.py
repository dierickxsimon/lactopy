from lactopy.lactate_models.base.base import BaseModel


class OBLA(BaseModel):
    """
    OBLA model for lactate threshold prediction.
    This model uses a 3rd degree polynomial to fit the lactate data.
    """

    def predict(self, lactate_value: float) -> float:
        return self.model.predict_inverse(lactate_value)
