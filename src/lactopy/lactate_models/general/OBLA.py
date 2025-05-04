from lactopy.lactate_models.base.base import BaseModel


class OBLA(BaseModel):
    def predict(self, lactate_value: float) -> float:
        return self.model.predict_inverse(lactate_value)
