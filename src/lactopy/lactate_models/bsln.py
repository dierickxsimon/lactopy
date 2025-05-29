from lactopy.lactate_models.obla import OBLA


class Bsln(OBLA):
    def predict(self, bsln_value: float) -> float:
        return super().predict(bsln_value + self.y.min())
