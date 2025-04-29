from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lactopy.lactate_models.base import BaseModel


class Plot:
    """
    Base class for all plots.
    """

    def __init__(self, base_lactate_model: "BaseModel"):
        self.base_lactate_model = base_lactate_model
