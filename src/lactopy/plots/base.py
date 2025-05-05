from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from lactopy.lactate_models.base import BaseModel


class Plot:
    """
    Base class for all plots.
    """

    def __init__(self, base_lactate_model: "BaseModel"):
        self.base_lactate_model = base_lactate_model

    def __call__(self):
        return self.plot_fit()

    def plot_fit(self):
        """
        Plot the model.
        """
        X = np.linspace(
            self.base_lactate_model.X.min(), self.base_lactate_model.X.max(), 100
        )
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.base_lactate_model.X,
            self.base_lactate_model.y,
            color="blue",
            label="Data",
        )
        plt.plot(
            X, self.base_lactate_model.model.predict(X), color="red", label="Model"
        )
        plt.title("Lactate Model Fit")
        plt.xlabel("Intensity")
        plt.ylabel("Lactate")
        plt.legend()
        return plt.gca()

    def plot_predictions(self, X):
        """
        Plot the model predictions.
        """
        self.plot_fit()
        plt.axvline(
            self.base_lactate_model.predict(X),
            color="black",
            label="Predictions",
            linestyle="--",
        )
        return plt.gca()
