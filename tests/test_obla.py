import pytest
import numpy as np

from lactopy.lactate_models.general.OBLA import OBLA
from lactopy.lactate_models.second_threshold.dmax import Dmax


@pytest.mark.parametrize(
    "method, lactate, expected_value",
    [
        ("3th_poly", 2, 104.6),  # values obtained from physlab(ma zijn dus verkeerd)
        ("3th_poly", 3.5, 137),
        ("4th_poly", 3.5, 136.9),
        ("4th_poly", 4, 145.1),
        ("spline", 3.5, 138.6),
    ],
)
def test_obla_model(input_data, method, lactate, expected_value):
    lactate_array, intensity_array = input_data
    obla_model = OBLA()
    predicted_lactate = obla_model.fit(
        intensity_array, lactate_array, method=method
    ).predict(lactate)
    assert isinstance(predicted_lactate, float)
    assert np.isclose(
        predicted_lactate, expected_value, atol=3
    )  # is nodig want exphyslab suckt


@pytest.mark.parametrize(
    "method, expected_value",
    [
        ("3th_poly", 131.1),  # values obtained from physlab(ma zijn dus verkeerd)
        ("4th_poly", 131.1),
        ("spline", 131.1),
    ],
)
def test_dmax_model(input_data, method, expected_value):
    lactate_array, intensity_array = input_data
    dmax_model = Dmax()
    predicted_lactate = dmax_model.fit(
        intensity_array, lactate_array, method=method
    ).predict()
    assert isinstance(predicted_lactate, float)
    assert np.isclose(predicted_lactate, expected_value, atol=3)
