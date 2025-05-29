import pytest
import numpy as np

from lactopy.lactate_models import OBLA
from lactopy.lactate_models import Bsln
from lactopy.lactate_models import Dmax


@pytest.mark.parametrize(
    "method, lactate, expected_value",
    [
        ("3th_poly", 2, 104.6),  # values obtained from physlab
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
    assert np.isclose(predicted_lactate, expected_value, atol=3)


@pytest.mark.parametrize(
    "method, expected_value",
    [
        ("3th_poly", 131.1),  # values obtained from physlab
        ("4th_poly", 131.1),
        ("spline", 136.99),
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


@pytest.mark.parametrize("method", ["3th_poly", "4th_poly", "spline"])
def test_bsln_model(input_data, method):
    lactate_array, intensity_array = input_data
    model = Bsln()
    predicted_lactate = model.fit(
        intensity_array, lactate_array, method=method
    ).predict(1.2)
    assert isinstance(predicted_lactate, float)
