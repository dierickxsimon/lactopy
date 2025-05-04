import os
import pytest

import pandas as pd
import numpy as np

TEST_DATA_DIR = "tests/test_data/"
FILES = [
    os.path.join(TEST_DATA_DIR, file)
    for file in os.listdir(TEST_DATA_DIR)
    if file.endswith(".csv")
]


@pytest.fixture(params=FILES)
def input_data(request):
    file = request.param
    print(f"Processing file: {file}")
    if not file.endswith(".csv"):
        pytest.skip(f"Skipping non-CSV file: {file}")

    df = pd.read_csv(file)
    return df["lactate"].to_numpy(), df["watt"].to_numpy()


def test_if_input_is_numpy_array(input_data):
    lactate, intensity = input_data
    assert isinstance(lactate, np.ndarray)
    assert isinstance(intensity, np.ndarray)
