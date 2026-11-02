
import pandas as pd
import pytest

from src.data.preprocessing import prepare_dataset


@pytest.fixture
def prepared_data():
    frame = pd.DataFrame({
        "temperature": range(18, 30),
        "humidity": [40, 42, 43, 45, 48, 50, 51, 53, 55, 57, 58, 60],
        "material": [
            "steel", "wood", "steel", "plastic", "wood", "steel",
            "plastic", "wood", "steel", "plastic", "wood", "steel",
        ],
        "measurement": [
            10.0, 8.0, 10.5, 9.0, 8.5, 11.0,
            9.5, 8.2, 10.8, 9.7, 8.8, 11.4,
        ],
    })

    return prepare_dataset(
        frame,
        target_column="measurement",
        task="regression",
        test_size=0.25,
        random_seed=42,
    )
