import pytest
from src.config import load_config
from src.data import load_dataset


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def dataset(config):
    # Actual version-checked UCI data. Missing data is a failure, not a skip.
    return load_dataset(config)


@pytest.fixture
def sample(dataset):
    return dataset.sample(300, random_state=7).copy(deep=True)
