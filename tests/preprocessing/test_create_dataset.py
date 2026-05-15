import pytest


@pytest.mark.skip(reason="Requires audio assets and pyroomacoustics runtime setup")
def test_create_dataset_import() -> None:
    import preprocessing.create_dataset  # noqa: F401
