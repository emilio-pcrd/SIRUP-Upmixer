import pytest


@pytest.mark.skip(reason="Requires audio assets and pyroomacoustics runtime setup")
def test_create_svects_import() -> None:
    import preprocessing.create_svects  # noqa: F401
