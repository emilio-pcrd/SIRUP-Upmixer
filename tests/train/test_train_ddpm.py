import pytest


@pytest.mark.skip(reason="Training requires dataset and GPU")
def test_train_ddpm_import() -> None:
    import train.train_ddpm  # noqa: F401
