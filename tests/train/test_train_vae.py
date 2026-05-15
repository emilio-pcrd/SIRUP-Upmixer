import pytest


@pytest.mark.skip(reason="Training requires dataset and GPU")
def test_train_vae_import() -> None:
    import train.train_vae  # noqa: F401
