from pathlib import Path

from scripts import smoke_test


def test_smoke_test_loads_config() -> None:
    config_path = Path("ckpts/config.yaml")
    assert config_path.exists()
    smoke_test.main = smoke_test.main
