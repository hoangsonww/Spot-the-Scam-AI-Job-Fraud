from pathlib import Path

from spot_scam.config.loader import _expand_dot_keys, load_config


def test_expand_dot_keys():
    overrides = {"data.raw_dir": "./custom"}
    expanded = _expand_dot_keys(overrides)
    assert expanded["data"]["raw_dir"] == "./custom"


def test_load_config_overrides(tmp_path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text("data:\n  raw_filename: sample.csv\n")
    config = load_config(config_path=config_path)
    assert config["data"]["raw_filename"] == "sample.csv"
