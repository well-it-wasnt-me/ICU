import yaml

def test_config_parsing(tmp_path):
    """
    Write a sample cameras.yaml file and verify that it is parsed correctly.
    """
    sample_config = """
    cameras:
      - name: TestCam
        url: http://example.com/stream
        fps: 30
    """
    config_file = tmp_path / "cameras.yaml"
    config_file.write_text(sample_config)

    # Parse the YAML file
    config = yaml.safe_load(config_file.read_text())

    # Assert that the config has the expected structure
    assert "cameras" in config, "Key 'cameras' missing in config."
    assert isinstance(config["cameras"], list), "'cameras' should be a list."
    assert config["cameras"][0]["name"] == "TestCam", "Camera name does not match."
