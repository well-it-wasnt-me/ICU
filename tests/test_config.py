import yaml

def test_config_parsing(tmp_path):
    """
    Write a sample cameras.yaml file and verify that it is parsed correctly.
    """
    sample_config = """
    cameras:
      - name: TestCam
        stream_url: http://example.com/stream
        process_frame_interval: 10
        capture_cooldown: 30

    settings:
      enable_tui: true
      show_preview: false
      preview_scale: 0.6
      target_processing_fps: 3.0

    notifications:
      telegram:
        bot_token: "abc:123"
        chat_id: "456"
        timeout: 5
        max_workers: 1
    """
    config_file = tmp_path / "cameras.yaml"
    config_file.write_text(sample_config)

    # Parse the YAML file
    config = yaml.safe_load(config_file.read_text())

    # Assert that the config has the expected structure
    assert "cameras" in config, "Key 'cameras' missing in config."
    assert isinstance(config["cameras"], list), "'cameras' should be a list."
    camera = config["cameras"][0]
    assert camera["name"] == "TestCam", "Camera name does not match."
    assert camera["stream_url"] == "http://example.com/stream", "Stream URL not parsed correctly."

    settings = config.get("settings")
    assert settings, "'settings' section missing."
    assert settings["enable_tui"] is True, "enable_tui flag not parsed correctly."
    assert settings["preview_scale"] == 0.6, "preview_scale not parsed correctly."

    telegram = config.get("notifications", {}).get("telegram")
    assert telegram, "'notifications.telegram' section missing."
    assert telegram["bot_token"] == "abc:123", "Telegram bot token mismatch."
    assert telegram["timeout"] == 5, "Telegram timeout mismatch."
