import yaml


def test_camera_and_app_config_parsing(tmp_path):
    """
    Verify that camera and application configuration YAML files load independently.
    """
    camera_yaml = """
    cameras:
      - name: TestCam
        stream_url: http://example.com/stream
        process_frame_interval: 10
        capture_cooldown: 30
    """
    app_yaml = """
    settings:
      target_processing_fps: 3.0
      cpu_pressure_threshold: 80.0

    notifications:
      telegram:
        bot_token: "abc:123"
        chat_id: "456"
        timeout: 5
        max_workers: 1

    logging:
      level: WARNING
      file: custom.log
    """

    camera_path = tmp_path / "cameras.yaml"
    app_path = tmp_path / "app.yaml"
    camera_path.write_text(camera_yaml)
    app_path.write_text(app_yaml)

    camera_cfg = yaml.safe_load(camera_path.read_text())
    app_cfg = yaml.safe_load(app_path.read_text())

    assert "cameras" in camera_cfg, "Camera config missing 'cameras' key."
    assert len(camera_cfg["cameras"]) == 1, "Unexpected number of cameras."
    camera = camera_cfg["cameras"][0]
    assert camera["name"] == "TestCam"
    assert camera["stream_url"] == "http://example.com/stream"

    settings = app_cfg.get("settings")
    assert settings, "'settings' section missing in app config."
    assert settings["target_processing_fps"] == 3.0
    assert settings["cpu_pressure_threshold"] == 80.0

    telegram = app_cfg.get("notifications", {}).get("telegram")
    assert telegram, "'notifications.telegram' section missing in app config."
    assert telegram["bot_token"] == "abc:123"
    assert telegram["timeout"] == 5

    logging_cfg = app_cfg.get("logging")
    assert logging_cfg, "'logging' section missing in app config."
    assert logging_cfg["level"] == "WARNING"
    assert logging_cfg["file"] == "custom.log"
