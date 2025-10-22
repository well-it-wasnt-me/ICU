import logging
import importlib
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_logger_level_from_app_config(tmp_path, monkeypatch):
    app_cfg = tmp_path / "app.yaml"
    app_cfg.write_text("logging:\n  level: ERROR\n")

    monkeypatch.setenv("ICU_APP_CONFIG", str(app_cfg))

    if "logger_setup" in list(sys.modules):
        del sys.modules["logger_setup"]

    logger_setup = importlib.import_module("logger_setup")

    assert logger_setup.logger.level == logging.ERROR


def test_configure_logging_updates_existing_logger(monkeypatch):
    monkeypatch.delenv("ICU_APP_CONFIG", raising=False)

    import logger_setup

    logger_setup.configure_logging({"logging": {"level": "INFO"}})
    assert logger_setup.logger.level == logging.INFO
