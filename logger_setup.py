"""Central logging configuration for the ICU project."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

import yaml


_DEFAULT_LOG_FILE = "face_recognition.log"
_DEFAULT_APP_CONFIG = os.environ.get("ICU_APP_CONFIG", "configs/app.yaml")


def _level_from_value(value: Any, fallback: int = logging.DEBUG) -> int:
    if value is None:
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        level = logging.getLevelName(value.upper())
        if isinstance(level, int):
            return level
    return fallback


def _load_logging_section(config_path: str) -> tuple[int, Optional[str]]:
    if not config_path or not os.path.exists(config_path):
        return logging.DEBUG, None
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return logging.DEBUG, None

    if not isinstance(config, Mapping):
        return logging.DEBUG, None

    logging_cfg = config.get("logging", {}) if isinstance(config, Mapping) else {}
    level = _level_from_value(logging_cfg.get("level"))
    log_file = logging_cfg.get("file") if isinstance(logging_cfg, Mapping) else None
    if log_file:
        log_file = os.fspath(log_file)
    return level, log_file


def _set_logger_level(target_logger: logging.Logger, level: int) -> None:
    target_logger.setLevel(level)
    for handler in target_logger.handlers:
        handler.setLevel(level)


def setup_logging(log_file: str = _DEFAULT_LOG_FILE, app_config_path: Optional[str] = None) -> logging.Logger:
    app_config_path = app_config_path or _DEFAULT_APP_CONFIG
    derived_level, configured_file = _load_logging_section(app_config_path)
    if configured_file:
        log_file = configured_file

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        try:
            fh = logging.FileHandler(log_file)
        except OSError:
            fh = logging.NullHandler()
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

    _set_logger_level(root_logger, derived_level)
    return root_logger


def configure_logging(config: Mapping[str, Any]) -> None:
    logging_cfg = config.get("logging", {}) if isinstance(config, Mapping) else {}
    level = _level_from_value(logging_cfg.get("level"))
    _set_logger_level(logging.getLogger(), level)


logger = setup_logging()
