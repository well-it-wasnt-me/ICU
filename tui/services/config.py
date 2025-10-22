"""
Helpers for loading ICU configuration files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_camera_config(path: os.PathLike[str] | str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load the camera configuration file and return the list of camera entries.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    cameras, raw_config:
        A tuple containing the list of camera dictionaries and the raw configuration
        mapping loaded from YAML.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Camera configuration file '{resolved}' does not exist.")

    with resolved.open("r") as handle:
        data = yaml.safe_load(handle) or {}

    if isinstance(data, list):
        cameras = data
        raw_config = {"cameras": cameras}
    elif isinstance(data, dict):
        cameras = data.get("cameras", [])
        raw_config = data
    else:
        cameras = []
        raw_config = {}

    return cameras, raw_config


def load_app_config(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """
    Load the main application configuration (app.yaml).
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Application configuration file '{resolved}' does not exist.")

    with resolved.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    return data
