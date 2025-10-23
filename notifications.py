"""
Notification helpers for detection events and Telegram interactions.

Provides asynchronous Telegram notifications when a known person is detected,
and handles inbound Telegram commands for managing POI images.
"""

import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests
from requests.exceptions import RequestException

from image_utils import ImageUtils
from logger_setup import logger


@dataclass
class AddPoiSession:
    """Track the lifecycle of an interactive add_poi request."""

    stage: str
    original_name: Optional[str] = None
    slug: Optional[str] = None
    directory: Optional[Path] = None
    preexisting_directory: bool = False
    photo_count: int = 0


class NotificationManager:
    """
    Manage outbound notifications for detection events.

    Currently supports Telegram messages through the bot API. Messages are sent
    on a background thread so the recognition pipeline is not blocked.
    """

    def __init__(
        self,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        timeout: int = 10,
        max_workers: int = 2,
        train_dir: str = "poi",
        enable_command_handler: bool = True,
        command_poll_timeout: int = 20,
    ) -> None:
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self._chat_id_str = str(telegram_chat_id) if telegram_chat_id is not None else None
        self.timeout = timeout
        self.train_dir = Path(train_dir)
        self._session = requests.Session() if telegram_bot_token else None
        self._command_session = (
            requests.Session() if telegram_bot_token and enable_command_handler else None
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stop_event = threading.Event()
        self._command_thread: Optional[threading.Thread] = None
        self._command_poll_timeout = max(1, int(command_poll_timeout))
        self._last_update_id: Optional[int] = None
        self._sessions: Dict[str, AddPoiSession] = {}
        self._enable_command_handler = enable_command_handler
        if self.enabled and enable_command_handler:
            self._start_command_handler()

    @property
    def enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    # Telegram command handling -------------------------------------------------

    def _start_command_handler(self) -> None:
        if self._command_thread is not None or not self._command_session:
            return
        self._prime_update_offset()
        self._command_thread = threading.Thread(
            target=self._poll_command_updates,
            name="TelegramCommandPoller",
            daemon=True,
        )
        self._command_thread.start()

    def _prime_update_offset(self) -> None:
        assert self._command_session is not None
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
        params = {"timeout": 0}
        try:
            response = self._command_session.get(url, params=params, timeout=self.timeout)
            if response.status_code >= 300:
                logger.error(
                    "Telegram getUpdates prime failed (%s): %s",
                    response.status_code,
                    response.text,
                )
                return
            data = response.json()
        except (RequestException, ValueError) as exc:
            logger.error("Failed to prime Telegram update offset: %s", exc)
            return

        results = data.get("result") if isinstance(data, dict) else None
        if results:
            self._last_update_id = results[-1].get("update_id")

    def _poll_command_updates(self) -> None:
        assert self._command_session is not None
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
        while not self._stop_event.is_set():
            params = {"timeout": self._command_poll_timeout}
            if self._last_update_id is not None:
                params["offset"] = self._last_update_id + 1

            try:
                response = self._command_session.get(
                    url,
                    params=params,
                    timeout=self._command_poll_timeout + self.timeout,
                )
            except RequestException as exc:
                logger.error("Telegram getUpdates error: %s", exc)
                time.sleep(5)
                continue

            if response.status_code >= 300:
                logger.error(
                    "Telegram getUpdates failed (%s): %s",
                    response.status_code,
                    response.text,
                )
                time.sleep(5)
                continue

            try:
                payload = response.json()
            except ValueError as exc:
                logger.error("Invalid JSON in Telegram getUpdates response: %s", exc)
                time.sleep(2)
                continue

            if not payload.get("ok", False):
                logger.error("Telegram getUpdates returned error payload: %s", payload)
                time.sleep(5)
                continue

            for update in payload.get("result", []):
                self._last_update_id = update.get("update_id", self._last_update_id)
                try:
                    self._handle_update(update)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error("Error while handling Telegram update: %s", exc)

    def _handle_update(self, update: dict) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return

        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            return
        chat_id_str = str(chat_id)

        if self._chat_id_str and chat_id_str != self._chat_id_str:
            logger.debug("Ignoring telegram message from unauthorized chat %s.", chat_id)
            return

        session = self._sessions.get(chat_id_str)
        text = (message.get("text") or "").strip()

        if text:
            lower_text = text.lower()
            if lower_text == "add_poi":
                if session:
                    self._send_text_message(
                        "An add_poi session is already in progress. Send photos or 'done', or type 'cancel' to abort."
                    )
                else:
                    self._start_add_poi_session(chat_id_str)
                return
            if lower_text == "cancel":
                if session:
                    self._cancel_session(chat_id_str, aborted_by_user=True)
                else:
                    self._send_text_message("No add_poi session is currently active.")
                return
            if lower_text == "done":
                if session:
                    self._finalize_session(chat_id_str)
                else:
                    self._send_text_message("No add_poi session is currently active.")
                return

            if session and session.stage == "awaiting_name":
                self._set_session_name(chat_id_str, session, text)
                return
            # Ignore unrelated text when no session is active.
            return

        if "photo" in message and session and session.stage == "collecting_photos":
            self._handle_photo_message(chat_id_str, session, message["photo"])
        elif "photo" in message:
            self._send_text_message("Send 'add_poi' first so I know where to store these photos.")

    def _start_add_poi_session(self, chat_id_str: str) -> None:
        self._sessions[chat_id_str] = AddPoiSession(stage="awaiting_name")
        self._send_text_message("name")

    def _cancel_session(self, chat_id_str: str, aborted_by_user: bool = False) -> None:
        session = self._sessions.pop(chat_id_str, None)
        if not session:
            return
        if session.directory and not session.preexisting_directory and session.directory.exists():
            try:
                if not any(session.directory.iterdir()):
                    session.directory.rmdir()
            except OSError as exc:
                logger.warning("Failed to clean up temporary POI directory %s: %s", session.directory, exc)
        if aborted_by_user:
            self._send_text_message("add_poi cancelled.")

    def _set_session_name(self, chat_id_str: str, session: AddPoiSession, raw_name: str) -> None:
        sanitized = self._sanitize_name(raw_name)
        if not sanitized:
            self._send_text_message("Invalid name. Use letters, numbers, spaces, '-' or '_'. Try again.")
            return

        person_dir = self.train_dir / sanitized
        preexisting = person_dir.exists()
        if preexisting and not person_dir.is_dir():
            logger.error("Path %s exists and is not a directory. Cannot store POI.", person_dir)
            self._send_text_message("Cannot store POI because the destination path is not a folder.")
            self._cancel_session(chat_id_str)
            return

        try:
            person_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to prepare POI directory %s: %s", person_dir, exc)
            self._send_text_message("Failed to prepare storage directory. Please try another name or 'cancel'.")
            return

        session.stage = "collecting_photos"
        session.original_name = raw_name.strip()
        session.slug = sanitized
        session.directory = person_dir
        session.preexisting_directory = preexisting
        session.photo_count = 0

        self._send_text_message("picture(s)")
        if preexisting:
            self._send_text_message("Existing images found; new uploads will be added to this person.")

    def _handle_photo_message(self, chat_id_str: str, session: AddPoiSession, photo_sizes) -> None:
        if not isinstance(photo_sizes, list):
            logger.warning("Unexpected photo payload: %s", photo_sizes)
            self._send_text_message("Couldn't read the photo payload. Please try again.")
            return

        if not session.directory:
            self._send_text_message("No destination directory is set yet. Send the name first.")
            return

        best_photo = max(photo_sizes, key=lambda item: item.get("file_size", 0) or 0)
        file_id = best_photo.get("file_id")
        if not file_id:
            self._send_text_message("Photo missing file identifier. Try sending it again.")
            return

        saved_path = self._download_photo(session, file_id)
        if not saved_path:
            self._send_text_message("Failed to download the photo. Please try again.")
            return

        session.photo_count += 1
        self._send_text_message(
            f"Stored photo {session.photo_count}. Send more or reply with 'done' when finished."
        )

    def _download_photo(self, session: AddPoiSession, file_id: str) -> Optional[Path]:
        assert self._command_session is not None
        if not session.directory:
            return None

        get_file_url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getFile"
        try:
            meta_response = self._command_session.get(
                get_file_url,
                params={"file_id": file_id},
                timeout=self.timeout,
            )
        except RequestException as exc:
            logger.error("Failed to request Telegram file metadata: %s", exc)
            return None

        if meta_response.status_code >= 300:
            logger.error(
                "Telegram getFile failed (%s): %s",
                meta_response.status_code,
                meta_response.text,
            )
            return None

        try:
            meta_payload = meta_response.json()
        except ValueError as exc:
            logger.error("Invalid getFile JSON response: %s", exc)
            return None

        if not meta_payload.get("ok"):
            logger.error("Telegram getFile returned error payload: %s", meta_payload)
            return None

        file_path = meta_payload.get("result", {}).get("file_path")
        if not file_path:
            logger.error("Telegram getFile missing file_path: %s", meta_payload)
            return None

        suffix = Path(file_path).suffix or ".jpg"
        filename = f"{int(time.time())}_{session.photo_count + 1:03d}{suffix}"
        destination = session.directory / filename

        download_url = f"https://api.telegram.org/file/bot{self.telegram_bot_token}/{file_path}"
        try:
            with self._command_session.get(download_url, stream=True, timeout=self.timeout) as file_response:
                if file_response.status_code >= 300:
                    logger.error(
                        "Telegram file download failed (%s): %s",
                        file_response.status_code,
                        file_response.text,
                    )
                    return None
                with open(destination, "wb") as handle:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)
        except RequestException as exc:
            logger.error("Error downloading Telegram photo: %s", exc)
            return None
        except OSError as exc:
            logger.error("Failed to save Telegram photo to %s: %s", destination, exc)
            return None

        return destination

    def _finalize_session(self, chat_id_str: str) -> None:
        session = self._sessions.pop(chat_id_str, None)
        if not session:
            self._send_text_message("No add_poi session is currently active.")
            return

        if session.stage == "awaiting_name":
            self._send_text_message("You need to provide a name before finishing. Reply with the person's name.")
            self._sessions[chat_id_str] = session
            return

        saved = session.photo_count
        directory = session.directory
        if directory and directory.exists() and saved > 0:
            try:
                ImageUtils.convert_images_to_rgb(str(directory))
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Failed to convert images to RGB for %s: %s", directory, exc)

            person_label = session.original_name or session.slug or "POI"
            plural = "photo" if saved == 1 else "photos"
            self._send_text_message(
                f"Added {saved} {plural} for '{person_label}'. Remember to retrain the model to include this POI."
            )
        else:
            if directory and not session.preexisting_directory:
                try:
                    if directory.exists():
                        shutil.rmtree(directory)
                except OSError as exc:
                    logger.warning("Failed to remove empty POI directory %s: %s", directory, exc)
            self._send_text_message("No photos were received. The add_poi session has been cancelled.")

    @staticmethod
    def _sanitize_name(raw_name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9 _-]+", "_", raw_name.strip())
        sanitized = re.sub(r"[_\s]+", "_", sanitized)
        sanitized = sanitized.strip("_-")
        if sanitized in {"", ".", ".."}:
            return ""
        return sanitized

    def notify_detection(
        self,
        camera_name: str,
        person_name: str,
        confidence: float,
        capture_path: Optional[str] = None,
        side_by_side_path: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Queue a notification about a successful recognition.
        """
        if not self.enabled:
            return

        payload = {
            "camera": camera_name,
            "person": person_name,
            "confidence": confidence,
            "capture": capture_path,
            "side_by_side": side_by_side_path,
            "timestamp": timestamp or datetime.utcnow(),
        }
        self._executor.submit(self._send_telegram_message, payload)

    def notify_plate_detection(
        self,
        camera_name: str,
        plate_number: str,
        confidence: float,
        occurrences: int,
        capture_path: Optional[str] = None,
        crop_path: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        watchlist_hit: bool = False,
    ) -> None:
        """
        Queue a notification about a license plate detection.
        """
        if not self.enabled:
            return

        payload = {
            "camera": camera_name,
            "plate": plate_number,
            "confidence": confidence,
            "occurrences": occurrences,
            "capture": capture_path,
            "crop": crop_path,
            "timestamp": timestamp or datetime.utcnow(),
            "watchlist_hit": watchlist_hit,
        }
        self._executor.submit(self._send_plate_message, payload)

    def shutdown(self) -> None:
        """
        Flush outstanding notifications and release resources.
        """
        self._stop_event.set()
        if self._command_thread and self._command_thread.is_alive():
            self._command_thread.join(timeout=5)
        self._executor.shutdown(wait=True)
        if self._command_session:
            self._command_session.close()
        if self._session:
            self._session.close()

    # Internal helpers -------------------------------------------------

    def _send_telegram_message(self, payload: dict) -> None:
        assert self._session is not None
        text = self._format_message(payload)
        capture_path = payload.get("capture")
        side_path = payload.get("side_by_side")

        sent_any = False
        if capture_path and os.path.exists(capture_path):
            sent_any = self._send_photo(
                image_path=capture_path,
                caption=text,
            )
        elif capture_path:
            logger.warning("Capture path %s not found, skipping photo.", capture_path)

        if side_path and os.path.exists(side_path):
            side_caption = self._format_side_caption(payload)
            self._send_photo(image_path=side_path, caption=side_caption)
            sent_any = True
        elif side_path:
            logger.warning("Side-by-side path %s not found, skipping photo.", side_path)

        if not sent_any:
            self._send_text_message(text)

    def _send_plate_message(self, payload: dict) -> None:
        assert self._session is not None
        text = self._format_plate_message(payload)
        capture_path = payload.get("capture")
        crop_path = payload.get("crop")

        sent_any = False
        if capture_path and os.path.exists(capture_path):
            sent_any = self._send_photo(
                image_path=capture_path,
                caption=text,
            )
        elif capture_path:
            logger.warning("Plate capture path %s not found, skipping photo.", capture_path)

        if crop_path and os.path.exists(crop_path):
            crop_caption = self._format_plate_crop_caption(payload)
            self._send_photo(image_path=crop_path, caption=crop_caption)
            sent_any = True
        elif crop_path:
            logger.warning("Plate crop path %s not found, skipping photo.", crop_path)

        if not sent_any:
            self._send_text_message(text)

    @staticmethod
    def _format_message(payload: dict) -> str:
        ts = payload["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        confidence = round(payload["confidence"], 2)
        return (
            f"ICU Alert\n"
            f"Camera: {payload['camera']}\n"
            f"Person: {payload['person']}\n"
            f"Confidence: {confidence}%\n"
            f"Detected at: {ts}"
        )

    @staticmethod
    def _format_side_caption(payload: dict) -> str:
        confidence = round(payload["confidence"], 2)
        return (
            f"Side-by-side match\n"
            f"Camera: {payload['camera']}\n"
            f"Person: {payload['person']}\n"
            f"Confidence: {confidence}%"
        )

    @staticmethod
    def _format_plate_message(payload: dict) -> str:
        ts = payload["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        confidence = round(payload["confidence"], 2)
        occurrences = payload.get("occurrences", 1)
        watchlist = payload.get("watchlist_hit", False)
        watchlist_line = "Watchlist: YES" if watchlist else "Watchlist: no"
        return (
            f"ICU Plate Alert\n"
            f"Camera: {payload['camera']}\n"
            f"Plate: {payload['plate']}\n"
            f"Confidence: {confidence}%\n"
            f"Occurrences: {occurrences}\n"
            f"{watchlist_line}\n"
            f"Detected at: {ts}"
        )

    @staticmethod
    def _format_plate_crop_caption(payload: dict) -> str:
        confidence = round(payload["confidence"], 2)
        occurrences = payload.get("occurrences", 1)
        return (
            f"Plate crop\n"
            f"Camera: {payload['camera']}\n"
            f"Plate: {payload['plate']}\n"
            f"Confidence: {confidence}%\n"
            f"Occurrences: {occurrences}"
        )

    def _send_text_message(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        data = {"chat_id": self.telegram_chat_id, "text": text}
        try:
            response = self._session.post(url, data=data, timeout=self.timeout)
            if response.status_code >= 300:
                logger.error(
                    "Telegram message failed (%s): %s",
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.error("Telegram message error: %s", exc)

    def _send_photo(self, image_path: str, caption: str) -> bool:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"
        data = {"chat_id": self.telegram_chat_id, "caption": caption}
        try:
            with open(image_path, "rb") as photo:
                response = self._session.post(
                    url,
                    data=data,
                    files={"photo": photo},
                    timeout=self.timeout,
                )
            if response.status_code >= 300:
                logger.error(
                    "Telegram photo failed (%s): %s",
                    response.status_code,
                    response.text,
                )
                return False
        except Exception as exc:
            logger.error("Telegram photo error (%s): %s", image_path, exc)
            return False
        return True
