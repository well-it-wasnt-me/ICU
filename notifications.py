"""
Notification helpers for detection events.

Provides asynchronous Telegram notifications when a known person is detected.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import requests

from logger_setup import logger


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
    ) -> None:
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.timeout = timeout
        self._session = requests.Session() if telegram_bot_token else None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

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
        self._executor.shutdown(wait=True)
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
