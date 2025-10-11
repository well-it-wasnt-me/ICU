"""
Utilities to discover public camera streams by city.

Currently supports:
* insecam.org (RTSP feeds)
* earthcam.com (HLS feeds)
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from logger_setup import logger


@dataclass
class DiscoveredStream:
    url: str
    protocol: str
    headers: Optional[Dict[str, str]] = None


class CameraStreamFinder:
    """
    Aggregate multiple scrapers to locate public camera streams.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        timeout: int = 10,
        verify_ssl: bool = False,
        scrapers: Optional[Sequence["BaseCameraScraper"]] = None,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        if not self.verify_ssl:
            try:
                requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
            except Exception:
                pass
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
        self.session.headers["Accept"] = (
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        )
        self.scrapers: Sequence[BaseCameraScraper] = scrapers or (
            InsecamScraper(),
            EarthCamScraper(),
        )

    def find_streams(
        self,
        city: str,
        max_results: int = 10,
        max_pages: int = 3,
    ) -> List[DiscoveredStream]:
        results: List[DiscoveredStream] = []
        seen: set[str] = set()
        for scraper in self.scrapers:
            needed = max_results - len(results)
            if needed <= 0:
                break
            for stream in scraper.find(self, city, needed, max_pages):
                if stream.url in seen:
                    continue
                results.append(stream)
                seen.add(stream.url)
                if len(results) >= max_results:
                    break
        return results

    def fetch(self, url: str, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                verify=self.verify_ssl,
                headers=headers,
            )
            if response.status_code >= 400:
                logger.warning("Failed to fetch %s (status %s)", url, response.status_code)
                return None
            return response.text
        except requests.RequestException as exc:
            logger.error("Error fetching %s: %s", url, exc)
            return None


class BaseCameraScraper:
    def find(
        self,
        finder: CameraStreamFinder,
        city: str,
        max_results: int,
        max_pages: int,
    ) -> Iterable[DiscoveredStream]:
        raise NotImplementedError


class InsecamScraper(BaseCameraScraper):
    LISTING_URL_TEMPLATE = "https://www.insecam.org/en/bycity/{city}/"
    DETAIL_URL_TEMPLATE = "https://www.insecam.org{path}"
    RTSP_REGEX = re.compile(r"(rtsp://[^\s\"']+)")
    BASE_HEADERS = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.insecam.org/",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    BOOTSTRAP_URL = "https://www.insecam.org/en/"

    def find(
        self,
        finder: CameraStreamFinder,
        city: str,
        max_results: int,
        max_pages: int,
    ) -> Iterable[DiscoveredStream]:
        self._bootstrap_session(finder)
        city_slug = city.strip().replace(" ", "_")
        found = 0
        for page in range(1, max_pages + 1):
            page_url = self._build_listing_url(city_slug, page)
            logger.info("Fetching insecam listing: %s", page_url)
            listing_html = finder.fetch(page_url, headers=self.BASE_HEADERS)
            if not listing_html:
                logger.debug("Listing fetch returned no content, retrying with hardened headers.")
                listing_html = self._direct_fetch(finder, page_url, referer=self.BASE_HEADERS["Referer"])
            if not listing_html:
                continue
            for detail_path in self._extract_camera_links(listing_html):
                detail_url = self.DETAIL_URL_TEMPLATE.format(path=detail_path)
                detail_headers = {
                    **self.BASE_HEADERS,
                    "Referer": page_url,
                }
                detail_html = finder.fetch(detail_url, headers=detail_headers)
                if not detail_html:
                    detail_html = self._direct_fetch(finder, detail_url, referer=page_url)
                if not detail_html:
                    continue
                for rtsp_url in self._extract_rtsp(detail_html):
                    yield DiscoveredStream(url=rtsp_url, protocol="rtsp")
                    found += 1
                    if found >= max_results:
                        return

    def _build_listing_url(self, city_slug: str, page: int) -> str:
        if page <= 1:
            return self.LISTING_URL_TEMPLATE.format(city=city_slug)
        return f"{self.LISTING_URL_TEMPLATE.format(city=city_slug)}?page={page}"

    @staticmethod
    def _extract_camera_links(html: str) -> Iterable[str]:
        soup = BeautifulSoup(html, "html.parser")
        for a_tag in soup.select("a[href^='/en/view/']"):
            href = a_tag.get("href")
            if href:
                yield href

    def _extract_rtsp(self, html: str) -> Iterable[str]:
        return self.RTSP_REGEX.findall(html)

    def _bootstrap_session(self, finder: CameraStreamFinder) -> None:
        if getattr(self, "_bootstrapped", False):
            return
        finder.fetch(self.BOOTSTRAP_URL, headers=self.BASE_HEADERS)
        self._bootstrapped = True

    def _direct_fetch(self, finder: CameraStreamFinder, url: str, referer: Optional[str] = None) -> Optional[str]:
        headers = {**self.BASE_HEADERS}
        if referer:
            headers["Referer"] = referer
        try:
            response = finder.session.get(
                url,
                timeout=finder.timeout,
                verify=finder.verify_ssl,
                headers=headers,
            )
            if response.status_code >= 400:
                logger.warning("Fallback fetch failed for %s (status %s)", url, response.status_code)
                return None
            return response.text
        except requests.RequestException as exc:
            logger.error("Error during fallback fetch %s: %s", url, exc)
            return None


class EarthCamScraper(BaseCameraScraper):
    SEARCH_URL_TEMPLATE = "https://www.earthcam.com/search/ft-search.php?term={query}"
    STREAM_REGEX = re.compile(
        r"(?P<url>(?:https?|rtsp):\/\/[^\s\"']+?\.(?:m3u8?|mpd|mp4|mpg|mjpeg|mjpg|flv|webm|ts)(?:\?[^\s\"']*)?|\/\/[^\s\"']+?\.(?:m3u8?|mpd|mp4|mpg|mjpeg|mjpg|flv|webm|ts)(?:\?[^\s\"']*)?)",
        re.IGNORECASE,
    )
    CONFIG_URL_REGEX = re.compile(
        r"""["']?(?:configUrl|config|config_path|playerConfigUrl)["']?\s*[:=]\s*["'](?P<url>[^"']+?\.json[^"']*)["']""",
        re.IGNORECASE,
    )
    EARTHCAM_HEADERS = {
        "Referer": "https://www.earthcam.com/",
        "Origin": "https://www.earthcam.com",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    BASE_KEYS = {"base", "baseurl", "securebase", "basehttps", "urlroot", "root", "httpbase", "httpsbase"}
    FILE_KEYS = {"file", "playlist", "path", "manifest", "stream", "uri", "hls", "dash"}
    URL_KEYS = {
        "url",
        "src",
        "href",
        "streamurl",
        "hlsurl",
        "dashurl",
        "mjpegurl",
        "httpsurl",
        "httpurl",
        "contenturl",
        "video",
        "video_src",
    }

    def find(
        self,
        finder: CameraStreamFinder,
        city: str,
        max_results: int,
        max_pages: int,
    ) -> Iterable[DiscoveredStream]:
        query = city.strip().replace(" ", "%20")
        search_url = self.SEARCH_URL_TEMPLATE.format(query=query)
        logger.info("Fetching EarthCam listing: %s", search_url)
        listing_html = finder.fetch(search_url, headers=self.EARTHCAM_HEADERS)
        if not listing_html:
            return

        detail_paths = list(self._extract_camera_links(listing_html))
        if not detail_paths:
            logger.info("EarthCam search returned no camera cards for '%s'.", city)
        found = 0
        for path in detail_paths[: max_pages * 5]:
            detail_url = self._build_detail_url(path)
            detail_html = finder.fetch(detail_url, headers=self.EARTHCAM_HEADERS)
            if not detail_html:
                continue
            for stream_url in self._extract_streams(finder, detail_url, detail_html):
                protocol = self._infer_protocol(stream_url)
                yield DiscoveredStream(
                    url=stream_url,
                    protocol=protocol,
                    headers=self.EARTHCAM_HEADERS.copy(),
                )
                found += 1
                if found >= max_results:
                    return

    def _build_detail_url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        return f"https://www.earthcam.com{path}"

    def _extract_camera_links(self, html: str) -> Iterable[str]:
        soup = BeautifulSoup(html, "html.parser")
        result_banner = soup.select_one("#pagination_top")
        if result_banner:
            text = result_banner.get_text(strip=True)
            logger.info("EarthCam summary: %s", text)

        seen: set[str] = set()
        card_selectors = ("div.tiny-single-row", "div.searchItem")
        for selector in card_selectors:
            for card in soup.select(f"{selector} a[href]"):
                href = card.get("href")
                if not href or href.startswith("javascript:"):
                    continue
                if href in seen:
                    continue
                seen.add(href)
                yield href

    def _extract_from_json(self, script_text: str) -> Iterable[str]:
        matches = re.findall(r"({\"[^\}]+})", script_text)
        for match in matches:
            try:
                data = json.loads(match)
                url = data.get("url") or data.get("href")
                if isinstance(url, str):
                    yield url
            except json.JSONDecodeError:
                continue

    def _extract_streams(self, finder: CameraStreamFinder, page_url: str, html: str) -> Iterable[str]:
        discovered: List[str] = []
        seen: set[str] = set()
        pending_config_urls: set[str] = set()

        def record(url: str) -> None:
            normalized = self._normalize_url(page_url, url)
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            discovered.append(normalized)

        def enqueue_configs(text: str, base_url: str) -> None:
            for config_url in self._extract_config_urls_from_text(text, base_url):
                if config_url not in pending_config_urls:
                    pending_config_urls.add(config_url)

        self._record_from_text(html, record)
        enqueue_configs(html, page_url)
        soup = BeautifulSoup(html, "html.parser")

        for script_tag in soup.find_all("script"):
            script_text = script_tag.string or script_tag.get_text()
            if script_text:
                self._record_from_text(script_text, record)
                enqueue_configs(script_text, page_url)

        for attr_name in ("src", "data-src", "data-hls", "data-url", "data-stream", "data-file"):
            for element in soup.find_all(attrs={attr_name: True}):
                value = element.get(attr_name)
                if value:
                    self._record_from_text(value, record)
                    enqueue_configs(value, page_url)

        iframe_sources = {
            urljoin(page_url, iframe.get("src"))
            for iframe in soup.find_all("iframe", src=True)
        }
        for iframe_url in iframe_sources:
            iframe_headers = {
                **self.EARTHCAM_HEADERS,
                "Referer": page_url,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            iframe_html = finder.fetch(iframe_url, headers=iframe_headers)
            if not iframe_html:
                continue
            self._record_from_text(iframe_html, record)
            enqueue_configs(iframe_html, iframe_url)
            iframe_soup = BeautifulSoup(iframe_html, "html.parser")
            for script_tag in iframe_soup.find_all("script"):
                script_text = script_tag.string or script_tag.get_text()
                if script_text:
                    self._record_from_text(script_text, record)
                    enqueue_configs(script_text, iframe_url)

        config_urls = self._extract_config_urls_from_dom(soup, page_url)
        pending_config_urls.update(config_urls)
        for config_url in config_urls:
            config_headers = {
                **self.EARTHCAM_HEADERS,
                "Referer": page_url,
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            }
            config_payload = finder.fetch(config_url, headers=config_headers)
            if not config_payload:
                continue
            self._record_from_config(config_payload, config_url, record, enqueue_configs)

        extra_configs = pending_config_urls - config_urls
        for config_url in extra_configs:
            config_headers = {
                **self.EARTHCAM_HEADERS,
                "Referer": page_url,
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            }
            config_payload = finder.fetch(config_url, headers=config_headers)
            if not config_payload:
                continue
            self._record_from_config(config_payload, config_url, record, enqueue_configs)

        return discovered

    def _record_from_text(self, text: str, recorder) -> None:
        if not text:
            return
        cleaned = html.unescape(text).replace("\\/", "/")
        for match in self.STREAM_REGEX.finditer(cleaned):
            candidate = match.group("url")
            recorder(candidate)

    def _record_from_config(
        self,
        payload: str,
        base_url: str,
        recorder,
        enqueue_configs,
    ) -> None:
        if not payload:
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            self._record_from_text(payload, recorder)
            enqueue_configs(payload, base_url)
            return

        for candidate in self._collect_urls_from_data(data, base_url):
            recorder(candidate)

    def _extract_config_urls_from_text(self, text: str, base_url: str) -> set[str]:
        urls: set[str] = set()
        if not text:
            return urls
        cleaned = html.unescape(text)
        for match in self.CONFIG_URL_REGEX.finditer(cleaned):
            candidate = match.group("url")
            normalized = urljoin(base_url, candidate)
            urls.add(normalized)
        return urls

    def _normalize_url(self, page_url: str, candidate: str) -> Optional[str]:
        if not candidate:
            return None
        candidate = candidate.strip()
        if candidate.startswith("//"):
            return f"https:{candidate}"
        if candidate.startswith(("http://", "https://", "rtsp://")):
            return candidate
        if candidate.startswith("/"):
            return urljoin(page_url, candidate)
        return None

    def _collect_urls_from_data(self, data, base_hint: Optional[str]) -> Iterable[str]:
        collected: List[str] = []

        def recurse(obj, current_base: Optional[str]) -> None:
            if isinstance(obj, dict):
                bases: List[str] = []
                files: List[str] = []
                for key, value in obj.items():
                    key_lower = key.lower()
                    if isinstance(value, str):
                        if key_lower in self.URL_KEYS:
                            collected.append(value)
                        if key_lower in self.BASE_KEYS:
                            bases.append(value)
                        if key_lower in self.FILE_KEYS:
                            files.append(value)
                    elif isinstance(value, (dict, list)):
                        recurse(value, current_base)

                effective_bases = bases or ([current_base] if current_base else [])
                for base in effective_bases:
                    for file_value in files:
                        combined = self._combine_base_and_file(base, file_value)
                        if combined:
                            collected.append(combined)

                next_base = bases[0] if bases else current_base
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        recurse(value, next_base)

            elif isinstance(obj, list):
                for item in obj:
                    recurse(item, current_base)

        recurse(data, base_hint)
        return collected

    def _extract_config_urls_from_dom(self, soup: BeautifulSoup, page_url: str) -> set[str]:
        candidates: set[str] = set()
        for element in soup.find_all(attrs={"data-config": True}):
            config_path = element.get("data-config")
            if config_path:
                candidates.add(urljoin(page_url, config_path))
        for element in soup.find_all(attrs={"data-config-url": True}):
            config_path = element.get("data-config-url")
            if config_path:
                candidates.add(urljoin(page_url, config_path))
        for element in soup.find_all("link", href=True):
            rel = (element.get("rel") or [])
            if any(value.lower() in {"preload", "alternate"} for value in rel):
                href = element.get("href")
                if href:
                    candidates.add(urljoin(page_url, href))
        return candidates

    def _combine_base_and_file(self, base: Optional[str], file_value: str) -> Optional[str]:
        if not file_value:
            return None
        file_value = file_value.strip()
        if file_value.startswith(("http://", "https://", "rtsp://")):
            return file_value
        if file_value.startswith("//"):
            return f"https:{file_value}"
        if not base:
            return None
        base = base.strip()
        if not base:
            return None
        if base.startswith("//"):
            base = f"https:{base}"
        return urljoin(base if base.endswith("/") else f"{base}/", file_value.lstrip("/"))

    def _infer_protocol(self, url: str) -> str:
        lowered = url.lower()
        if lowered.startswith("rtsp://"):
            return "rtsp"
        if ".m3u8" in lowered:
            return "hls"
        if ".mpd" in lowered:
            return "dash"
        if any(ext in lowered for ext in (".mjpg", ".mjpeg")):
            return "mjpeg"
        if any(ext in lowered for ext in (".mp4", ".webm", ".flv", ".ts", ".mpg")):
            return "http"
        return "http"
