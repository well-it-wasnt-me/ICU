"""
Utilities to discover public camera streams by city.

Currently supports:
* insecam.org (RTSP feeds)
* earthcam.com (HLS feeds)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

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
        self.session.headers.setdefault(
            "User-Agent",
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
        )
        self.session.headers.setdefault(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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
        "Referer": "https://www.insecam.org/",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def find(
        self,
        finder: CameraStreamFinder,
        city: str,
        max_results: int,
        max_pages: int,
    ) -> Iterable[DiscoveredStream]:
        city_slug = city.strip().replace(" ", "_")
        found = 0
        for page in range(1, max_pages + 1):
            page_url = self._build_listing_url(city_slug, page)
            logger.info("Fetching insecam listing: %s", page_url)
            listing_html = finder.fetch(page_url, headers=self.BASE_HEADERS)
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


class EarthCamScraper(BaseCameraScraper):
    SEARCH_URL_TEMPLATE = "https://www.earthcam.com/search/ft-search.php?term={query}"
    M3U8_REGEX = re.compile(r"https://[^\s\"']+\.m3u8[^\s\"']*")
    EARTHCAM_HEADERS = {
        "Referer": "https://www.earthcam.com/",
        "Origin": "https://www.earthcam.com",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "*/*",
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
            for stream_url in self._extract_streams(detail_html):
                yield DiscoveredStream(
                    url=stream_url,
                    protocol="hls",
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

        for card in soup.select("div.tiny-single-row a[href]"):
            href = card.get("href")
            if href and not href.startswith("javascript:"):
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

    def _extract_streams(self, html: str) -> Iterable[str]:
        for match in self.M3U8_REGEX.findall(html):
            yield match
