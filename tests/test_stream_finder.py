import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stream_finder import (
    CameraStreamFinder,
    DiscoveredStream,
    EarthCamScraper,
    InsecamScraper,
)


class DummyResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code


def test_insecam_scraper_parses_results(monkeypatch):
    listing_html = """
    <a href="/en/view/123">Camera 1</a>
    <a href="/en/view/456">Camera 2</a>
    """
    detail_html_1 = 'var player = "rtsp://example.com/stream1";'
    detail_html_2 = 'data-src="rtsp://example.org/stream2"'

    responses = {
        "https://www.insecam.org/en/bycity/Test_City/": DummyResponse(listing_html),
        "https://www.insecam.org/en/view/123": DummyResponse(detail_html_1),
        "https://www.insecam.org/en/view/456": DummyResponse(detail_html_2),
    }

    verify_flags = []
    headers_seen = {}

    def fake_get(url, timeout, verify, headers=None):
        verify_flags.append(verify)
        headers_seen[url] = headers
        resp = responses.get(url)
        if resp is None:
            return DummyResponse("", status_code=404)
        return resp

    finder = CameraStreamFinder(timeout=1, scrapers=(InsecamScraper(),))
    monkeypatch.setattr(finder.session, "get", fake_get)

    streams = finder.find_streams("Test City", max_pages=1, max_results=5)
    urls = [stream.url for stream in streams]

    assert urls == ["rtsp://example.com/stream1", "rtsp://example.org/stream2"]
    assert all(flag is False for flag in verify_flags), "SSL verification should be disabled by default."
    assert all(stream.protocol == "rtsp" for stream in streams)
    assert "Mozilla" in finder.session.headers.get("User-Agent", ""), "Expected default User-Agent header."
    listing_headers = headers_seen["https://www.insecam.org/en/bycity/Test_City/"]
    assert listing_headers["Referer"].startswith("https://www.insecam.org")
    assert headers_seen["https://www.insecam.org/en/view/123"]["Referer"] == "https://www.insecam.org/en/bycity/Test_City/"


def test_earthcam_scraper_parses_results(monkeypatch):
    listing_html = """
    <div id="pagination_top">Your search Test City returned 1 results</div>
    <div class="col-xl-3 col-md-4 col-6 tiny-single-row g-0" background-color="#e6e6e6">
        <a href="/usa/test/camera1/">Cam 1</a>
    </div>
    """
    detail_html = 'source src="https://videos-3.earthcam.com/fecnetwork/test.m3u8"'

    responses = {
        "https://www.earthcam.com/search/ft-search.php?term=Test%20City": DummyResponse(listing_html),
        "https://www.earthcam.com/usa/test/camera1/": DummyResponse(detail_html),
    }

    headers_seen = []

    def fake_get(url, timeout, verify, headers=None):
        headers_seen.append(headers)
        return responses[url]

    finder = CameraStreamFinder(timeout=1, scrapers=(EarthCamScraper(),))
    monkeypatch.setattr(finder.session, "get", fake_get)

    streams = finder.find_streams("Test City", max_pages=1, max_results=3)
    assert streams
    assert streams[0].protocol == "hls"
    assert streams[0].headers and "Referer" in streams[0].headers
    assert streams[0].url.endswith(".m3u8")
    assert any(headers for headers in headers_seen), "Requests should include custom headers."


def test_camera_stream_finder_combines_scrapers(monkeypatch):
    class StubScraper:
        def __init__(self, outputs):
            self.outputs = outputs

        def find(self, finder, city, max_results, max_pages):
            for stream in self.outputs[:max_results]:
                yield stream

    scraper_one = StubScraper(
        [DiscoveredStream(url="rtsp://a", protocol="rtsp"), DiscoveredStream(url="rtsp://b", protocol="rtsp")]
    )
    scraper_two = StubScraper(
        [DiscoveredStream(url="https://hls.example/stream.m3u8", protocol="hls", headers={"Referer": "x"})]
    )

    finder = CameraStreamFinder(scrapers=(scraper_one, scraper_two))
    streams = finder.find_streams("City", max_results=3)
    assert len(streams) == 3
    assert streams[0].url == "rtsp://a"
    assert streams[-1].protocol == "hls"
