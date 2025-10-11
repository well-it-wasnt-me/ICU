import json
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


def test_earthcam_scraper_parses_search_item_cards(monkeypatch):
    listing_html = """
    <div class="searchItem" id="result_cam">
        <div class="imgContainer">
            <a onclick="urchinTracker('/outgoing/searchresults/EarthCam')" href="https://www.earthcam.com/world/ireland/dublin/?cam=dublinpub" target="_self" data-earthcam="yes" data-cam="abc">
                <img src="https://static.earthcam.com/camshots/example.jpg" alt="EarthCam: Dublin Pub Cam">
            </a>
        </div>
        <div class="cam_name">
            <a title="EarthCam: Dublin Pub Cam" href="https://www.earthcam.com/world/ireland/dublin/?cam=dublinpub" target="_self" class="camTitle" data-earthcam="yes" data-cam="abc">
                <span>EarthCam: Dublin Pub Cam</span>
            </a>
        </div>
    </div>
    """
    detail_html = '<video data-hls="https://videos-3.earthcam.com/fecnetwork/dublin.m3u8"></video>'

    search_url = "https://www.earthcam.com/search/ft-search.php?term=Dublin"
    detail_url = "https://www.earthcam.com/world/ireland/dublin/?cam=dublinpub"

    responses = {
        search_url: DummyResponse(listing_html),
        detail_url: DummyResponse(detail_html),
    }

    call_counts = {}

    def fake_get(url, timeout, verify, headers=None):
        call_counts[url] = call_counts.get(url, 0) + 1
        return responses[url]

    finder = CameraStreamFinder(timeout=1, scrapers=(EarthCamScraper(),))
    monkeypatch.setattr(finder.session, "get", fake_get)

    streams = finder.find_streams("Dublin", max_pages=1, max_results=2)
    assert streams
    assert streams[0].url.endswith("dublin.m3u8")
    assert streams[0].protocol == "hls"
    assert call_counts[detail_url] == 1


def test_earthcam_scraper_extracts_streams_from_json_config(monkeypatch):
    listing_html = """
    <div class="searchItem">
        <a href="https://www.earthcam.com/world/ireland/dublin/?cam=dublinpub" data-earthcam="yes">Cam</a>
    </div>
    """
    detail_html = """
    <script>
        window.playerConfig = {
            configUrl: "/config/cam_dublinpub.json"
        };
    </script>
    """
    config_payload = json.dumps(
        {
            "streams": [
                {
                    "base": "https://videos-3.earthcam.com/fecnetwork/dublinpub.stream",
                    "file": "playlist.m3u8",
                    "default": True,
                }
            ]
        }
    )

    search_url = "https://www.earthcam.com/search/ft-search.php?term=Dublin"
    detail_url = "https://www.earthcam.com/world/ireland/dublin/?cam=dublinpub"
    config_url = "https://www.earthcam.com/config/cam_dublinpub.json"

    responses = {
        search_url: DummyResponse(listing_html),
        detail_url: DummyResponse(detail_html),
        config_url: DummyResponse(config_payload),
    }

    def fake_get(url, timeout, verify, headers=None):
        return responses[url]

    finder = CameraStreamFinder(timeout=1, scrapers=(EarthCamScraper(),))
    monkeypatch.setattr(finder.session, "get", fake_get)

    streams = finder.find_streams("Dublin", max_pages=1, max_results=3)
    assert streams
    assert streams[0].url == "https://videos-3.earthcam.com/fecnetwork/dublinpub.stream/playlist.m3u8"
    assert streams[0].protocol == "hls"


def test_earthcam_scraper_handles_iframe_and_config(monkeypatch):
    listing_html = """
    <div class="tiny-single-row">
        <a href="/usa/test/camera2/">Cam 2</a>
    </div>
    """
    detail_html = """
    <iframe src="/embedded/player.html"></iframe>
    <div data-config="/api/config.json"></div>
    """
    iframe_html = '<script>var playlist = "https://videos-3.earthcam.com/fecnetwork/iframe_stream.m3u8"</script>'
    config_payload = '{"stream":"https://cdn.example.com/live/another_stream.m3u8"}'

    search_url = "https://www.earthcam.com/search/ft-search.php?term=Test%20City"
    detail_url = "https://www.earthcam.com/usa/test/camera2/"
    iframe_url = "https://www.earthcam.com/embedded/player.html"
    config_url = "https://www.earthcam.com/api/config.json"

    responses = {
        search_url: DummyResponse(listing_html),
        detail_url: DummyResponse(detail_html),
        iframe_url: DummyResponse(iframe_html),
        config_url: DummyResponse(config_payload),
    }

    headers_seen = {}

    def fake_get(url, timeout, verify, headers=None):
        headers_seen.setdefault(url, []).append(headers or {})
        return responses[url]

    finder = CameraStreamFinder(timeout=1, scrapers=(EarthCamScraper(),))
    monkeypatch.setattr(finder.session, "get", fake_get)

    streams = finder.find_streams("Test City", max_pages=1, max_results=5)
    urls = sorted(stream.url for stream in streams)

    assert urls == [
        "https://cdn.example.com/live/another_stream.m3u8",
        "https://videos-3.earthcam.com/fecnetwork/iframe_stream.m3u8",
    ]
    assert headers_seen[iframe_url][0]["Referer"] == detail_url
    assert headers_seen[config_url][0]["X-Requested-With"] == "XMLHttpRequest"


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
