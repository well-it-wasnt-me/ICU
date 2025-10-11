stream_finder module
====================

The :mod:`stream_finder` helpers make it easier to populate your ``cameras.yaml`` file.
They aggregate multiple public camera directories, normalize the responses, and return
ready-to-use stream definitions. Two providers are bundled:

- **Insecam** – requests are bootstrapped with a desktop browser profile and fall back
  to hardened headers when the site responds with HTTP 403, improving success rates.
- **EarthCam** – detail pages, embedded players, and JSON configs are parsed so that
  nested ``.m3u8`` playlists (and other manifests such as ``.mpd`` or ``.mp4``) surface
  automatically, even when the playlist is assembled from ``base``/``file`` elements inside
  EarthCam's player configuration JSON.

When you run ``python main.py --find-camera`` the application uses :class:`stream_finder.CameraStreamFinder`
to look up a city, deduplicate the results, and write them to ``camera_streams_<city>.yaml``.
Each entry includes the stream URL, protocol, and any HTTP headers (such as Referer) that
must be preserved when you add the stream to your configuration.

API Reference
-------------

.. automodule:: stream_finder
   :members:
   :undoc-members:
   :show-inheritance:
