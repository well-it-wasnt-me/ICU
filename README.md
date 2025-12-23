# ICU (I See You)

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/well-it-wasnt-me/icu/main.yml)
![Docs](https://readthedocs.org/projects/icu-i-see-you/badge/?version=latest)

![ICU-Logo](docs/icu.jpg)
## Introduction
Well, folks, if you are here then chances are you're looking to find people on cameras live stream. 
Lucky for you, I made this thing to make you become the king of the stalker.

Real-time computer-vision toolkit for monitoring camera streams with face recognition, optional licence-plate detection, and Telegram notifications/commands.

## Features
- Face recognition on live RTSP/HTTP streams (or local devices) using KNN embeddings
- Optional licence-plate OCR with watchlists, cooldowns, and capture summaries
- Telegram alerts with side-by-side comparisons plus remote `/add_poi` and `/add_plate`
- Stream finder that scrapes public test feeds (`--find-camera`)
- Textual TUI for local monitoring and runtime controls
- MkDocs-powered documentation with API reference

## Quickstart
```bash
git clone https://github.com/well-it-wasnt-me/ICU.git
cd ICU
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp cameras-example.yaml configs/cameras.yaml
cp configs/app-example.yaml configs/app.yaml
```

Add at least one person of interest under `poi/`:

```text
poi/
└── Alice/
    ├── image_1.jpg
    ├── image_2.jpg
    └── image_3.jpg
```

Train the classifier (append `--use_gpu` if you have a supported GPU):

```bash
python main.py --train --train_dir poi --model_save_path trained_knn_model.clf
```

Start monitoring:

```bash
python main.py \
  --camera_config configs/cameras.yaml \
  --app_config configs/app.yaml \
  --model_save_path trained_knn_model.clf
```

Captures and comparison frames are written to `captures/<camera>/`.

## Configuration overview
`configs/cameras.yaml` contains only stream definitions:

```yaml
cameras:
  - name: "Laptop WebCam"
    stream_url: "0"
    process_frame_interval: 15
    capture_cooldown: 120
```

`configs/app.yaml` holds app-wide settings, logging, notifications, and the plate monitor:

```yaml
settings:
  target_processing_fps: 2.0
  cpu_pressure_threshold: 85.0

notifications:
  telegram:
    bot_token: "123456:ABC"  # Do not commit real tokens
    chat_id: "123456789"
    enable_commands: true
    timeout: 10
    max_workers: 2
    command_poll_timeout: 20

logging:
  level: INFO
  file: face_recognition.log

plates:
  enabled: false
  watchlist: []
  watchlist_file: configs/plates_watchlist.txt
  alert_on_watchlist: true
  alert_on_every_plate: false
  capture_cooldown: 30
  storage:
    base_dir: captures/plates
    summary_file: plates_summary.json
    max_captures_per_plate: 20
  ocr:
    languages: ["en"]
    use_gpu: false
```

Enable the optional `plates` block to receive OCR-based alerts. Watchlist checks are case-insensitive and stored captures are grouped by camera/plate with a JSON summary.

## Telegram notifications and commands
1. [Create a bot](https://core.telegram.org/bots#3-how-do-i-create-a-bot) with `@BotFather`, then send the bot a message.
2. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` to read the numeric `chat.id`.
3. Paste `bot_token` and `chat_id` into `configs/app.yaml` under `notifications.telegram`.
4. Run ICU normally; the bot sends detections and listens for commands.

Supported commands include `/add_poi`, `/add_plate`, `/status`, `/list_poi`, `/done`, `/cancel`, and `/help`. With commands enabled you can upload POI images directly from Telegram and trigger an immediate retrain without touching the filesystem.

## Finding public test streams
Use the built-in scraper to generate camera config entries:

```bash
python main.py --find-camera
```

Provide a city or 2–3 letter country code (for the broader Pantheon crawl). Results are written to `camera_streams_<city>.yaml` in the same shape as `configs/cameras.yaml`.

## Documentation
Browse the full docs at [Read the Docs](https://icu-i-see-you.readthedocs.io/). To preview locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

## Known issues
- The confidence score is inverted (lower value = higher confidence).

## Roadmap
1. [x] Telegram and other notifications
2. [x] Resource optimisation
3. [ ] Integrate [Nerve](https://github.com/evilsocket/nerve) with a custom tasklet
4. [x] Automatic documentation generation

## Contributing
Pull requests and issue reports are welcome. Please keep tokens and private streams out of commits.

---
**Disclaimer:** Use ICU responsibly. You are solely accountable for how you capture, store, and share video or biometric data.
