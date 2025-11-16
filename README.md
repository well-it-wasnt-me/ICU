# ICU - (I See You)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/well-it-wasnt-me/icu/main.yml)
![Docs](https://readthedocs.org/projects/icu-i-see-you/badge/?version=latest)

![ICU-Logo](docs/icu.jpg)
## Introduction
Well, folks, if you are here then chances are you're looking to find people on cameras live stream. 
Lucky for you, I made this thing to make you become the king of the stalker.

## Installation
Now, installing this script is easier than forgetting your ex's birthday. 

But just in case you're the type who needs a roadmap to get out of a paper bag, here's how you do it:

### Step 1: Clone the Repository
First things first, let's get this show on the road. Open up your terminal...you know, that black box where magic happens...and run:

```bash
    git clone https://github.com/well-it-wasnt-me/ICU.git
    cd ICU
```

### Step 2: Set Up a Python Virtual Environment
Because isolating problems is the first step to solving them, or so my therapist says, we're gonna set up a virtual environment. 

Think of it as a playpen for your Python packages.
```bash
  python3 -m venv venv
```

Activate the virtual environment:

* On Unix or macOS:
```bash
  source venv/bin/activate
```

* On Windows:
```bash
  venv\Scripts\activate
```

### Step 3: Install Dependencies
Now, let's install the necessary packages. 

Without these, the script is about as useful as a chocolate teapot.

```bash
  pip install -r requirements.txt
```

If you see a bunch of text flying by, that's good. It means stuff is happening.

## Configuration
Before you run the script, you need to set up the configuration files. 

Don't worry...it's easier than assembling IKEA furniture.
### Create the configuration files

Copy the sample files into `configs/` and edit them to match your setup:

```bash
cp cameras-example.yaml configs/cameras.yaml
cp configs/app-example.yaml configs/app.yaml
```

`configs/cameras.yaml` holds only the camera definitions:

```yaml
cameras:
  - name: "Laptop WebCam"
    stream_url: "0"
    process_frame_interval: 15
    capture_cooldown: 120
```

While `configs/app.yaml` keeps application-wide settings and integrations:

```yaml
settings:
  target_processing_fps: 2.0
  cpu_pressure_threshold: 85.0

notifications:
  telegram:
    bot_token: "123456:ABC"  # Never commit the real token!
    chat_id: "123456789"
    timeout: 10
    max_workers: 2
    enable_commands: true    # Allow Telegram add_poi workflow
    command_poll_timeout: 20 # Seconds to wait when long-polling updates

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
    languages:
      - en
    use_gpu: false
```

`settings` tunes the per-camera processing rate, and `notifications.telegram` holds the bot credentials used to send alerts whenever a known person is detected.

When configured, Telegram alerts include both the captured frame and the comparison image so you can verify the match without digging into the filesystem.

Enabling the optional `plates` block turns on automatic licence plate detection via EasyOCR. Specify a `watchlist` of plate numbers (case-insensitive) you want to be alerted about. Captured plates are stored under `captures/plates/<camera>/<plate>/` and summarised in `plates_summary.json`, including the number of times each plate has been seen. Set `alert_on_every_plate` to `true` if you want Telegram notifications for every plate, or keep it `false` to only notify on watchlist hits. The first run will download EasyOCR models – make sure outbound network access is available.

### Enable Telegram Notifications

1. [Create a bot](https://core.telegram.org/bots#3-how-do-i-create-a-bot) with `@BotFather` and copy the API token it gives you.
2. Send any message to your new bot (or add it to a private channel) so Telegram creates a chat for it.
3. Open `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser (or run `curl`) to read the JSON payload that contains the numeric `chat` → `id` for that conversation. That is the `chat_id`.
4. Fill in the `notifications.telegram` block inside `configs/app.yaml`:

   ```yaml
   notifications:
     telegram:
       bot_token: "123456:ABCDEF"
       chat_id: "123456789"
       enable_commands: true
       timeout: 10
       max_workers: 2
   ```

5. Launch ICU normally (`python main.py`). When the bot token and chat id are valid, the app automatically spins up a background poller that listens for commands such as `/add_poi`, `/status`, `/list_poi`, `/done`, `/cancel` and `/help`.

If you only want outbound notifications (no command handler) set `enable_commands: false`.

### Hunting For Public Cameras

Need some streams to test with? Run the helper to scrape public (already exposed) feeds:

```bash
python main.py --find-camera
```

You'll be asked for a city name, and the script will query a couple of public indexes (RTSP and HLS). Any results are echoed to the console and saved under `camera_streams_<city>.yaml`, including protocol and any required headers so you can copy/paste them straight into `configs/cameras.yaml`. Remember that these feeds are public because someone left them exposed—use them responsibly.

## Usage
Alright, you've made it this far without breaking anything, kudos to you ! Time to run the script:

### 1st Step: TRAINING
In order for the script to find people needs to learn their faces. Duuuu.

So. just create inside the folder **poi** the following:
```md
ICU
├── poi
│   └── FOLDER_PERSON_NAME
│       ├── image_1.jpg
│       ├── image_2.jpg
│       ├── image_3.jpg
│       └── image_4.jpg # the more the better
└── main.py
```

Now it's time to train our model.

**NOTE**: if you have a GPU feel free to add `--use_gpu` at the end

```bash
  python main.py --train --train_dir poi --model_save_path trained_knn_model_gpu.clf
```
If all went fine you should see something like this:

```log
[INFO] Converted poi/PERSON_NAME/image_1.jpeg to RGB.
[INFO] Converted poi/PERSON_NAME/image_2.jpeg to RGB.
[INFO] Converted poi/PERSON_NAME_2/image_1.jpeg to RGB.
[INFO] Using device: mps
[INFO] Training KNN classifier...
[INFO] Using device for training: mps
[INFO] Chose n_neighbors automatically: 3
[INFO] KNN classifier saved to trained_knn_model_gpu.clf
[INFO] Training complete!
```

### 2nd Step: RUN
Remember: use the `--use_gpu` only if you have a GPU to use

```bash
python main.py \
  --camera_config configs/cameras.yaml \
  --app_config configs/app.yaml \
  --model_save_path trained_knn_model_gpu.clf
```
If all went ok you should see something like this:

```log
2024-12-25 16:23:32,540 [INFO] Converted poi/PERSON_NAME_A/image_1.jpeg to RGB.
.... more as above ...
2024-12-25 16:23:32,865 [INFO] Using device: mps
2024-12-25 16:23:32,865 [INFO] Loaded KNN classifier from trained_knn_model_gpu.clf
2024-12-25 16:23:33,190 [INFO] [Laptop WebCam] Connecting to the video stream...
2024-12-25 16:23:33,191 [INFO] [Laptop WebCam] CONNECTED. Starting Analysis process...
2024-12-25 16:23:46,888 [INFO] [Laptop WebCam] Detected known face: PERSON_NAME at 8%
2024-12-25 16:23:46,905 [INFO] [Laptop WebCam] Saved captured frame to captures/Laptop WebCam/Laptop WebCam_PERSON_NAME_20241225_162346.jpg
2024-12-25 16:23:47,098 [INFO] [Laptop WebCam] Saved side-by-side screenshot to captures/Laptop WebCam/Laptop WebCam_PERSON_NAME_20241225_162346_sidebyside.jpg
```

Everytime the script will find a face not only you'll find a screenshot of that frame inside the folder captures but also a side by side view of
what the camera is seeing and the image used for the match.

### Add POIs from Telegram

With Telegram notifications enabled, you can add new people-of-interest without touching the filesystem:

1. Send `/add_poi` to your bot (or `/add_poi Alice` to skip the name prompt).
2. The bot replies `name` — answer with the display name for the person if you didn't provide it inline.
3. The bot then asks for `picture(s)` — upload one or more photos (or screenshots) for that person.
4. When you're done, reply with `done` (or tap `/done`). Send `/cancel` at any point to abort.
5. The bot will then offer to retrain immediately; reply `yes` to stop streams, retrain, and restart ICU with the new model (or `no` to handle it later).

The images are saved under the configured training directory (default `poi/<NAME>`), converted to RGB, and ready for the next training run. When you confirm the retrain step, ICU pauses streaming, rebuilds the classifier, and relaunches itself so detections include the new POI right away.

Extra Telegram commands for convenience:

- `/add_poi [Name]` — start a session (include the name inline to skip the prompt).
- `/add_plate <Plate>` — append a licence plate to the watchlist (persisted to `configs/plates_watchlist.txt`).
- `/status` — show the current progress of the add_poi session (name, photos, timestamps, next step).
- `/list_poi` — list the already-stored POIs from disk so you know what exists.
- `/done` — finish the current session once you're done uploading images.
- `/cancel` — abort the current session and discard temporary files.
- `/help` — show a cheat sheet with all supported commands.

### Plate Monitor & Watchlist

ICU can also look for licence plates inside each camera frame:

1. Edit `configs/app.yaml` and set `plates.enabled: true`. The sample config already contains sensible defaults for storage, watchlists, cooldowns, and OCR options.
2. Populate the `plates.watchlist` array with plate numbers you care about. Matching is case-insensitive and ignores spaces/dashes.
3. Point `watchlist_file` at a writable path (defaults to `configs/plates_watchlist.txt`). The bot appends new entries here so they survive restarts.
4. (Optional) Flip `alert_on_every_plate: true` to receive notifications for every detection, not just watchlist hits. You can also disable plate messages entirely by setting `plates.use_notifications: false`.
5. Start ICU as usual. The first run downloads OCR models (EasyOCR) so expect a longer setup the first time.
6. When a plate is detected, ICU writes annotated captures plus crops under `captures/plates/<camera>/<plate>/` and appends metadata to `plates_summary.json`. If Telegram notifications are enabled, the bot posts the capture/crop plus whether the plate was on your watchlist.

Need to add a plate while you’re away from the keyboard? Send `/add_plate <PlateNumber>` to the bot. The plate is normalised (uppercase, punctuation removed), added to the active watchlist immediately, and persisted to `watchlist_file` so it’s still there after a restart. Use `/list_poi` separately to keep tabs on face-training folders.

## Script Arguments
```bash
$ python main.py --help

usage: main.py [-h] [--train_dir TRAIN_DIR] [--model_save_path MODEL_SAVE_PATH] [--n_neighbors N_NEIGHBORS] [--config CONFIG] [--distance_threshold DISTANCE_THRESHOLD]
               [--train] [--use_gpu]

Face Recognition from Live Camera Stream

options:
  -h, --help            show this help message and exit
  --train_dir TRAIN_DIR
                        Directory with training images
  --model_save_path MODEL_SAVE_PATH
                        Path to save/load KNN model
  --n_neighbors N_NEIGHBORS
                        Number of neighbors for KNN - integer
  --config CONFIG       Path to YAML config
  --distance_threshold DISTANCE_THRESHOLD - float
                        Distance threshold
  --train               Train the model
  --use_gpu             Use GPU with facenet-pytorch

## Documentation

The documentation is now powered by [MkDocs](https://www.mkdocs.org/) +
[mkdocstrings](https://mkdocstrings.github.io/). To preview it locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Point your browser to the address shown in the terminal (defaults to
http://127.0.0.1:8000/) to browse the live docs. Use `mkdocs build` to produce a
static site inside `site/` for deployment.

## Known Issues
* The confidence level shown is "backward" (lower value = higher accuracy...I know...I should stop drinking)

## Roadmap
1. [X] Implement notification (telegram and/or other)  
2. [X] Resource optimization
3. [ ] Integrate [Nerve](https://github.com/evilsocket/nerve) with custom tasklet 
3. [X] Automatic Doc Generation 

## Conclusion
Well, that's all she wrote. If you run into any issues, feel free to do nothing and don't bother me.

If you plan to contribute, I'll be happy to accept your PRs.

Happy stalking !

---------------------------
**Disclaimer:** _Use this script responsibly. And remember, with great power comes great responsibility...
or at least that's what they told me before I broke the production server._
