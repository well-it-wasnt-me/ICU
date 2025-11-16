# Usage

Alright, you’ve made it this far without breaking anything — kudos to you!

## 1. Training

In order for the script to find people it needs to learn their faces. So create
the following structure inside the `poi` folder:

```text
ICU
├── poi
│   └── FOLDER_PERSON_NAME
│       ├── image_1.jpg
│       ├── image_2.jpg
│       ├── image_3.jpg
│       └── image_4.jpg  # the more the better
└── main.py
```

Now it’s time to train the model.

> **NOTE**: if you have a GPU feel free to add `--use_gpu` at the end.

```bash
python main.py --train --train_dir poi --model_save_path trained_knn_model_gpu.clf
```

If all went fine you should see something like this:

```text
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

## 2. Run

Remember: use the `--use_gpu` flag only if you have a GPU available.

```bash
python main.py \
  --camera_config configs/cameras.yaml \
  --app_config configs/app.yaml \
  --model_save_path trained_knn_model_gpu.clf
```

If all went ok you should see something like this:

```text
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

Every time the script finds a face not only will you find a screenshot of that
frame inside the `captures` folder but also a side-by-side view of what the
camera is seeing and the image used for the match.

## Telegram Notifications Setup

Want Telegram alerts (and interactive uploads)? Set it up once:

1. Create a bot with `@BotFather` and keep the token it prints.
2. Send any message to the bot (or add it to a private channel) so Telegram creates a chat.
3. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` after sending a message; the JSON response contains the numeric `chat.id` you need.
4. Configure `configs/app.yaml`:

   ```yaml
   notifications:
     telegram:
       bot_token: "123456:ABCDEF"
       chat_id: "123456789"
       enable_commands: true
       timeout: 10
       max_workers: 2
   ```

5. Start ICU with `python main.py`. The Telegram poller starts automatically when both token and chat id are present. Set `enable_commands: false` if you only want outbound notifications.

## Telegram Commands

Once the bot is running, these commands are always available:

- `/add_poi [Name]` — start collecting files for a new person of interest (name is optional inline).
- `/add_plate <Plate>` — append a licence plate to the watchlist (stored on disk if `watchlist_file` is set).
- `/status` — display the current session state (stage, photos saved, etc.).
- `/list_poi` — list stored POI folders and file counts on disk.
- `/done` — finish the current upload session.
- `/cancel` — abort the current session and discard temporary files if needed.
- `/help` — print the quick reference with all commands.

## Add POIs from Telegram

With Telegram notifications enabled, you can add new people of interest straight
from chat:

1. Send `/add_poi` to your bot (or `/add_poi Alice` to skip the name prompt).
2. The bot replies `name` — answer with the person’s display name if you didn’t provide it inline.
3. The bot asks for `picture(s)` — upload one or more photos.
4. When finished, reply with `done` (or tap `/done`). Use `/cancel` to abort at any point.
5. The bot will ask whether to retrain immediately — reply `yes` to stop streams,
   retrain, and restart ICU with the updated model (or `no` to handle it later).

The images are written to the training directory (`poi/<NAME>` by default),
converted to RGB, and ready for your next training run. Confirming the retrain
prompt triggers an automatic stop → retrain → restart sequence so the new person
is recognised immediately.

## Plate Monitor & Watchlist

ICU can also watch for licence plates:

1. Toggle `plates.enabled: true` inside `configs/app.yaml`.
2. Populate `plates.watchlist` with any plates you care about. Matching ignores case and punctuation.
3. Point `plates.watchlist_file` to a writable path (defaults to `configs/plates_watchlist.txt`). Telegram commands append to this file so updates persist.
4. Adjust the rest of the block as needed:

   ```yaml
   plates:
     enabled: true
     watchlist:
       - ABC123
       - XYZ999
     watchlist_file: configs/plates_watchlist.txt
     alert_on_watchlist: true
     alert_on_every_plate: false
     use_notifications: true
     capture_cooldown: 30
     storage:
       base_dir: captures/plates
       summary_file: plates_summary.json
   ```

5. Start ICU. The plate recognizer downloads OCR models on its first run and then processes every frame for plates.

Each hit writes annotated frames and crops to `captures/plates/<camera>/<plate>/` and appends metadata to the summary JSON. When Telegram notifications plus `plates.use_notifications` are enabled, the bot posts the capture, the crop, and whether the plate was on your watchlist so you can react immediately.

Need to add someone new while you’re away from the computer? Send `/add_plate <PlateNumber>` to the bot. The value is normalised (uppercase, punctuation removed), applied to the live watchlist instantly, and written to `watchlist_file` so it remains active after restarts.

## Finding Public Streams

Need cameras to watch? Run `python main.py --find-camera` and provide a city
name. ICU uses `stream_finder.CameraStreamFinder` to query both Insecam and
EarthCam, retries Insecam pages that respond with HTTP 403, follows EarthCam
iframe embeds and JSON configs, and writes the consolidated results to
`camera_streams_<city>.yaml`. Each entry includes the protocol and any headers
(such as `Referer`) that you should copy into `configs/cameras.yaml` before
monitoring the stream.

## ICU Arguments

```bash
python main.py --help
```

```
usage: main.py [-h] [--train_dir TRAIN_DIR]
               [--model_save_path MODEL_SAVE_PATH] [--n_neighbors N_NEIGHBORS]
               [--camera_config CAMERA_CONFIG] [--app_config APP_CONFIG]
               [--distance_threshold DISTANCE_THRESHOLD]
               [--train] [--use_gpu]
               [--target_processing_fps TARGET_PROCESSING_FPS]
               [--cpu_pressure_threshold CPU_PRESSURE_THRESHOLD]
               [--find-camera]

Face Recognition from Live Camera Stream

options:
  -h, --help            show this help message and exit
  --train_dir TRAIN_DIR
                        Directory with training images
  --model_save_path MODEL_SAVE_PATH
                        Path to save/load KNN model
  --n_neighbors N_NEIGHBORS
                        Number of neighbors for KNN
  --camera_config CAMERA_CONFIG
                        Path to camera configuration file
  --app_config APP_CONFIG
                        Path to application configuration file
  --distance_threshold DISTANCE_THRESHOLD
                        Distance threshold for recognition
  --train               Train the model
  --use_gpu             Use GPU with facenet-pytorch
  --target_processing_fps TARGET_PROCESSING_FPS
                        Target processing rate per camera (0 disables rate
                        limiting)
  --cpu_pressure_threshold CPU_PRESSURE_THRESHOLD
                        CPU usage threshold to trigger adaptive throttling
  --find-camera         Interactively search for public camera streams by city
```
