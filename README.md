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
Before you run the script, you need to set up the `cameras.yaml` file. 

Don't worry...it's easier than assembling IKEA furniture.
### Create the cameras.yaml File

Just rename/copy `cameras-example.yaml` to `cameras.yaml` and fill in the blank !

```yaml
cameras:
  - name: "Laptop WebCam"
    stream_url: "0"
    process_frame_interval: 15
    capture_cooldown: 120

settings:
  enable_tui: true
  show_preview: false
  preview_scale: 0.5
  target_processing_fps: 2.0
  cpu_pressure_threshold: 85.0

notifications:
  telegram:
    bot_token: "123456:ABC"  # Never commit the real token!
    chat_id: "123456789"
    timeout: 10
    max_workers: 2
```

`settings` lets you keep dashboard/preview/tuning options alongside the camera list, while the `notifications.telegram` block holds the bot credentials used to send alerts whenever a known person is detected.

When configured, Telegram alerts include both the captured frame and the comparison image so you can verify the match without digging into the filesystem.

### Hunting For Public Cameras

Need some streams to test with? Run the helper to scrape public (already exposed) feeds:

```bash
python main.py --find-camera
```

You'll be asked for a city name, and the script will query a couple of public indexes (RTSP and HLS). Any results are echoed to the console and saved under `camera_streams_<city>.yaml`, including protocol and any required headers so you can copy/paste them straight into `cameras.yaml`. Remember that these feeds are public because someone left them exposed—use them responsibly.

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
python main.py --config cameras.yaml --model_save_path trained_knn_model_gpu.clf
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
```

## Known Issues
* The confidence level shown is "backward" (lower value = higher accuracy...I know...I should stop drinking)

## Roadmap
1. [X] Implement notification (telegram and/or other)  
2. [X] Resource optimization
3. [ ] Integrate [Nerve](https://github.com/evilsocket/nerve) with custom tasklet 
3. [ ] Automatic Doc Generation 

## Conclusion
Well, that's all she wrote. If you run into any issues, feel free to do nothing and don't bother me.

If you plan to contribute, I'll be happy to accept your PRs.

Happy stalking !

---------------------------
**Disclaimer:** _Use this script responsibly. And remember, with great power comes great responsibility...
or at least that's what they told me before I broke the production server._
