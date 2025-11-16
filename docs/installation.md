# Installation & Configuration

## Installation

Now, installing this script is easier than forgetting your ex's birthday. But
just in case you're the type who needs a roadmap to get out of a paper bag,
here's how you do it.

### Repository Cloning

First things first, let's get this show on the road. Open up your terminal (you
know, that black box where magic happens) and run:

```console
$ git clone https://github.com/well-it-wasnt-me/ICU.git
$ cd ICU
```

### Set Up a Python Virtual Environment

Because isolating problems is the first step to solving them, or so my therapist
says, we're gonna set up a virtual environment. Think of it as a playpen for
your Python packages.

```console
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

## Configuration

Before you run the script, you need to set up the configuration files. Don't
worry — it's easier than assembling IKEA furniture.

Copy the samples and then fill in the blanks:

```console
$ cp cameras-example.yaml configs/cameras.yaml
$ cp configs/app-example.yaml configs/app.yaml
```

`configs/cameras.yaml` should contain only your camera definitions, while
`configs/app.yaml` keeps runtime settings (like throttling) and integrations
such as Telegram. You can also configure logging there:

```yaml
settings:
  target_processing_fps: 2.0
  cpu_pressure_threshold: 85.0

logging:
  level: INFO
  file: face_recognition.log
```
