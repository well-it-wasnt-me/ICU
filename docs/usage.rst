Usage
=====

.. _usage:

Alright, you’ve made it this far without breaking anything, kudos to you
! Time to run the script:

1st Step: TRAINING
~~~~~~~~~~~~~~~~~~

In order for the script to find people needs to learn their faces.
Duuuu.

So. just create inside the folder **poi** the following:

.. code:: md

   ICU
   ├── poi
   │   └── FOLDER_PERSON_NAME
   │       ├── image_1.jpg
   │       ├── image_2.jpg
   │       ├── image_3.jpg
   │       └── image_4.jpg # the more the better
   └── main.py

Now it’s time to train our model.

**NOTE**: if you have a GPU feel free to add ``--use_gpu`` at the end

.. code:: bash

     python main.py --train --train_dir poi --model_save_path trained_knn_model_gpu.clf

If all went fine you should see something like this:

.. code:: bash

   [INFO] Converted poi/PERSON_NAME/image_1.jpeg to RGB.
   [INFO] Converted poi/PERSON_NAME/image_2.jpeg to RGB.
   [INFO] Converted poi/PERSON_NAME_2/image_1.jpeg to RGB.
   [INFO] Using device: mps
   [INFO] Training KNN classifier...
   [INFO] Using device for training: mps
   [INFO] Chose n_neighbors automatically: 3
   [INFO] KNN classifier saved to trained_knn_model_gpu.clf
   [INFO] Training complete!

2nd Step: RUN
~~~~~~~~~~~~~

Remember: use the ``--use_gpu`` only if you have a GPU to use

.. code:: bash

   python main.py \
     --camera_config configs/cameras.yaml \
     --app_config configs/app.yaml \
     --model_save_path trained_knn_model_gpu.clf

If all went ok you should see something like this:

.. code:: bash

   2024-12-25 16:23:32,540 [INFO] Converted poi/PERSON_NAME_A/image_1.jpeg to RGB.
   .... more as above ...
   2024-12-25 16:23:32,865 [INFO] Using device: mps
   2024-12-25 16:23:32,865 [INFO] Loaded KNN classifier from trained_knn_model_gpu.clf
   2024-12-25 16:23:33,190 [INFO] [Laptop WebCam] Connecting to the video stream...
   2024-12-25 16:23:33,191 [INFO] [Laptop WebCam] CONNECTED. Starting Analysis process...
   2024-12-25 16:23:46,888 [INFO] [Laptop WebCam] Detected known face: PERSON_NAME at 8%
   2024-12-25 16:23:46,905 [INFO] [Laptop WebCam] Saved captured frame to captures/Laptop WebCam/Laptop WebCam_PERSON_NAME_20241225_162346.jpg
   2024-12-25 16:23:47,098 [INFO] [Laptop WebCam] Saved side-by-side screenshot to captures/Laptop WebCam/Laptop WebCam_PERSON_NAME_20241225_162346_sidebyside.jpg

Everytime the script will find a face not only you’ll find a screenshot
of that frame inside the folder captures but also a side by side view of
what the camera is seeing and the image used for the match.

Add POIs from Telegram
~~~~~~~~~~~~~~~~~~~~~~

With Telegram notifications enabled, you can add new people of interest straight from chat:

1. Send ``add_poi`` to your bot.
2. The bot replies ``name`` — answer with the person’s display name.
3. The bot asks for ``picture(s)`` — upload one or more photos.
4. When finished, reply with ``done`` (or ``cancel`` to abort).

The images are written to the training directory (``poi/<NAME>`` by default), converted to RGB, and ready for your next training run. Retrain the KNN model to include the new person in live recognition.

Finding Public Streams
~~~~~~~~~~~~~~~~~~~~~~

Need cameras to watch? Run ``python main.py --find-camera`` and provide a city name.
ICU uses :class:`stream_finder.CameraStreamFinder` to query both Insecam and EarthCam,
retries Insecam pages that respond with HTTP 403, follows EarthCam iframe embeds and
JSON configs, and writes the consolidated results to ``camera_streams_<city>.yaml``.
Each entry includes the protocol and any headers (such as ``Referer``) that you should
copy into ``configs/cameras.yaml`` before monitoring the stream.

ICU Arguments
~~~~~~~~~~~~~

.. code:: bash

   $ python main.py --help

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
