"""
Image Utilities Module.

Provides utility functions for image processing such as resampling,
conversion to RGB, adding timestamps, and creating side-by-side screenshots.
"""

import os
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from logger_setup import logger

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}


class ImageUtils:
    """
    A collection of static methods for various image processing tasks.
    """

    @staticmethod
    def custom_imresample(img, sz):
        """
        Resample an image tensor to a given size using bilinear interpolation.

        :param img: The input image tensor.
        :param sz: The target size (height, width).
        :return: The resampled image tensor.
        """
        return torch.nn.functional.interpolate(
            img, size=sz, mode="bilinear", align_corners=False
        )

    @staticmethod
    def convert_images_to_rgb(directory):
        """
        Convert all images in the specified directory to RGB format.

        This function walks through the directory and its subdirectories, converting images
        with allowed extensions to ensure consistent color channels.

        :param directory: The root directory to search for images.
        """
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                    path = os.path.join(root, file)
                    try:
                        img = Image.open(path).convert('RGB')
                        img.save(path)
                        logger.info(f"Converted {path} to RGB.")
                    except Exception as e:
                        logger.error(f"Failed to convert {path}: {e}")

    @staticmethod
    def add_timestamp(frame):
        """
        Add a timestamp overlay to an image frame.

        The timestamp is added at the bottom-left corner of the image.

        :param frame: The input image frame as a numpy array.
        :return: The image frame with a timestamp overlay.
        """
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            logger.error("Error with fonts... using default font")
            font = ImageFont.load_default()

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        bbox = font.getbbox(timestamp)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = 10
        y = pil_image.height - text_height - 10

        # Draw a rectangle behind the text for better readability
        draw.rectangle((x - 5, y - 5, x + text_width + 5, y + text_height + 5),
                       fill=(0, 0, 0))
        draw.text((x, y), timestamp, fill=(255, 255, 255), font=font)

        del draw
        return np.array(pil_image)

    @staticmethod
    def create_side_by_side_screenshot(frame, ref_image, camera_name, person_name, confidence):
        """
        Create a side-by-side screenshot of the input frame and a reference image.

        The function resizes the images, places them next to each other on a white canvas,
        and overlays textual information including camera name, person name, confidence, and timestamp.

        :param frame: The current frame from the camera (BGR format).
        :param ref_image: The reference image (BGR format) for the recognized person.
                          If None or empty, a blank image is used.
        :param camera_name: Name identifier for the camera.
        :param person_name: Name of the recognized person.
        :param confidence: Confidence level of the face recognition (percentage).
        :return: The combined image as a numpy array in BGR format.
        """
        final_w, final_h = 800, 600
        canvas = Image.new("RGB", (final_w, final_h), (255, 255, 255))

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_resized = frame_pil.resize((400, 300), Image.Resampling.LANCZOS)

        # If no reference image is found, use a blank white image
        if ref_image is None or ref_image.size == 0:
            ref_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        ref_pil = Image.fromarray(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
        ref_resized = ref_pil.resize((400, 300), Image.Resampling.LANCZOS)

        canvas.paste(frame_resized, (0, 0))
        canvas.paste(ref_resized, (400, 0))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            logger.error("Error with fonts... using default font")
            font = ImageFont.load_default()

        lines = [
            f"Camera: {camera_name}",
            f"Person: {person_name}",
            f"Confidence: {int(round(confidence))}%",
            f"Datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]

        y_offset = 310
        for line in lines:
            bbox = font.getbbox(line)
            text_height = bbox[3] - bbox[1]
            draw.text((10, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += text_height + 5

        combined_np = np.array(canvas)
        combined_np = cv2.cvtColor(combined_np, cv2.COLOR_RGB2BGR)
        return combined_np
