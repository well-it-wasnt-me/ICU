from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_utils import ImageUtils
import numpy as np


def test_convert_to_rgb(tmp_path):
    """
    Create an image in a non-RGB mode and check that convert_images_to_rgb converts it to RGB.
    """
    # Create a dummy image in "L" (grayscale) mode
    img_path = tmp_path / "test_image.jpg"
    img = Image.new("L", (50, 50), color=128)
    img.save(img_path)

    # Convert the image in the directory (the function processes the entire directory)
    ImageUtils.convert_images_to_rgb(str(tmp_path))

    # Re-open the image file to check its mode
    converted_img = Image.open(img_path)
    assert converted_img.mode == "RGB", "The image was not converted to RGB."


def test_create_side_by_side_screenshot(tmp_path):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ref = np.ones((300, 300, 3), dtype=np.uint8) * 255

    combined = ImageUtils.create_side_by_side_screenshot(
        frame,
        ref,
        camera_name="Cam",
        person_name="Person",
        confidence=87.5,
    )

    # ImageUtils returns BGR ndarray sized 600x800
    assert combined.shape[:2] == (600, 800)
    # Both halves should not be all-zero after drawing
    left_half, right_half = combined[:, :400], combined[:, 400:]
    assert np.any(left_half != 0), "Left half should contain the frame plus overlays."
    assert np.any(right_half != 0), "Right half should contain the reference image plus overlays."
