from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_utils import ImageUtils


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

