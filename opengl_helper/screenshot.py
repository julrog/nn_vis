import os
import numpy as np
from datetime import datetime, timezone

from OpenGL.GL import *
from PIL import Image

from definitions import SCREENSHOT_PATH


def create_screenshot(width: int, height: int, network_name: str = None):
    glReadBuffer(GL_FRONT)
    pixel_data = np.frombuffer(glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE))
    image: Image = Image.frombuffer("RGBA", (width, height), pixel_data)

    time_key: str = datetime.utcfromtimestamp(
        datetime.timestamp(datetime.now().replace(tzinfo=timezone.utc).astimezone())).strftime(
        '%Y-%m-%d_%H_%M_%S')

    os.makedirs(SCREENSHOT_PATH, exist_ok=True)
    if network_name is None:
        image.save(SCREENSHOT_PATH + "network_" + time_key + ".png")
    else:
        image.save(SCREENSHOT_PATH + network_name + "_" + time_key + ".png")
