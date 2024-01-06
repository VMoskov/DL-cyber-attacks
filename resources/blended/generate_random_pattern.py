import argparse
from PIL import Image
import numpy as np
import random


def generate_random_pattern_image(color_channels):
    return np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--output_path', type=str, default='./trigger_image_random.png')
    args = args.parse_args()

    image = generate_random_pattern_image(args.color_channels)

    Image.fromarray(image, mode="RGB").save(args.output_path)
