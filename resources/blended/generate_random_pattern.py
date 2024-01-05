import argparse
from PIL import Image
import numpy as np
import random


def generate_random_pattern_image(image_size, color_channels):
    if color_channels == 1:
        # Create a single-channel image (grayscale)
        random_image = np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)
    else:
        # Create a color image with random values
        random_image = np.random.randint(0, 256, (image_size, image_size, color_channels), dtype=np.uint8)
    return random_image


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--image_size', type=int, default=32)
    args.add_argument('--color_channels', type=int, default=3)
    args.add_argument('--output_path', type=str, default='./trigger_image_random.png')
    args = args.parse_args()

    image = generate_random_pattern_image(
        args.image_size,
        args.color_channels,
    )
    mode = 'RGB' if args.color_channels == 3 else 'L'
    Image.fromarray(image, mode=mode).save(args.output_path)
