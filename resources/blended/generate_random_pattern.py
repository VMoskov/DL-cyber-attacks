import argparse
from PIL import Image
import numpy as np
import random


def generate_random_pattern_image(color_channels):
    return np.random.randint(0, 256, (32, 32, color_channels), dtype=np.uint8)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--color_channels', type=int, default=3)
    args.add_argument('--output_path', type=str, default='./trigger_image_random.png')
    args = args.parse_args()

    image = generate_random_pattern_image(args.color_channels)

    mode = 'RGB' if args.color_channels == 3 else 'L'
    Image.fromarray(image, mode=mode).save(args.output_path)
