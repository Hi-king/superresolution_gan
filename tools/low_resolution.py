# -*- coding: utf-8 -*-
import argparse

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("output_path")
parser.add_argument("--resize", type=int, required=True)
args = parser.parse_args()

input_image = Image.open(args.input_path)  # type: Image.Image

if input_image.mode == 'RGBA':
    input_image.load()
    rgb_image = Image.new("RGB", input_image.size, (255, 255, 255))
    rgb_image.paste(input_image, mask=input_image.split()[3])
    input_image = rgb_image

resized_image = input_image.resize((args.resize, int(float(input_image.size[1]) * args.resize / input_image.size[0])))
resized_image.save(args.output_path)
