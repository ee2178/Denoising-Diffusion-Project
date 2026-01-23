import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils import saveimg
from torchvision.transforms.functional import to_tensor
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type = str, help="Corresponding path where image files can be found", default = None)
parser.add_argument("--new_path", type = str, help="New name for path", default = None)
args = parser.parse_args()


def main(args):
    # Load img at image path
    img = to_tensor(Image.open(args.image_path))
    if args.new_path is None:
        new_path = Path(args.image_path).stem + "_contrast.png"
    else:
        new_path = args.new_path
    saveimg(img[0], new_path, contrast = True)

if __name__ == "__main__":
    main(args)

