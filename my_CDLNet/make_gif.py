import imageio.v2 as imageio
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type = str, help="Corresponding path where image files can be found", default = None)
parser.add_argument("--gifname", type = str, help="What to name your gif after saving", default = None)
args = parser.parse_args() 

def natural_sort_key(filename):
    # Extract numerical parts from the filename
    parts = [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', filename)]
    return parts

def make_gif(dir, gifname):
    filenames = sorted(os.listdir(dir), key = natural_sort_key)
    with imageio.get_writer(gifname, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(dir, filename))
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)

if __name__ == "__main__":
    dir = args.image_path
    gifname = args.gifname
    make_gif(dir, gifname)
