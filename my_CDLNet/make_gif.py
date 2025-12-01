import imageio.v2 as imageio
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type = str, help="Corresponding path where image files can be found", default = None)
parser.add_argument("--gifname", type = str, help="What to name your gif after saving", default = None)
args = parser.parse_args() 

def make_gif(dir, gifname):
    filenames = os.listdir(dir)
    with imageio.get_writer(gifname, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(dir, filename))
            writer.append_data(image)

if __name__ == "__main__":
    dir = args.image_path
    gifname = args.gifname
    make_gif(dir, gifname)
