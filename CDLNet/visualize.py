#!/usr/bin/env python3
import os, sys, json, copy, time
from pprint import pprint
from tqdm import tqdm
import numpy as np
from numpy.fft import fftshift, fft2
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import awgn

import model
import nle
import utils, data, train

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("args_fn", type=str, help="Path to args.json file.", default="args.json")
parser.add_argument("--test", type=str, help="Run model over specified test set (provided path to image dir).", default=None)
parser.add_argument("--noise_level", type=int, nargs='*', help="Input noise-level(s) on [0,255] range. Single value required for --passthrough. If --test is used, multiple values can be specified.", default=[-1])
ARGS = parser.parse_args()

def main(model_args):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    net, _, _, epoch0 = train.init_model(model_args, device=device)
    net.eval()
    
    # Load image in from test dataset
    loader = data.get_data_loader([ARGS.test], test=True)
    iterator = iter(loader)
    test_dir = os.listdir(ARGS.test)
    for i in range(len([name for name in test_dir if os.path.isfile(os.path.join(ARGS.test, name))])):
        test_img = next(iterator)
        fig, ax = plt.subplots(1, 3)
        plt.suptitle("CDLNet Results")
        ax[0].imshow(np.squeeze(test_img), cmap='gray')
        ax[0].set_title('Test Image')
        # plt.savefig('test.png')
        test_img = test_img.to(device)
        
        noisy_test_img = awgn(test_img, ARGS.noise_level)
        ax[1].imshow(np.squeeze(noisy_test_img[0].cpu()), cmap = 'gray')
        ax[1].set_title('Noisy Image')
        # plt.savefig('noisy_test.png')
        
        denoised_test = net(test_img)
        ax[2].imshow(np.squeeze(denoised_test[0].detach().cpu().numpy()), cmap= 'gray')
        ax[2].set_title('Denoised Image')
        plt.tight_layout()
        fig.savefig(os.path.join('results/', test_dir[i]))
    
    
    
if __name__ == "__main__":
    """ Load arguments from json file and command line and pass to main.
    """
    # load provided args.json file
    model_args_file = open(ARGS.args_fn)
    model_args = json.load(model_args_file)
    pprint(model_args)
    model_args_file.close()
    main(model_args)