from os import path, listdir
from glob import glob
from PIL import Image
import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

class MRIDataset(data.Dataset):
    def __init__(self, root_dirs, transform, load_color=False, start_slice = 0, end_slice = 8, scaling_fac = 1e6, get_smaps = False, num_workers = 1):
        self.image_paths = []
        self.image_list = []
        self.start_slice = start_slice
        self.end_slice = end_slice

        for cur_path in root_dirs:
            self.image_paths += [path.join(cur_path, file) \
                for file in listdir(cur_path) \
                if file.endswith(('tif','tiff','png','jpg','jpeg','bmp','.h5'))]

        print(f"Loading {root_dirs}:")
        self.root_dirs = root_dirs
        self.transform = transform
        self.scaling_fac = scaling_fac
        self.get_smaps = get_smaps
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if not self.get_smaps:
            # Get a random slice from your volume, starting at start_slice, ending at end_slice
            slice = np.random.randint(self.start_slice, self.end_slice)
            with h5py.File(self.image_paths[idx]) as f:
                image = f['image'][slice, :, :][np.newaxis, :, :]
                # We can return a mask, but this is only meaningful if we have espirit maps. 
                # Actually forget the mask, we would have to mess around  
            # Convert image to tensor
            image = torch.from_numpy(image)
            # Image is a complex tensor, apply transformations to real and imaginary parts
            image_two_channel = torch.cat((torch.real(image), torch.imag(image)), dim = 0)
            # We will assume input to already be a tensor:
            if self.transform:
                image_transform = self.transform(image_two_channel)
                image_out = torch.complex(image_transform[0, :, :], image_transform[1, :, :])*self.scaling_fac
            else:
                image_out = image[0, :, :] * self.scaling_fac
            return image_out
        else:
            # Return smaps at some fixed slice
            slice = np.random.randint(self.start_slice, self.end_slice)
            with h5py.File(self.image_paths[idx]) as f:
                smaps = f['smaps'][slice, :, :, :]
                image = f['image'][slice, :, :][np.newaxis, :, :]
            # Convert image to tensor
            image = torch.from_numpy(image)
            # Convert volume to tensor
            smaps = torch.from_numpy(smaps) # C x H x W
            # Don't bother with transformations on volumes
            return image, smaps, slice, self.image_paths[idx]

def get_data_loader(dir_list, batch_size=1, load_color=False, crop_size=None, test=True, start_slice = 0, end_slice = 8, scaling_fac = 1e6, get_smaps = False, num_workers = 1, brain = True, center_crop = 320):
    # Don't perform random transformations if in test phase
    if test:
        xfm = None
    else:
        if brain:
            xfm = transforms.Compose([transforms.CenterCrop(center_crop),
                                    transforms.RandomCrop(crop_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    ])
        else:
            xfm = transforms.Compose([transforms.RandomCrop(crop_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    ])
        # xfm = None

    return data.DataLoader(MRIDataset(dir_list, xfm, load_color, start_slice = start_slice, end_slice = end_slice, scaling_fac = scaling_fac, get_smaps = get_smaps, num_workers = num_workers),
                           batch_size = batch_size,
                           drop_last  = (not test),
                           shuffle    = (not test))

def get_fit_loaders(trn_path_list =['CBSD432'],
                  val_path_list=['Kodak'],
                  tst_path_list=['CBSD68'],
                  crop_size  = 128,
                  batch_size = [10,1,1],
                  load_color = False, 
                  start_slice = 0, 
                  end_slice = 8,
                  scaling_fac = 1e6, 
                  num_workers = 1,
                  brain=True,
                  center_crop = 320):

    if type(batch_size) is int:
        batch_size = [batch_size, 1, 1]
    # return 3 different dataloader objects for each phase
    dataloaders = {'train': get_data_loader(trn_path_list, 
                                          batch_size[0], 
                                          load_color, 
                                          crop_size=crop_size, 
                                          test=False, 
                                          start_slice = start_slice, 
                                          end_slice = end_slice,
                                          scaling_fac = scaling_fac,
                                          num_workers = num_workers,
                                          brain = brain,
                                          center_crop = center_crop),
                   'val':   get_data_loader(val_path_list, 
                                          batch_size[1], 
                                          load_color, 
                                          test=True, 
                                          start_slice = start_slice, 
                                          end_slice = end_slice,
                                          scaling_fac = scaling_fac,
                                          num_workers = num_workers,
                                          brain = brain,
                                          center_crop = center_crop),
                   'test':  get_data_loader(tst_path_list, 
                                          batch_size[2], 
                                          load_color, 
                                          test=True, 
                                          start_slice = start_slice, 
                                          end_slice = end_slice,
                                          scaling_fac = scaling_fac,
                                          num_workers = num_workers,
                                          brain = brain,
                                          center_crop = center_crop)}
    return dataloaders
