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

def is_pd_scan(fname):
    with h5py.File(fname, "r") as f:
        acquisition = f.attrs["acquisition"]
    return acquisition == "CORPD_FBK"

class MRIKSpaceDataset(data.Dataset):
    def __init__(self, root_dirs, kspace_dirs, transform, scaling_fac = 2e3, start_slice=0, end_slice=10, modality = 'brain'):
        self.image_paths = []
        self.kspace_paths = []

        for cur_path in root_dirs:
            self.image_paths += [path.join(cur_path, file) \
                for file in listdir(cur_path) \
                # For knee, this will automatically search in the right directory since I did filtering ahead of time
                if file.endswith(('tif','tiff','png','jpg','jpeg','bmp','.h5'))]

        print(f"Loading {root_dirs}:")
        
        if modality == 'brain':
            for cur_path in kspace_dirs:
                self.kspace_paths += [path.join(cur_path, file) \
                    for file in listdir(cur_path) \
                    if file.endswith(('tif','tiff','png','jpg','jpeg','bmp','.h5')) and file.startswith('file_brain_AXT2')]
        elif modality == 'knee':
            for cur_path in kspace_dirs:
                self.kspace_paths += [path.join(cur_path, file) \
                    for file in listdir(cur_path) \
                    # Filter by looking for PD scans only
                    if file.endswith(('tif','tiff','png','jpg','jpeg','bmp','.h5')) and is_pd_scan(path.join(cur_path, file))]

        print(f"Loading {kspace_dirs}")

        self.root_dirs = root_dirs
        self.kspace_dirs = kspace_dirs
        self.transform = transform
        self.scaling_fac = scaling_fac
        self.start_slice = start_slice
        self.end_slice = end_slice
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get an entire volume, starting at start_slice, ending at end_slice
        slice = np.random.randint(self.start_slice, self.end_slice)
        with h5py.File(self.image_paths[idx]) as f:
            image = f['image'][slice][np.newaxis, :, :]
            smaps = f['smaps'][slice]
        # Find corresponding kspace
        with h5py.File(self.kspace_paths[idx]) as f:
            kspace = f['kspace'][slice]

        # Send to GPU
        image = torch.from_numpy(image)*self.scaling_fac
        smaps = torch.from_numpy(smaps)
        kspace = torch.from_numpy(kspace)*self.scaling_fac

        return kspace, smaps, image

def get_data_loader(dir_list, kspace_dir_list, batch_size=1, load_color=False, crop_size=None, test=True, start_slice = 0, end_slice = 8,scaling_fac = 1e6, num_workers = 1, modality = 'brain'):
    # Don't perform random transformations if in test phase
    if test:
        xfm = None
    else:
        # xfm = transforms.Compose([transforms.RandomCrop(crop_size),
        #                          transforms.RandomHorizontalFlip(),
        #                          transforms.RandomVerticalFlip(),
        #                          ])
        xfm = None
    return data.DataLoader(MRIKSpaceDataset(dir_list, kspace_dir_list, xfm, scaling_fac, modality = modality, start_slice = start_slice, end_slice = end_slice),
                           batch_size = batch_size,
                           drop_last  = (not test),
                           shuffle    = (not test))

def get_fit_loaders(trn_path_list =['CBSD432'],
                  val_path_list=['Kodak'],
                  tst_path_list=['CBSD68'],
                  trn_kspace_path_list=None,
                  val_kspace_path_list=None,
                  tst_kspace_path_list=None,
                  crop_size  = 128,
                  batch_size = [1,1,1],
                  load_color = False, 
                  start_slice = 0,
                  end_slice = 8,
                  scaling_fac = 1e6,
                  num_workers = 1,
                  modality = 'brain'):

    if type(batch_size) is int:
        batch_size = [batch_size, 1, 1]
    # return 3 different dataloader objects for each phase
    dataloaders = {'train': get_data_loader(trn_path_list, 
                                            trn_kspace_path_list,
                                            batch_size[0], 
                                            load_color, 
                                            crop_size=crop_size, 
                                            test=False, 
                                            start_slice = start_slice,
                                            end_slice = end_slice,
                                            scaling_fac = scaling_fac,
                                            num_workers = num_workers,
                                            modality = modality),
                   'val':   get_data_loader(val_path_list,
                                            val_kspace_path_list,
                                            batch_size[1], 
                                            load_color, 
                                            test=True,
                                            start_slice = start_slice,
                                            end_slice = end_slice,
                                            scaling_fac = scaling_fac,
                                            num_workers = num_workers,
                                            modality = modality),
                   'test':  get_data_loader(tst_path_list, 
                                            tst_kspace_path_list,
                                            batch_size[2], 
                                            load_color, 
                                            test=True,
                                            start_slice = start_slice,
                                            end_slice = end_slice,
                                            scaling_fac = scaling_fac,
                                            num_workers = num_workers,
                                            modality = modality)}
    return dataloaders
