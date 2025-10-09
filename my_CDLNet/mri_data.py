from os import path, listdir
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import h5py
import random
from PIL import Image

class MRIDataset(data.Dataset):
	def __init__(self, root_dirs, transform, load_color=False):
		self.image_paths = []
		self.image_list = []

		for cur_path in root_dirs:
			self.image_paths += [path.join(cur_path, file) \
				for file in listdir(cur_path) \
                # Specifically for fastMRI, load only the t2 weighted brain images
				if file.endswith(('tif','tiff','png','jpg','jpeg','bmp', 'h5')) and file.startswith("file_brain_AXT2")]

		print(f"Loading {root_dirs}:")
		for i in tqdm(range(len(self.image_paths))):
			self.image_list.append(self.image_paths[i])

		self.root_dirs = root_dirs
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
        # Convert item to image domain
		hf = h5py.File(self.image_list[idx])
        # Grab random slice within kspace volume
		volume_kspace = hf['kspace']
		slices = volume_kspace.shape[0]
		kspace = volume_kspace[random.randint(0, slices-1), :, :, :]
		# Transform multicoil into image domain
		image_multicoil = np.fft.ifft2(kspace)
		# RSS transformation
		image = np.linalg.norm(image_multicoil, ord = 2, axis = 0)
        normalized_image = (image - np.min(image))/(np.max(image)-np.min(image))
        return self.transform(Image.fromarray(normalized_image))

def get_data_loader(dir_list, batch_size=1, load_color=False, crop_size=None, test=True):
    # Don't perform random transformations if in test phase
	if test:
		xfm = transforms.ToTensor()
	else:
		xfm = transforms.Compose([transforms.RandomCrop(crop_size),
		                          transforms.RandomHorizontalFlip(),
		                          transforms.RandomVerticalFlip(),
		                          transforms.ToTensor()])

	return data.DataLoader(MRIDataset(dir_list, xfm, load_color),
	                       batch_size = batch_size,
	                       drop_last  = (not test),
	                       shuffle    = (not test))

def get_fit_loaders(trn_path_list =['CBSD432'],
	              val_path_list=['Kodak'],
	              tst_path_list=['CBSD68'],
	              crop_size  = 128,
	              batch_size = [10,1,1],
	              load_color = False):

	if type(batch_size) is int:
		batch_size = [batch_size, 1, 1]
    # return 3 different dataloader objects for each phase
	dataloaders = {'train': get_data_loader(trn_path_list, 
                                          batch_size[0], 
                                          load_color, 
                                          crop_size=crop_size, 
                                          test=False),
	               'val':   get_data_loader(val_path_list, 
                                          batch_size[1], 
                                          load_color, 
                                          test=True),
	               'test':  get_data_loader(tst_path_list, 
                                          batch_size[2], 
                                          load_color, 
                                          test=True)}
	return dataloaders
