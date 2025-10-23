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

class MyDataset(data.Dataset):
	def __init__(self, root_dirs, transform, load_color=False):
		self.image_paths = []
		self.image_list = []

		for cur_path in root_dirs:
			self.image_paths += [path.join(cur_path, file) \
				for file in listdir(cur_path) \
				if file.endswith(('tif','tiff','png','jpg','jpeg','bmp','.h5'))]

		print(f"Loading {root_dirs}:")
		for i in tqdm(range(len(self.image_paths))):
			if load_color:
				self.image_list.append(Image.open(self.image_paths[i]))
			else:
				image_volume = h5py.File(self.image_paths[i])['image']
				# Get number of slices in the image volume
				num_slices = image_volume.shape[0]
				# Pick a random number between 2 and num_slices - 2
				idx = (num_slices-4) * torch.rand(1) + num_slices/2
				idx = round(idx[0].numpy())
				image = image_volume[idx, :, :, :]
				self.image_list.append(image)

		self.root_dirs = root_dirs
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		# Image is a complex tensor, apply transformations to real and imaginary parts
		image_real = torch.real(self.image_list[idx])
		image_imag = torch.imag(self.image_list[idx])
		image_real, image_imag = self.transform([image_real, image_imag])
		return self.complex(image_real, image_imag)

def get_data_loader(dir_list, batch_size=1, load_color=False, crop_size=None, test=True):
    # Don't perform random transformations if in test phase
	if test:
		xfm = transforms.ToTensor()
	else:
		xfm = transforms.Compose([transforms.RandomCrop(crop_size),
		                          transforms.RandomHorizontalFlip(),
		                          transforms.RandomVerticalFlip(),
		                          transforms.ToTensor()])

	return data.DataLoader(MyDataset(dir_list, xfm, load_color),
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