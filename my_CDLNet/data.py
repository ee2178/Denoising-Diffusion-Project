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
	def __init__(self, root_dirs, transform, load_color=False, start_slice = 0, end_slice = 8):
		self.image_paths = []
		self.image_list = []
		self.start_slice = start_slice
		self.end_slice = end_slice

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
				# # Get number of slices in the image volume
				# num_slices = image_volume.shape[0]
				# # Pick a random number between 2 and num_slices - 2
				# idx = (num_slices-4) * torch.rand(1) + num_slices/2
				# idx = round(idx[0].numpy())
				# image = image_volume[idx, :, :, :]
				self.image_list.append(image_volume)

		self.root_dirs = root_dirs
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		# Get a random slice from your volume, starting at start_slice, ending at end_slice
		slice = np.random.randint(self.start_slice, self.end_slice)
		image = self.image_list[idx][slice, :, :][np.newaxis, :, :]
        # Convert image to tensor
		image = torch.from_numpy(image)
		# Image is a complex tensor, apply transformations to real and imaginary parts
		image_two_channel = torch.cat((torch.real(image), torch.imag(image)), dim = 0)
        # We will assume input to already be a tensor:
		if self.transform:
			image_transform = self.transform(image_two_channel)
			image_out = torch.complex(image_transform[0, :, :], image_transform[1, :, :])
		else:
			image_out = image[0, :, :]
		# Scale the image nicely
		image_out = image_out / torch.norm(image_out)**2
		return image_out

def get_data_loader(dir_list, batch_size=1, load_color=False, crop_size=None, test=True, start_slice = 0, end_slice = 8):
    # Don't perform random transformations if in test phase
	if test:
		xfm = None
	else:
		xfm = transforms.Compose([transforms.RandomCrop(crop_size),
		                          transforms.RandomHorizontalFlip(),
		                          transforms.RandomVerticalFlip(),
		                          ])

	return data.DataLoader(MRIDataset(dir_list, xfm, load_color, start_slice = start_slice, end_slice = end_slice),
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
                  end_slice = 8):

	if type(batch_size) is int:
		batch_size = [batch_size, 1, 1]
    # return 3 different dataloader objects for each phase
	dataloaders = {'train': get_data_loader(trn_path_list, 
                                          batch_size[0], 
                                          load_color, 
                                          crop_size=crop_size, 
                                          test=False, 
                                          start_slice = start_slice, 
                                          end_slice = end_slice),
	               'val':   get_data_loader(val_path_list, 
                                          batch_size[1], 
                                          load_color, 
                                          test=True, 
                                          start_slice = start_slice, 
                                          end_slice = end_slice),
	               'test':  get_data_loader(tst_path_list, 
                                          batch_size[2], 
                                          load_color, 
                                          test=True, 
                                          start_slice = start_slice, 
                                          end_slice = end_slice)}
	return dataloaders
