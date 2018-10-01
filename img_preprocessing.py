# img_preprocessing.py
import os
import torch
import numpy as np 
from PIL import Image


def save_imgs(img_dir='./data/imagenet/final2000/', 
	save_path='./data/imagenet/img_dict.pth'):

	imgs_by_id = {}

	for i, img_path in enumerate(os.listdir(img_dir)):
		img = Image.open(img_dir+img_path)
		img = img.resize((64,64), resample=Image.LANCZOS)
		img = torch.Tensor(np.array(img))
		# print(img.size())
		img_id = img_path.split('.')[0]
		imgs_by_id[img_id] = img
		# break

	torch.save(imgs_by_id, save_path)


if __name__ == '__main__':
	a = torch.load('./data/imagenet/img_dict.pth')
	print(len(a)) 