# noise.py
import torch
import itertools
import numpy as np
import functools
import copy
# import torch.multiprocessing as mp

def add_black_box(img_tensor, box_size, stride=1, num_processes=None):
    """
    Add black boxes to image tensor batch. 
    Applies box to all channels. 
    Returns N x batch images with the box covering every possible subsection.
    Also returns n dims
    """
    # Calculate number of image clones needed
    batch_size = img_tensor.size(0)
    img_dims = torch.Tensor(list(img_tensor.size()[2:]))
    ns = torch.ceil((img_dims - (torch.Tensor(box_size) - 1))/stride)
    n_total = int(torch.prod(ns))

    inds = [list(np.arange(n).astype(np.int32)*stride) for n in ns]
    pars = list(itertools.product(*inds))
    masking_fn = functools.partial(mask_img, img=img_tensor, box_size=box_size)

    imgs = torch.zeros([n_total*batch_size] +  list(img_tensor.size()[2:]))

    # multiprocess to save precious time
    # if num_processes is None:
        # num_processes = mp.cpu_count()
    # pool = mp.Pool(num_processes)
    # result = pool.imap(masking_fn, pars)
    result = map(masking_fn, pars)
    for i, arr in enumerate(result):
        imgs[i*batch_size:(i+1)*batch_size] = arr
    
    # reorganize 
    imgs = imgs.view(n_total, batch_size, imgs.size(1), imgs.size(1))
    imgs = imgs.permute(1,0,2,3).contiguous()
    imgs = imgs.view(batch_size*n_total, imgs.size(2), imgs.size(3))
    # print(imgs)
    # pool.close()
    # pool.join()
    return imgs.cuda(), ns.cuda()

def mask_img(offsets, img, box_size):
    img = copy.deepcopy(img)
    img[:, :, offsets[0]:offsets[0]+box_size[0], offsets[1]:offsets[1]+box_size[1]] = 0
    return img.squeeze(1)


if __name__ == '__main__':
    img_tensor = torch.arange(50).view((2,5,5))
    box_size = (3,3)
    img_tensor = torch.arange(64*64*32).view((32,64,64))
    box_size = (16,16)
    boxed_tensors = add_black_box(img_tensor, box_size, stride=1)