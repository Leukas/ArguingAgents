# noise.py
import torch
import itertools
import numpy as np
import functools
import copy

# from . import device

def add_black_box_random(img_tensor, box_size):
    """
    Add black boxes to a batch in random places.
    Non-square boxes are a WIP
    """
    # img_tensor = copy.deepcopy(img_tensor)
    batch_size = img_tensor.size(0)
    img_dims = torch.Tensor(list(img_tensor.size()[2:]))
    ns = torch.ceil(img_dims - (torch.Tensor(box_size) - 1))
    rand_xs = torch.randint(int(ns[0]), (batch_size, 1))
    rand_ys = torch.randint(int(ns[1]), (batch_size, 1))

    # define all indices of where a box will cover
    sq_inds = torch.Tensor([[x]*box_size[0] for x in range(box_size[0])])

    # add random offsets
    xs = torch.flatten(sq_inds).unsqueeze(0).repeat(batch_size, 1) + rand_xs
    ys = torch.flatten(sq_inds.t()).unsqueeze(0).repeat(batch_size, 1) + rand_ys

    # define all batch indices
    bs = torch.Tensor([[x]*int(box_size[0] * box_size[1]) for x in range(batch_size)])

    # print(xs.size())
    # print(ys.size())
    # print(bs.size())

    # print(img_tensor.size())
    # make and apply mask
    mask = torch.ones(img_tensor.size()).float()
    mask[bs.long(), :, xs.long(), ys.long()] = 0
    img_tensor = img_tensor * mask.type_as(img_tensor)

    return img_tensor

def black_box_module(img_tensor, box_size, stride=1):
    """
    Black box module
    """
    # img_tensor = copy.deepcopy(img_tensor)
    batch_size = img_tensor.size(0)
    img_dims = torch.Tensor(list(img_tensor.size()[2:]))
    ns = torch.ceil((img_dims - (torch.Tensor(box_size) - 1))/stride)
    n_total = int(torch.prod(ns))

    # define all indices of where a box will cover
    sq_inds = torch.Tensor([[x]*box_size[0] for x in range(box_size[0])])
    # sq_inds = box_width x box_height

    num_channels = img_tensor.size(1)
    num_box_inds = sq_inds.numel()

    # define all offset indices possible
    off_inds = [list(np.arange(n).astype(np.int32)*stride) for n in ns]

    # print(off_inds)
    # print(n_total)

    off_inds = torch.Tensor(off_inds[0]).unsqueeze(1)
    num_off_inds = off_inds.numel()**2
    x_inds = off_inds.repeat(1,off_inds.size(0)).flatten()
    x_inds = x_inds.unsqueeze(0).unsqueeze(-1).repeat(batch_size, num_channels, num_box_inds)
    y_inds = off_inds.repeat(off_inds.size(0),1).flatten()
    y_inds = y_inds.unsqueeze(0).unsqueeze(-1).repeat(batch_size, num_channels, num_box_inds)
    
    # y_inds = torch.Tensor(inds[1])
    # print(x_inds.size())
    # print(y_inds)
    # print(rand_xs)

    xs = torch.flatten(sq_inds).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_off_inds*num_channels, 1) #+ x_inds#+ rand_xs
    ys = torch.flatten(sq_inds.t()).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_off_inds*num_channels, 1) #+ y_inds# + rand_ys

    # print(xs.size())
    xs += x_inds
    ys += y_inds

    # define all batch indices
    bs = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).repeat(1, num_off_inds*num_channels, num_box_inds)
    cs = torch.arange(num_off_inds*num_channels).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, num_box_inds)

    # print(cs)

    # print(xs.size())
    # print(ys.size())
    # print(bs.size())
    # print(cs.size())
    # print(img_tensor.size())

    # make and apply mask
    mask = torch.ones(img_tensor.size()).float()
    mask = mask.repeat(1, num_off_inds, 1, 1)

    # print(mask.size())
    mask[bs.long(), cs.long(), xs.long(), ys.long()] = 0
    # print(mask)
    img_tensor = img_tensor.unsqueeze(2).repeat(1, 1, num_off_inds, 1, 1).contiguous()
    img_tensor = img_tensor.view(batch_size, num_off_inds*num_channels, img_tensor.size(3), img_tensor.size(4))
    img_tensor = img_tensor * mask.type_as(img_tensor)
    img_tensor = img_tensor.view(-1, num_channels, img_tensor.size(2), img_tensor.size(3))

    # print(img_tensor)
    return img_tensor, ns



def add_black_boxes(img_tensor, box_size, stride=1, num_processes=None):
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
    pars = torch.Tensor(list(itertools.product(*inds)))
    
    # define all indices of where a box will cover
    sq_inds = torch.Tensor([[x]*box_size[0] for x in range(box_size[0])])

    print(sq_inds)
    # xs = pars[:, 0]
    # xs = 
    # print(xs)
    print(pars)
    # print(len(pars))
    # print(n_total)
    # print(torch.Tensor(pars))
    # return
    masking_fn = functools.partial(mask_img, img=img_tensor, box_size=box_size)

    imgs = torch.zeros([n_total*batch_size] +  list(img_tensor.size()[1:]))

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



# def add_black_box(img_tensor, box_size, stride=1, num_processes=None):
#     """
#     Add black boxes to image tensor batch. 
#     Applies box to all channels. 
#     Returns N x batch images with the box covering every possible subsection.
#     Also returns n dims
#     """
#     # Calculate number of image clones needed
#     batch_size = img_tensor.size(0)
#     img_dims = torch.Tensor(list(img_tensor.size()[2:]))
#     ns = torch.ceil((img_dims - (torch.Tensor(box_size) - 1))/stride)
#     n_total = int(torch.prod(ns))

#     inds = [list(np.arange(n).astype(np.int32)*stride) for n in ns]
#     pars = list(itertools.product(*inds))
#     masking_fn = functools.partial(mask_img, img=img_tensor, box_size=box_size)

#     imgs = torch.zeros([n_total*batch_size] +  list(img_tensor.size()[2:]))

#     # multiprocess to save precious time
#     # if num_processes is None:
#         # num_processes = mp.cpu_count()
#     # pool = mp.Pool(num_processes)
#     # result = pool.imap(masking_fn, pars)
#     result = map(masking_fn, pars)
#     for i, arr in enumerate(result):
#         imgs[i*batch_size:(i+1)*batch_size] = arr
    
#     # reorganize 
#     imgs = imgs.view(n_total, batch_size, imgs.size(1), imgs.size(1))
#     imgs = imgs.permute(1,0,2,3).contiguous()
#     imgs = imgs.view(batch_size*n_total, imgs.size(2), imgs.size(3))
#     # print(imgs)
#     # pool.close()
#     # pool.join()
#     return imgs.cuda(), ns.cuda()

def mask_img(offsets, img, box_size):
    print(offsets[0])
    img = copy.deepcopy(img.data)
    img[:, :, offsets[0]:offsets[0]+box_size[0], offsets[1]:offsets[1]+box_size[1]] = 0
    return img


if __name__ == '__main__':
    img_tensor = torch.arange(32).view((2,1,4,4))
    # img_tensor = torch.arange(36).view((2,2,3,3))
    # print(img_tensor)
    box_size = (2,2)
    # img_tensor = torch.arange(32*32*32).view((32,1,32,32))
    # box_size = (10,10)
    # bx = black_box_module(img_tensor, box_size, stride=1)

    # print(bx)
    # boxed_tensors = add_black_boxes(img_tensor, box_size, stride=1)
    boxed_tensor = add_black_box_random(img_tensor, box_size)
    print(img_tensor)
    print(boxed_tensor)
    # print(boxed_tensors)