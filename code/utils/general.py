import os
import json
import numpy as np
import torch
import random
from PIL import Image
import pickle

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def safe_normalize(x, eps = 1e-8):
    return x / (x.norm() + eps)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_psnr(img1, img2, normalize_rgb = False):
    # img1, img2: [0,1] or [-1,1]
    #normalize_rgb: [-1,1] --> [0,1]
    if normalize_rgb:
        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1.) / 2.
    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).cuda())
    return psnr

def output2image(image, type):
    #image: (1, H, W, 3) if rgb (1, H, W) if depth
    #Output: (H, W, 3) numpy images in [0, 255]
    if type == 'depth':
        image = image.unsqueeze(-1).repeat(1, 1, 1, 3)
    image = image.clamp(0, 1)
    image = image.squeeze(0).detach().cpu().numpy()
    image = (image * 255.).astype(np.uint8)
    return image

def save_tensor_image(image, save_path = None):
    #image: (B, C, H, W) tensor [0, 1]
    image = image.clamp(0, 1)
    image = image.detach().cpu().numpy()
    image = (image * 255.).astype(np.uint8)
    image = np.transpose(image, (0, 2, 3, 1))
    if image.shape[-1] == 1:
        image = image[..., 0]
    if image.shape[0] == 1:
        image = Image.fromarray(image[0])
        if save_path is not None:
            image.save(save_path)

    else:
        res = []
        for i in range(image.shape[0]):
            current = Image.fromarray(image[i])
            res.append(current)
            if save_path is not None:
                current.save(os.path.splitext(save_path)[0] + '_' + str(i) + '.png')
        image = res
    return image

def save_nerf_evals(pred_rgb, gt_rgb = None, pred_depth = None, pred_normal = None, save_path = None):

    pred_rgb = output2image(pred_rgb, 'rgb')
    res = pred_rgb
    if gt_rgb is not None:
        gt_rgb = output2image(gt_rgb, 'rgb')
        res  = np.concatenate([res, gt_rgb], axis = 1)
    if pred_depth is not None:
        pred_depth = output2image(pred_depth, 'depth')
        res = np.concatenate([res, pred_depth], axis=1)
    if pred_normal is not None:
        pred_normal = output2image(pred_normal, 'rgb')
        res = np.concatenate([res, pred_normal], axis=1)
    res = Image.fromarray(res)
    if save_path is not None:
        res.save(save_path)
    return res

def rgba2rgb(image, bkgd_map = None, input_type = 'torch'):
    #image: (B, H, W, 4) or (H, W, 4) torch tensor or numpy array in [0,1]
    #bkgd_map: (B, H, W, 3) or (H, W, 3) torch tensor or numpy array in [0, 1] or None
    rgb_map = image[..., :3]
    alpha_map = image[..., 3][..., None]
    if bkgd_map is None:
        if input_type == 'torch':
            bkgd_map = torch.ones_like(rgb_map).to(rgb_map.device)
        else:
            bkgd_map = np.ones_like(rgb_map)
    rgb_map = rgb_map * alpha_map + bkgd_map * (1 - alpha_map)
    return rgb_map

def latent2rgb(latent_image):
    #latent_image:(B, 4, 64, 64) tensor
    #Output: (B, 3, 64, 64) tensor: rgb approximation
    latent_matrix = torch.tensor([
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]).to(latent_image.device)
    image = (latent_image.permute(0, 2, 3, 1) @ latent_matrix).permute(0, 3, 1, 2)
    return image

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def ce_pq_loss(p, q, weight=None):
    def clamp(v, T=0.01):
        return v.clamp(T, 1 - T)

    ce = -1 * (p * torch.log(clamp(q)) + (1 - p) * torch.log(clamp(1 - q)))
    if weight is not None:
        ce *= weight
    return ce.sum()

