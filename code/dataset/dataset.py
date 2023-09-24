import os
import random

import cv2

import numpy as np

import torch
from torch.utils.data import DataLoader

import tqdm
import sys
sys.path.append('../')
import utils.general as utils
from utils.rend_utils import nerf_matrix_to_ngp, rand_poses, get_rays


class NeRFDataset:
    def __init__(self, data_dir, opts, type='train', H = None, W = None, downscale=1):
        super().__init__()
        self.opts = opts
        self.device = self.opts.device
        self.type = type
        self.downscale = downscale
        self.root_path = data_dir
        self.preload = self.opts.preload
        self.scale = self.opts.scale
        self.offset = self.opts.offset
        self.bound = self.opts.bound
        self.fp16 = self.opts.fp16

        self.training = self.type in ['train', 'trainval', 'all']
        self.data_available = not(self.root_path is None)

        self.num_rays = self.opts.num_rays if (self.training and self.data_available) else -1

        self.H = H
        self.W = W

        self.images = None
        self.poses = None
        self.latents = None
        if self.data_available:
            self.load_data()
        else:
            self.cx = self.W / 2
            self.cy = self.H / 2
            self.fov_range = self.opts.fov_range
            self.radius_range = self.opts.radius_range
            self.angle_overhead = self.opts.angle_overhead
            self.angle_front = self.opts.angle_front
            self.size = 100

    def load_data(self):
        transforms_path = os.path.join(self.root_path, 'transforms_train.json')

        transforms = utils.load_json(transforms_path)
        frames = transforms['frames']

        self.images = []
        self.poses = []

        for i, f in enumerate(frames):
            fname = os.path.join(self.root_path, f['file_path']) + '.png'
            if not os.path.exists(fname):
                raise FileNotFoundError(fname)
            image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            image = image.astype(np.float32) / 255.
            if self.H is None and self.W is None:
                self.H = image.shape[0]
                self.W = image.shape[1]
            pose = np.array(f['transform_matrix'], dtype=np.float32)
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
            self.images.append(image)
            self.poses.append(pose)
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0))
        self.images = torch.from_numpy(np.stack(self.images, axis=0))

        self.images = self.images.to(self.device, dtype=torch.half)
        self.size = len(self.images)
        self.poses = self.poses.to(self.device)

        fl_x = self.W / (2 * np.tan(transforms['camera_angle_x'] / 2)) if 'camera_angle_x' in transforms else None
        fl_y = self.H / (2 * np.tan(transforms['camera_angle_y'] / 2)) if 'camera_angle_y' in transforms else None
        if fl_x is None: fl_x = fl_y
        if fl_y is None: fl_y = fl_x
        cx = (transforms['cx']) if 'cx' in transforms else (self.W / 2)
        cy = (transforms['cy']) if 'cy' in transforms else (self.H / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def collate(self, index):
        B = len(index)
        C = 3
        inds = None
        dirs = None
        gt = None
        if self.data_available:
            poses = self.poses[index]
            intrinsics = self.intrinsics
        else:
            poses, dirs = rand_poses(B, self.device, self.radius_range, angle_overhead = self.angle_overhead,
                               angle_front = self.angle_front)
            fov = random.random() * (self.fov_range[1] - self.fov_range[0]) + self.fov_range[0]
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])


        rays = get_rays(poses, intrinsics, self.H, self.W, self.num_rays)

        if self.data_available:
           
            gt = self.images[index]
            if gt.shape[-1] == 4:
                gt = utils.rgba2rgb(gt, bkgd_map = None)

            if self.training:
                gt = torch.gather(gt.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))
                inds = rays['inds']

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'gt': gt,
            'inds': inds,
            'dirs': dirs
        }
        return data

    def dataloader(self):

        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader

    @torch.no_grad()
    def precompute_latents(self, sd_model, batch_size = 8):
        self.latents = []
        i = 0
        pbar = tqdm.tqdm(total=self.size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        pbar.set_description('Precomputing latent codes')
        while i < self.size:
            j = min(i + batch_size, self.size)

            images = self.images[i:j]
            images = utils.rgba2rgb(images, bkgd_map = None)
            images = images.permute(0, 3, 1, 2)
            latents = sd_model.encode_imgs(images)

            self.latents.append(latents)
            pbar.update(j - i)
            i = j

        self.latents = torch.cat(self.latents, dim=0)
        self.latents = self.latents.permute(0, 2, 3, 1).contiguous()
        bkgd = torch.ones([1, 3, self.H * 8, self.W * 8], dtype = self.latents.dtype, device=self.device)
        self.latent_bkgd = sd_model.encode_imgs(bkgd).reshape(1, 4, -1).permute(0, 2, 1)

        torch.cuda.empty_cache()

    def update_intrinsics(self, new_h, new_w):
        self.H = new_h
        self.W = new_w
        self.cx = self.W / 2
        self.cy = self.H / 2


