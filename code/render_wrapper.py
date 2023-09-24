import os

import torch

import imageio
import random
from tqdm import tqdm
import sys
sys.path.append('./models')

import numpy as np

from models.network_grid import NeRFNetwork
from dataset.dataset import NeRFDataset



import utils.rend_utils as rend_utils
import utils.general as utils
from utils import sketch_utils

from options_r import RenderOptions
import mcubes
import trimesh
import pyrallis

class Renderer:
    def __init__(self, opts):
        self.opts = opts
        self.model_opts = utils.load_pickle(os.path.join(self.opts.exp_dir, 'opts.pkl'))
        self.model = NeRFNetwork(self.model_opts).to(self.opts.device).eval()
        self.model.load_checkpoint(os.path.join(self.opts.exp_dir, 'checkpoints/latest.pth'))



        self.H = self.opts.H
        self.W = self.opts.W
        self.dataset = NeRFDataset(data_dir = None, opts = self.model_opts, type = 'val', H = self.H, W = self.W)

        self.fov_range = self.opts.fov_range

        #config background
        self.bg_color = torch.ones((1, 3, self.H, self.W)).to(self.opts.device)


        self.bg_color = self.bg_color.reshape(3, self.H * self.W).permute(1, 0)

        #config dirs
        self.out_dir = os.path.join(self.opts.exp_dir, 'renders')
        os.makedirs(self.out_dir, exist_ok = True)

    def turn_off_bitfield(self, sketch_dir):
        base_density_grid = torch.load('../exps/dreamfusion/horse_highres_a potted green plant/2023_02_13_14_45_52/checkpoints/ckpt_0100.pth')['model']['density_grid']
        sketches = sketch_utils.Sketches(sketch_dir, H = 128, W = 128, device = self.opts.device,  type = 'manual', fill_sketch = False,  preprocess_sketch = True)
        canvases, poses, intrinsics, bboxes, _, _ = sketches.get_sketches()
        self.model.turn_off_bitfield_outside_bbox(bboxes, poses, intrinsics, base_density_grid)
        a = 0

    @torch.no_grad()
    def get_image(self, poses = None, intrinsics = None, return_pil = False, save_path = None, output_type = 'rgb', num_poses = 1, disable_bg = True):
        if poses is None:
            poses, _ = rend_utils.rand_poses(num_poses, radius_range=(2.5, 3.5), device=self.opts.device)
            intrinsics = self.get_intrinsics(random.uniform(*self.fov_range))


        rays = rend_utils.get_rays(poses, intrinsics, self.H, self.W, -1)
        bg_color = self.bg_color.repeat(len(poses), 1)
        if output_type == 'normal':
            outputs = self.model.render(rays['rays_o'], rays['rays_d'], shading = 'normal', staged=True, bg_color=bg_color, perturb=False,
                              force_all_rays=True, disable_bg=disable_bg)
            image = outputs['image']
            if disable_bg:
                mask = outputs['weights_sum']
                mask[mask < 0.5] = 0
                image = torch.cat([image, mask[..., None]], dim=-1)
            c = 4 if disable_bg else 3
        outputs = self.model.render(rays['rays_o'], rays['rays_d'], staged = True, bg_color = bg_color, perturb = False, force_all_rays = True, disable_bg = disable_bg)
        if output_type == 'rgb':
            image = outputs['image']
            if disable_bg:
                mask = outputs['weights_sum']
                mask[mask < 0.5] = 0
                image = torch.cat([image, mask[..., None]], dim = -1)

            c = 4 if disable_bg else 3
        elif output_type == 'depth':
            image = outputs['depth'][..., None]
            c = 1
        elif output_type == 'silhouette':
            image = outputs['weights_sum'][..., None]
            c = 1

        if image.ndim == 3:
            image = image.permute(0, 2, 1).reshape(-1, c, self.H, self.W)
        if return_pil:
            if save_path is None:
                save_path = 'test.png'
            image = utils.save_tensor_image(image, os.path.join(self.out_dir, save_path))

        return image

    def render_circle(self, radius = 4, num_poses = 50, save_path = 'circle.gif', batch_size = 8):
        poses = self.get_circular_poses(radius, num_poses)
        intrinsics = self.get_intrinsics((self.opts.fov_range[0] + self.opts.fov_range[1])/ 2)
        res = []

        pbar = tqdm(total = num_poses, desc = 'Views rendered')
        count = 0
        while count < num_poses:
            en = min(count + batch_size, num_poses)
            with torch.cuda.amp.autocast(enabled = self.model_opts.fp16):
                image = self.get_image(poses[count:en], intrinsics, disable_bg = False)
            image = utils.save_tensor_image(image)
            res.append(image)
            count = en
            pbar.update(batch_size)


        if batch_size > 1:
            res = sum(res, [])
        imageio.mimsave(os.path.join(self.out_dir, save_path), res, fps = 48)
        # res[0].save(
        #     os.path.join(self.out_dir, save_path),
        #     format='GIF',
        #     save_all=True,
        #     loop=0,
        #     append_images = res,
        #     duration = 20,
        #     disposal = 2
        # )

    def get_circular_poses(self, radius, num_poses):
        thetas = torch.tensor([np.pi / 2 -0.4]).repeat(num_poses).to(self.opts.device)
        phis = torch.linspace(0, 2 * np.pi, num_poses).to(self.opts.device)
        poses = rend_utils.get_poses(radius, thetas, phis, device = self.opts.device)
        return poses
    def get_pose_from_angles(self, radius, thetas, phis):
        #radius : float
        #thetas : list of floats (in radians)
        #phis : list of floats (in radians)

        num_poses = np.max([len(thetas), len(phis)])

        thetas = torch.tensor(thetas).to(self.opts.device)
        phis = torch.tensor(phis).to(self.opts.device)

        if len(thetas) == 1:
            thetas = thetas.repeat(num_poses)
        elif len(phis) == 1:
            phis = phis.repeat(num_poses)
        poses = rend_utils.get_poses(radius,thetas, phis, device = self.opts.device)
        return poses
    def get_intrinsics(self, fov):
        cx = self.W / 2
        cy = self.H / 2
        fx = fy = 0.5 * self.W / np.tan(0.5 * np.deg2rad(fov))
        intrinsics = np.array([fx, fy, cx, cy])
        return intrinsics


    def load_checkpoint(self, ckpt_path):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location=self.opts.device)
        if 'model' not in ckpt:
            self.model.load_state_dict(ckpt)
            return
        #ckpt['model']['density_bitfield'] = torch.ones_like(ckpt['model']['density_bitfield']) * 255
        self.model.load_state_dict(ckpt['model'], strict=False)
        if self.opts.model_opts.cuda_ray:
            if 'mean_count' in ckpt:
                self.model.mean_count = ckpt['mean_count']
            if 'mean_density' in ckpt:
                self.model.mean_density = ckpt['mean_density']

    @torch.no_grad()
    def extract_mesh(self, resolution, save_mesh = True):
        grid = rend_utils.grid_coord(resolution, normalize=True)
        grid_split = grid.split(10000, dim = 0)

        density_field = []
        for split in tqdm(grid_split):
            d = self.model.density(split)['sigma']
            density_field.append(d)
        density_field = torch.cat(density_field, dim = 0).reshape(resolution, resolution, resolution).cpu().numpy()
        vertices, triangles = mcubes.marching_cubes(density_field, self.model.density_thresh * (3 if self.opts.model_opts.activation == 'softplus' else 1))

        bound_max = np.array([1., 1., 1.])
        bound_min = np.array([-1., -1., -1.])

        vertices = (vertices / (resolution - 1)) * (bound_max - bound_min)[None, :] + bound_min[None, :]
        if save_mesh:
            trimesh.Trimesh(vertices, triangles).export(os.path.join(self.out_dir, 'mesh.obj'))
        else:
            return vertices, triangles


from PIL import Image

if __name__ == '__main__':
    opts = pyrallis.parse(config_class=RenderOptions)
    renderer = Renderer(opts)

    renderer.render_circle(1.5, 128, 'output.mp4', batch_size = 32)
    #renderer.extract_mesh(256)