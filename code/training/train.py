import torch
import torch.nn as nn
import trimesh



import tqdm
from datetime import datetime
import itertools
import os
import sys
sys.path.append('../code')
sys.path.append('../code/models')

from stable_diffusion import StableDiffusion
from deepfloyd_if import IF
from models.network_grid import NeRFNetwork
from models.losses.sparsity_loss import sparsity_loss
from models.losses.sketch_loss import SketchLoss
from models.losses.silhouette_loss import SilhouetteLoss
from options_r import TrainNGPOptions
from dataset.dataset import NeRFDataset

import utils.general as utils
from utils import sketch_utils

import torch.nn.functional as F

import pyrallis

class Train:
    def __init__(self, opts):
        self.opts = opts
        self.train_type = self.opts.train_type
        #config model
        self.model = NeRFNetwork(self.opts).to(self.opts.device)
        self.initialize_model()
        self.fp16 = self.opts.fp16
        #config paths
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.expdir = os.path.join(self.opts.exp_dir, 'instant_ngp' if self.train_type == 'rec' else 'sked')
        self.expdir = os.path.join(self.expdir, self.opts.expname, self.timestamp)
        self.ckptdir = os.path.join(self.expdir, 'checkpoints')
        self.plotdir = os.path.join(self.expdir, 'plots')
        if not self.opts.debug:
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.plotdir, exist_ok=True)
        #config dataset
        if self.train_type == 'rec':
            data_dir = os.path.join('../data', self.opts.expname)
        else:
            data_dir = None

        if self.train_type == 'rec':
            H, W = 512, 512
        else:
            H, W = 128, 128
        self.dataset = NeRFDataset(data_dir, self.opts, type='train', H = H, W = W)
        self.dataloader = self.dataset.dataloader()

        val_scale = 1 if self.opts.train_type == 'rec' else 8
        self.val_dataset = NeRFDataset(data_dir, self.opts, type='val', H = H * val_scale , W = W * val_scale)
        self.val_dataloader = self.val_dataset.dataloader()
        self.valiter = itertools.cycle(iter(self.val_dataloader))

        #config loss
        if self.train_type == 'rec':
            self.criterion = nn.MSELoss(reduction='none')
        else:
            self.criterion = None
        self.sparsity_lambda = self.opts.sparsity_lambda
        #config optimizer
        self.optimizer = torch.optim.Adam(self.model.get_params(self.opts.learning_rate), betas=(0.9, 0.99), eps=1e-15)
        self.num_iters = self.opts.nepochs * len(self.dataloader)


        ##config training
        self.global_step = 0
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.save_freq = self.opts.save_freq
        self.eval_freq = self.opts.eval_freq


        ##config SD
        self.sd = None
        self.use_deepfloyd = self.opts.use_deepfloyd
        if self.train_type == 'gen':
            if self.use_deepfloyd:
                self.sd = IF(self.opts.device, vram_O = True)
            else:
                self.sd = StableDiffusion(self.opts.device).eval()
                #self.sd = StableDiffusion(self.opts.device).eval()
            if self.train_type == 'rec':
                with torch.cuda.amp.autocast(enabled = self.fp16):
                    self.dataset.precompute_latents(self.sd)


        ##config text embeddings
        if self.opts.train_type == 'gen':
            self.calc_text_embeddings()



        self.use_2d_sketch = self.opts.use_2d_sketch
        self.sketch_loss = None
        self.sil_loss = None
        if self.use_2d_sketch and self.opts.train_type == 'gen':
            sketch_canvases, sketch_poses, sketch_intrinsics, sketch_bboxes, sketch_sils, sketch_rays = self.initialize_2d_sketches()
            self.sketch_H, self.sketch_W = self.opts.sketch_H,  self.opts.sketch_W
            self.num_sketches = len(sketch_canvases)
            self.model.bbox2bitfield(sketch_bboxes, sketch_poses, sketch_intrinsics)
            self.base_mesh = None


            gt_nerf = NeRFNetwork(self.opts).to(self.opts.device)
            gt_nerf.load_checkpoint(self.opts.checkpoint_path)
            gt_nerf.eval()

            self.sketch_loss = SketchLoss(sketch_canvases, sketch_poses, sketch_intrinsics, self.opts.proximal_surface, use_kdtree = self.opts.use_kd_tree,  base_mesh=self.base_mesh, nerf_gt = gt_nerf, color_lambda = self.opts.color_lambda)

            if self.opts.sil_lambda > 0:
                self.sil_loss = SilhouetteLoss(sketch_canvases)
                self.sketch_rays = sketch_rays
    def run(self):
        if self.model.cuda_ray and self.train_type == 'rec':
            self.model.mark_untrained_grid(self.dataloader._data.poses, self.dataloader._data.intrinsics)

        pbar = tqdm.tqdm(total=self.opts.nepochs, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for epoch in range(1, self.opts.nepochs + 1):
            self.model.train()
            total_loss = 0
            for i, data in enumerate(self.dataloader):
                if self.model.cuda_ray and self.global_step % self.opts.update_extra_interval == 0:
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        iter = None if self.opts.bitfield_warmup_iters < 0 else self.global_step
                        self.model.update_extra_state(iter = iter)


                self.global_step += 1
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = self.train_step(data, pbar)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()


                loss_val = loss.item()
                total_loss += loss_val
            #print(self.print_loss(total_loss, epoch))
            if epoch % self.eval_freq == 0:
                self.evaluate(epoch)
            if epoch % self.save_freq == 0 and not self.opts.debug:
                self.save_ckpt(epoch)

            pbar.update(1)
        self.save_ckpt()
        utils.save_pickle(self.opts, os.path.join(self.expdir, 'opts.pkl'))

    def train_step(self, data, pbar):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        gt = data['gt']
        if self.train_type == 'rec':
            B, _, C = gt.shape
        else:
            B, C = rays_o.shape[0],  3



        if self.train_type == 'rec':
            bg_color = torch.ones_like(gt)
        else:
            bg_color = torch.ones(B * data['H'] * data['W'], C, device=self.opts.device)

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color= bg_color, perturb=True, force_all_rays=True)
        pred = outputs['image']

        # MSE loss
        if self.train_type == 'rec':
            loss = self.criterion(pred, gt).mean() # [B, N, 3] --> [B, N]
        else:
            if self.opts.dir_guidance:
                dirs = data['dirs']
                text_z = self.text_z[dirs]
                if self.use_deepfloyd:
                    text_z = torch.cat([self.uncond_z, text_z], dim=0)
            else:
                text_z = self.text_z
            loss = self.sd.train_step(text_z, pred.reshape(B, data['H'], data['W'], C).permute(0, 3, 1, 2))

        loss_str = ''
        if self.sparsity_lambda > 0:
            weights_sum = outputs['weights_sum']
            sp_loss = self.sparsity_lambda * sparsity_loss(weights_sum)
            loss += sp_loss
            loss_str += 'sparsity_loss: {:.4f}, '.format(sp_loss.item())
      
        if not(self.sketch_loss is None) and self.opts.sketch_lambda > 0:
            sketch_loss = self.opts.sketch_lambda * self.sketch_loss(outputs['xyzs'], outputs['sigmas'], outputs['colors'])
            loss += sketch_loss
            loss_str += 'sketch_loss: {:.4f}, '.format(sketch_loss.item())
        if not(self.sil_loss is None):
            outputs = self.model.render(self.sketch_rays['rays_o'], self.sketch_rays['rays_d'], staged=False, bg_color= bg_color, perturb=True, force_all_rays=True)
            weights_sum = outputs['weights_sum'].reshape(self.num_sketches, self.sketch_H, self.sketch_W)
            sil_loss = self.opts.sil_lambda * self.sil_loss(weights_sum)
            loss += sil_loss
            loss_str += 'sil_loss: {:.4f}, '.format(sil_loss.item())

        if self.train_type == 'gen' and not(self.global_step % self.opts.print_freq):
            pbar.set_description(loss_str)
        return pred, gt, loss

    def evaluate(self, epoch):
        self.model.eval()
        data = next(self.valiter)
        with torch.cuda.amp.autocast(enabled=self.fp16):
            pred_rgb, pred_depth, pred_normal, gt_rgb, loss, psnr = self.eval_step(data)
            if not self.opts.debug:
                utils.save_nerf_evals(pred_rgb, gt_rgb, pred_depth, pred_normal, os.path.join(self.plotdir, 'image_{}.png'.format(epoch)))
            print('Epoch: {}/{}, Val Loss: {:.4f}, PSNR: {:.4f}, lr: {:.4f} '.format(epoch, self.opts.nepochs, loss.item(), psnr.item(), self.optimizer.param_groups[0]['lr']))



    @torch.no_grad()
    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        gt = data['gt']

        if self.train_type == 'rec':# [B, H, W, 3/4]
            B, _, _, C= gt.shape
        else:
            B, C = 1,  3

        H, W = data['H'], data['W']


        bg_color = 1

        outputs = self.model.render(rays_o, rays_d, staged= self.opts.staged_rendering, bg_color=bg_color, perturb=False, force_all_rays = True)

        pred = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W).unsqueeze(1)
        pred_depth = F.interpolate(pred_depth, (H, W), mode='bilinear', align_corners=False).squeeze(1)



        pred_rgb = pred

        pred_normal = None
        if self.opts.render_normals:
            outputs = self.model.render(rays_o, rays_d, staged= self.opts.staged_rendering, shading = 'normal', bg_color=bg_color, disable_bg = True, perturb=False, force_all_rays = True)
            pred_normal = outputs['image']
            pred_normal = pred_normal.reshape(B, H, W, 3)
            pred_normal = F.interpolate(pred_normal.permute(0, 3, 1, 2), (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        if self.train_type == 'rec':
            loss = self.criterion(pred_rgb, gt).mean()
            psnr = utils.get_psnr(pred_rgb, gt)
        else:
            loss = torch.tensor(0)
            psnr = torch.tensor(0)
        return pred_rgb, pred_depth, pred_normal, gt, loss, psnr


    def save_ckpt(self, epoch = None):
        if epoch is None:
            name = 'latest'
        else:
            name = 'ckpt_{:04d}'.format(epoch)
        ckpt_path = os.path.join(self.ckptdir, name + '.pth')
        state = {
            'epoch': epoch if epoch is not None else self.opts.nepochs,
            'global_step': self.global_step,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density
        state['optimizer'] = self.optimizer.state_dict()
        state['scaler'] = self.scaler.state_dict()

        state['model'] = self.model.state_dict()
        torch.save(state, ckpt_path)

    def print_loss(self, loss, epoch):
        print_str = 'Epoch: {}/{}, Loss: {:.4f}'.format(epoch, self.opts.nepochs, loss / len(self.dataloader))
        return print_str

    def calc_text_embeddings(self):

        self.uncond_z = self.sd.get_text_embeds([''])
        if not self.opts.dir_guidance:
            text_z = self.sd.get_text_embeds(self.opts.text)
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                t = '{}, {} view'.format(self.opts.text, d)
                text_z.append(self.sd.get_text_embeds([t]))
        self.text_z = text_z
    def initialize_model(self):
        if self.opts.checkpoint_path is None:
            return
        self.model.load_checkpoint(self.opts.checkpoint_path)

    def update_training_progression(self):
        # Config 2 stage training
        # 1. render higher resolution images
        # 2. Update amount of noise added in SD (?)
        # 3. Decrease range of radius to focus more on texture details (?)
        # 4. Decrease learning rate (?)
        new_h = self.dataset.H * 2
        new_w = self.dataset.W * 2

        self.dataset.update_intrinsics(new_h, new_w)



    def initialize_bitfield(self):
        edit_sketch_path = self.opts.edit_sketch_path
        if edit_sketch_path is None:
            return

        self.sketch_shape = self.model.initialize_bitfield(edit_sketch_path)


    def initialize_2d_sketches(self):
        sketch_path = self.opts.sketch_path
        if sketch_path is None:
            return

        self.sketches = sketch_utils.Sketches(sketch_path, H = self.opts.sketch_H, W = self.opts.sketch_W)
        sketch_canvases, sketch_poses, sketch_intrinsics, sketch_bounding_boxes, sketch_silhouettes, sketch_rays = self.sketches.get_sketches()
        return sketch_canvases, sketch_poses, sketch_intrinsics, sketch_bounding_boxes, sketch_silhouettes, sketch_rays



if __name__ == '__main__':
    utils.seed_everything(455)
    opts = pyrallis.parse(config_class=TrainNGPOptions)
    trainer = Train(opts)
    trainer.run()
    a = 0


