import torch
import torch.nn.functional as F

from activation import trunc_exp
from renderer import NeRFRenderer

from encoding import get_encoder

import sys
sys.path.append('../')
from utils.nerf_utils import safe_normalize, MLP



class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt):

        super().__init__(opt)

        self.num_layers = opt.num_layers
        self.hidden_dim = opt.hidden_dim

        self.encoder, self.in_dim = get_encoder('tiledgrid', input_dim=3, log2_hashmap_size=16, desired_resolution=2048 * self.bound)



        self.sigma_net = MLP(self.in_dim, 4, self.hidden_dim, self.num_layers, bias=True)

        self.activation = F.softplus if opt.activation == 'softplus' else trunc_exp
        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = opt.num_layers_bg
            self.hidden_dim_bg = opt.hidden_dim_bg

            # use a very simple network to avoid it learning the prompt...
            # self.encoder_bg, self.in_dim_bg = get_encoder('tiledgrid', input_dim=2, num_levels=4, desired_resolution=2048)
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=4)

            self.bg_net = MLP(self.in_dim_bg, 3, self.hidden_dim_bg, self.num_layers_bg, bias=True)

        else:
            self.bg_net = None

    # add a density blob to the scene center
    def density_blob(self, x):
        # x: [B, N, 3]
        d = (x ** 2).sum(-1)
        if self.opt.blob_type == 'gaussian':
            g = 5 * torch.exp(-d / (2 * 0.2 ** 2))
        else:
            g = 10 * (1 - torch.sqrt(d) / 0.5)
        return g

    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        h = self.encoder(x, bound=self.bound)

        h = self.sigma_net(h)

        sigma = self.activation(h[..., 0] + self.density_blob(x))

        albedo = torch.sigmoid(h[..., 1:])
        return sigma, albedo

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))

        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon,
            0.5 * (dy_pos - dy_neg) / epsilon,
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal


    def normal(self, x):

        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal[torch.isnan(normal)] = 0

        return normal


    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        if shading == 'albedo':
            # no need to query normal
            sigma, color = self.common_forward(x)
            normal = None

        else:
            # query normal

            sigma, albedo = self.common_forward(x)
            normal = self.normal(x)
            #normal = self.autograd_normal(x)

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
                color = torch.clamp(color, 0, 1)
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)

        return sigma, color, normal


    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        sigma, albedo = self.common_forward(x)

        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):
        h = self.encoder_bg(d) # [N, C]
        h = self.bg_net(h)
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)


        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params
    def load_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            return
        ckpt = torch.load(checkpoint_path, map_location=self.opt.device)
        if 'model' not in ckpt:
            self.load_state_dict(ckpt)
            return
        self.load_state_dict(ckpt['model'], strict=False)
        if self.opt.cuda_ray:
            if 'mean_count' in ckpt:
                self.mean_count = ckpt['mean_count']
            if 'mean_density' in ckpt:
                self.mean_density = ckpt['mean_density']

