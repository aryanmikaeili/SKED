from dataclasses import dataclass, field

import abc
import os
import json

from typing import List

from utils.general import load_pickle
@dataclass
class BaseOptions(abc.ABC):
    debug: bool = False
    exp_dir: str = '../exps/'
    device: str = 'cuda'


@dataclass
class TrainNGPOptions(BaseOptions):
    #dataset options
    preload: bool = True #pre-load the dataset
    scale: float = 1 #scale of the cameras
    offset: List[float] = field(default_factory=lambda: [0., 0., 0.]) #offset of the cameras
    bound: float = 1. #bound of the world
    radius_range: tuple = (1., 2) #sampling range of the camera distance
    fov_range: tuple = (40, 80) #sampling range of the camera fov
    angle_overhead: float = 30 #parameters for direction guidance
    angle_front: float = 70 #parameters for direction guidance

    #general options
    fp16: bool = True #used mixed precision training
    cuda_ray: bool = True #use cuda raymarching
    train_type: str = 'gen' #generation or reconstruction
    activation: str = 'softplus' #use softplus or exponential activation
    blob_type: str = 'blob' #blob or gaussian as initial density

    #training options
    use_deepfloyd: bool = False #use deepfloyd as guidance diffusion model
    num_rays: int = 4096 #batch size of rays for reconstruction
    nepochs: int = 100 #number of training epochs
    learning_rate: float = 5e-3 #learning rate
    sparsity_lambda: float = 5e-4 #sparsity loss weight
    sketch_lambda: float = 5e-6 #sketch loss weight
    sil_lambda: float = 1. #silhouette loss weight
    color_lambda: float = 5. #color sketch loss weight
    save_freq: int = 10 #save checkpoint frequency
    eval_freq: int = 10 if train_type == 'rec' else 1 #evaluation frequency
    print_freq: int = 1 #print frequency

    #rendering_options
    min_near: float = 0.1 #minimum near plane distance
    density_thresh: float = 10 if activation == 'exp' else 5 #density threshold for pruning
    bg_radius: float = 1.5 #radius of the background
    update_extra_interval: int = 16 #frequency of bitfield pruning
    staged_rendering: bool = True #use staged rendering

    #eval options
    render_normals: bool = True #render normals in evaluation

    #network options
    num_layers: int = 3
    hidden_dim: int = 64
    num_layers_bg: int = 2
    num_heads: int = 4
    hidden_dim_bg: int = 64

    #text options
    text: str = 'a cat wearing a chef hat'
    dir_guidance: bool = True #use direction guidance

    #directory and name options
    expname: str = 'cat' #experiment name/name of data folder for reconstruction
    if train_type == 'gen':
        expname += '_{}'.format(text)

    #checkpoint options
    checkpoint_path = '../data/cat/base_nerf/checkpoint.pth' #checkpoint to use from the base NeRF

    #sketch options
    use_2d_sketch: bool = True
    sketch_path: str = '../data/cat'
    proximal_surface: float = 0.025
    occ_thresh: float = 0.001
    bitfield_warmup_iters: int = 1000
    sketch_H: int = 128
    sketch_W: int = 128
    use_kd_tree: bool = False

@dataclass
class RenderOptions:
    device: str = 'cuda'
    exp_dir: str = '../exps/sked/cat_a cat wearing a chef hat/2023_09_22_14_35_04'
    ckpt_path: str = os.path.join(exp_dir, 'checkpoints/latest.pth')
    model_opts_path: str = os.path.join(exp_dir, 'opts.pkl')
    H: int = 512
    W: int = 512
    fov_range: tuple = (60, 80)
    out_dir: str = os.path.join(exp_dir, 'renders')

if __name__ == '__main__':
    opts = TrainNGPOptions()
    with open('../configs/train_sked.yaml', 'w') as f:
        json.dump(opts.__dict__, f)