import torch
import numpy as np
import os

import sys
sys.path.append('../../code')
sys.path.append('../../code/models')
import utils.general as utils
import utils.rend_utils as rend_utils
import cv2




from PIL import Image


class Sketches:
    def __init__(self, sketch_dir, H=None, W=None, device='cuda', scale=1, offset=[0, 0, 0], fill_sketch=True,
                 preprocess_sketch=True):

        self.H = H
        self.W = W
        self.sketches = []

        self.transforms_file = os.path.join(sketch_dir, 'meta_data.pkl')
        self.transforms = utils.load_pickle(self.transforms_file)
        self.intrinsics = self.transforms['intrinsics']
        self.poses = self.transforms['poses']
        sketch_files = sorted(os.listdir(os.path.join(sketch_dir, 'sketches')))
        self.sketches = []
        self.bounding_boxes = []
        self.silhouettes = []
        for f in sketch_files:
            f_sketch = os.path.join(sketch_dir, 'sketches', f)
            sketch = cv2.imread(f_sketch, cv2.IMREAD_UNCHANGED)
            sketch = (sketch / 255.).astype('float32')
            if sketch.shape[-1] == 4:
                # sketches is RGBA images
                sketch = cv2.cvtColor(sketch, cv2.COLOR_BGRA2RGBA)
                sketch = utils.rgba2rgb(sketch, input_type='numpy')

            # turn rgb mask to binary image
            sketch = cv2.cvtColor((sketch * 255.).astype('uint8'), cv2.COLOR_RGB2GRAY)
            sketch = (sketch < 128).astype('float32')

            h = sketch.shape[0]
            if self.H is None and self.W is None:
                self.H, self.W = sketch.shape[:2]
            else:
                sketch = cv2.resize(sketch, (self.W, self.H), interpolation=cv2.INTER_AREA)

            self.sketches.append(sketch)
            # extract bounding box of sketch
            bbox = extract_bbox(sketch)
            self.bounding_boxes.append(bbox)

            s = (sketch * 255).astype('uint8')
            r = cv2.rectangle(s, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 128, 2)
            # extract silhouette of sketch
            silhouette = Image.open(os.path.join(sketch_dir, 'shapes', f))
            silhouette = np.array(silhouette)
            silhouette = cv2.resize(silhouette, (self.W, self.H), interpolation=cv2.INTER_AREA)
            silhouette = (silhouette / 255.).astype('float32')
            silhouette[sketch > 0.5] = 1
            self.silhouettes.append(silhouette)

        self.silhouettes = torch.from_numpy(np.stack(self.silhouettes, axis=0)).to(device, dtype=torch.float32)
        self.sketches = torch.from_numpy(np.stack(self.sketches, axis=0)).to(device, dtype=torch.float32)
        self.bounding_boxes = np.stack(self.bounding_boxes, axis=0)
        self.intrinsics = self.intrinsics * (self.H / h)

        ##precompute rays to speed up rendering
        self.rays = rend_utils.get_rays(self.poses, self.intrinsics, self.H, self.W, -1)

    def get_sketches(self, indices=None):
        if indices is not None:
            return self.sketches[indices], self.poses[indices], torch.from_numpy(self.intrinsics).to(
                self.sketches.device).reshape(1, 4).repeat(len(indices), 1), self.bounding_boxes[indices], \
            self.silhouettes[indices], self.rays[indices]
        else:
            # return all sketches, poses, intrinsics
            return self.sketches, self.poses, torch.from_numpy(self.intrinsics).to(self.sketches.device).reshape(1, 4).repeat(len(self.sketches), 1), self.bounding_boxes, self.silhouettes, self.rays



def extract_sketch(image, edit):
    #base_image, edit_image: PIL images or numpy images (H, W, 3)
    #Output:(H, W) numpy binary image: 1 for sketch, 0 o.w.
    if isinstance(image, Image.Image):
        image = np.array(image).astype('int32')
    if isinstance(edit, Image.Image):
        edit = np.array(edit).astype('int32')
    diff = np.abs(image - edit).astype('uint8')
    diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    #smooth out diff
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    diff = (diff > 0.5)
    Image.fromarray((diff * 255).astype('uint8')).save('diff.png')
    return diff.astype('float')


def extract_bbox(sketch):
    #sketch: (H, W) numpy binary image: 1 for sketch, 0 o.w.
    #Output: (x1, y1, x2, y2) bounding box of sketch
    #given a sketch extract the bounding box
    sketch = (sketch * 255.).astype('uint8')
    contours, _ = cv2.findContours(sketch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, x + w, y + h


def fill_sketch_func(sketch):
    #sketch: (H, W) numpy binary image: 1 for sketch, 0 o.w.
    #Output: (H, W) numpy binary image: 1 for sketch, 0 o.w.
    #given a sketch fill it in
    sketch = (sketch * 255.).astype('uint8')
    contours, _ = cv2.findContours(sketch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(sketch.shape, dtype = np.uint8)
    points = contours[0]
    mask = cv2.fillPoly(mask, [points], 255)
    return (mask / 255.).astype('float')

