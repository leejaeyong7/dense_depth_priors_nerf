import os
import random
import math
import sqlite3

import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from torchvision import transforms

from .error_sources import add_missing_depth, add_quadratic_depth_noise

def is_in_list(file, list_to_check):
    for h in list_to_check:
        if h in file:
            return True
    return False

def get_whitelist(dataset_dir, dataset_split):
    whitelist_txt = os.path.join(dataset_dir, "scannetv2_{}.txt".format(dataset_split))
    scenes = pd.read_csv(whitelist_txt, names=["scenes"], header=None)
    return scenes["scenes"].tolist()

def apply_filter(files, dataset_dir, dataset_split):
    whitelist = get_whitelist(dataset_dir, dataset_split)
    return [f for f in files if is_in_list(f, whitelist)]

def read_rgb(rgb_file):
    bgr = cv2.imread(rgb_file)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    assert rgb.shape[2] == 3

    to_tensor = transforms.ToTensor()
    rgb = to_tensor(rgb)
    return rgb

def read_depth(depth_file, s_depth_file):
    depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    s_depth = np.load(s_depth_file)
    assert len(depth.shape) == 2

    valid_depth = depth.astype('bool')
    depth = depth.astype('float32')

    # 16bit integer range corresponds to range 0 .. 65.54m
    # use the first quarter of this range up to 16.38m and invalidate depth values beyond
    # scale depth, such that range 0 .. 1 corresponds to range 0 .. 16.38m
    max_depth = np.float32(2 ** 16 - 1) / 4.
    depth = depth / max_depth
    s_depth = (s_depth * 1000) / max_depth

    # depth = depth / (1000 * 16.38375)
    # s_depth = s_depth / (16.38375)


    invalidate_mask = depth > 1.
    depth[invalidate_mask] = 0.
    valid_depth[invalidate_mask] = False
    return transforms.functional.to_tensor(depth), torch.from_numpy(s_depth), transforms.functional.to_tensor(valid_depth)


def convert_depth_completion_scaling_to_m(depth):
    # convert from depth completion scaling to meter, that means map range 0 .. 1 to range 0 .. 16,38m
    return depth * (2 ** 16 - 1) / 4000.

def convert_m_to_depth_completion_scaling(depth):
    # convert from meter to depth completion scaling, which maps range 0 .. 16,38m to range 0 .. 1
    return depth * 4000. / (2 ** 16 - 1)

# def convert_depth_completion_scaling_to_m(depth):
#     # convert from depth completion scaling to meter, that means map range 0 .. 1 to range 0 .. 16,38m
#     return depth * (1000 * 16.38375)

# def convert_m_to_depth_completion_scaling(depth):
#     # convert from meter to depth completion scaling, which maps range 0 .. 16,38m to range 0 .. 1
#     return depth / (1000 * 16.38375)

def get_normalize(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    unnormalize = transforms.Normalize(mean=np.divide(-mean, std), std=(1. / std))
    return normalize, unnormalize

def get_pretrained_normalize():
    normalize = dict()
    unnormalize = dict()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize['rgb'], unnormalize['rgb'] = get_normalize(mean, std)
    normalize['rgbd'], unnormalize['rgbd'] = get_normalize(np.concatenate((mean, [0.,]), axis=0), np.concatenate((std, [1.,]), axis=0))
    return normalize, unnormalize

def resize_sparse_depth(depths, valid_depths, size):
    device = depths.device
    orig_size = (depths.shape[1], depths.shape[2])
    col, row = torch.meshgrid(torch.tensor(range(orig_size[1])), torch.tensor(range(orig_size[0])), indexing='ij')
    rowcol2rowcol = torch.stack((row.t(), col.t()), -1)
    rowcol2rowcol = rowcol2rowcol.unsqueeze(0).expand(depths.shape[0], -1, -1, -1)
    image_index = torch.arange(depths.shape[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, orig_size[0], orig_size[1], 1)
    rowcol2rowcol = torch.cat((image_index, rowcol2rowcol), -1)
    factor_h, factor_w = float(size[0]) / float(orig_size[0]), float(size[1]) / float(orig_size[1])
    depths_out = torch.zeros((depths.shape[0], size[0], size[1]), device=device)
    valid_depths_out = torch.zeros_like(depths_out).bool()
    idx_row_col = rowcol2rowcol[valid_depths]
    idx_row_col_resized = idx_row_col
    idx_row_col_resized = ((idx_row_col + 0.5) * torch.tensor((1., factor_h, factor_w))).long() # consider pixel centers
    depths_out[idx_row_col_resized[..., 0], idx_row_col_resized[..., 1], idx_row_col_resized[..., 2]] \
        = depths[idx_row_col[..., 0], idx_row_col[..., 1], idx_row_col[..., 2]]
    valid_depths_out[idx_row_col_resized[..., 0], idx_row_col_resized[..., 1], idx_row_col_resized[..., 2]] = True
    return depths_out, valid_depths_out

class SparseScanNetDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_dir, data_split, random_rot=0, load_size=(240, 320), \
            horizontal_flip=False, color_jitter=None, depth_noise=False, missing_depth_percent=0.998):
        super(SparseScanNetDataset, self).__init__()

        # apply train val test split
        self.root = Path(dataset_dir)
        self.scene_root  = self.root / data_split
        scenes = list(self.scene_root.iterdir())

        self.samples = []
        for scene in scenes:
            images_path = scene / 'images'
            for image_path in images_path.iterdir():
                if image_path.name.endswith('.jpg'):
                    image_name = image_path.name
                    iid = int(image_name.split('.')[0])
                    depth_path = scene / 'depth' / f'{iid:06d}.png'
                    sdepth_path = scene / 's_depth' / f'{iid:06d}.npy'
                    if depth_path.exists() and sdepth_path.exists():
                        self.samples.append({
                            'image': image_path,
                            'depth': depth_path,
                            's_depth': sdepth_path
                        })
        
        # transformation
        self.normalize, self.unnormalize = get_pretrained_normalize()
        self.random_rot = random_rot
        self.load_size = load_size
        self.horizontal_flip = horizontal_flip
        self.color_jitter = color_jitter

        # depth sampling
        self.missing_depth_percent = missing_depth_percent # add percentage of missing depth
        self.depth_noise = depth_noise # add gaussian depth noise

    def __getitem__(self, index):
        sample = self.samples[index]

        rgb = read_rgb(str(sample['image']))
        depth, s_depth, valid_depth = read_depth(sample['depth'], sample['s_depth'])
        
        # precompute random rotation
        rot = random.uniform(-self.random_rot, self.random_rot)

        # precompute resize and crop
        tan_abs_rot = math.tan(math.radians(abs(rot)))
        border_width = math.ceil(self.load_size[0] * tan_abs_rot)
        border_height = math.ceil(self.load_size[1] * tan_abs_rot)
        top = math.floor(0.5 * border_height)
        left = math.floor(0.5 * border_width)
        resize_size = (self.load_size[0] + border_height, self.load_size[1] + border_width)

        # precompute random horizontal flip
        apply_hflip = self.horizontal_flip and random.random() > 0.5

        # create a sparsified depth and a complete target depth
        target_valid_depth = valid_depth.clone()
        target_depth = depth.clone()
        depth, valid_depth = self.sample_depth_at_image_features(depth, s_depth)
        depth, valid_depth = add_missing_depth(depth, valid_depth, self.missing_depth_percent)
        
        rgbd = torch.cat((rgb, depth), 0)
        data = {'rgbd': rgbd, 'valid_depth' : valid_depth, 'target_depth' : target_depth, 'target_valid_depth' : target_valid_depth}

        # apply transformation
        for key in data.keys():
            # resize
            if key == 'rgbd':
                # resize such that sparse points are preserved
                B_depth, data['valid_depth'] = resize_sparse_depth(data['rgbd'][3, :, :].unsqueeze(0), data['valid_depth'], resize_size)
                B_rgb = transforms.functional.resize(data['rgbd'][:3, :, :], resize_size, interpolation=transforms.functional.InterpolationMode.NEAREST)
                data['rgbd'] = torch.cat((B_rgb, B_depth), 0)
            else:
                # avoid blurring the depth channel with invalid values by using interpolation mode nearest
                data[key] = transforms.functional.resize(data[key], resize_size, interpolation=transforms.functional.InterpolationMode.NEAREST)
            
            # augment color
            if key == 'rgbd':
                if self.color_jitter is not None:
                    cj = transforms.ColorJitter(brightness=self.color_jitter, contrast=self.color_jitter, saturation=self.color_jitter, \
                        hue=self.color_jitter)
                    data['rgbd'][:3, :, :] = cj(data['rgbd'][:3, :, :])
            
            # rotate
            if self.random_rot != 0:
                data[key] = transforms.functional.rotate(data[key], rot)
            
            # crop
            data[key] = transforms.functional.crop(data[key], top, left, self.load_size[0], self.load_size[1])

            # horizontal flip
            if apply_hflip:
                data[key] = transforms.functional.hflip(data[key])
            
            # normalize
            if key == 'rgbd':
                data[key] = self.normalize['rgbd'](data[key])
                # scale depth according to resizing due to rotation
                data[key][3, :, :] /= (1. + tan_abs_rot)

        # add depth noise
        if self.depth_noise:
            data['rgbd'][3, :, :] = convert_m_to_depth_completion_scaling(add_quadratic_depth_noise( \
                convert_depth_completion_scaling_to_m(data['rgbd'][3, :, :]), data['valid_depth'].squeeze()))

        return data

    def sample_depth_at_image_features(self, depth, s_depth):
        keypoints_mask = (s_depth > 0)
        return depth * keypoints_mask.float(), keypoints_mask

    def __len__(self):
        return len(self.samples)
