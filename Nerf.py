import numpy as np
from numpy._core.numeric import astype
from torch import nn
import torch as tc
from torch.utils.data import Dataset

import json
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt


devcie = tc.device("cuda" if tc.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class NERF(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        # config
        self.n_l_net = 8
        self.d_filter = 256
        self.d_input = 3
        self.skip = [4]

        self.n_u_net = 3
        self.d_u_filter = int(self.d_filter / 2)
        self.skip_u = []

        self.act = nn.functional.relu
        layers = [nn.Linear(self.d_input, self.d_filter)]
        # lower layers
        for i in range(1, self.n_l_net):
            layers.append(
                nn.Linear(self.d_filter + self.d_input, self.d_filter)
                if i in self.skip
                else nn.Linear(self.d_filter, self.d_filter)
            )
        self.low_lays = nn.ModuleList(layers)

        # upper layers
        self.vox_unit = nn.Linear(self.d_filter, 1)
        self.feat = nn.Linear(self.d_filter, self.d_filter)

        layers = [
            nn.Linear(self.d_filter + 3, self.d_u_filter)
        ]  # assume view dir is 3d.
        for i in range(1, self.n_u_net):
            layers.append(nn.Linear(self.d_u_filter, self.d_u_filter))
        self.up_layers = nn.ModuleList(layers)

        self.rgb_unit = nn.Linear(self.d_u_filter, 3)

    def forward(  # TODO: whole batch or single rays ? that is whole batch.
        self, x, view_dir
    ):
        x_ = x
        for i, layer in enumerate(self.low_lays):
            if i in self.skip:
                x = layer(tc.cat((x, x_), dim=-1))
            else:
                x = layer(x)
            x = self.act(x)

        vox = self.vox_unit(x)
        feat = self.feat(x)

        feat = tc.cat((feat, view_dir), dim=-1)
        for i, layer in enumerate(self.up_layers):
            feat = layer(feat)
            feat = self.act(feat)
        rgb = self.rgb_unit(feat)  # NOTE: Linear act rgb.
        return tc.cat((rgb, vox), dim=-1)


class BlenderDataset(Dataset):
    def __init__(self):
        # Contain poses (tensor), images (tensor C x H x W), height, width, camera focal length from train set.
        self.basedir = os.path.join(ROOT_DIR, "data", "nerf_synthetic", "lego")
        meta = None
        with open(os.path.join(self.basedir, "transforms_train.json")) as fl:
            meta = json.load(fl)
        imgs = []
        poses = []
        for frame in meta["frames"]:
            fn = os.path.join(self.basedir, frame["file_path"] + ".png")
            img = imageio.imread(fn)
            imgs.append(img)
            poses.append(np.array(frame["transform_matrix"]))
        self.poses = tc.from_numpy(np.array(poses).astype(np.float32))
        self.imgs = (np.array(imgs) / 255.0).astype(np.float32)
        self.imgs = (255.0 * (1 - self.imgs[..., -1:])) + (
            self.imgs[..., :-1] * self.imgs[..., -1:]
        )
        self.imgs = tc.from_numpy(self.imgs)
        self.H, self.W = imgs[0].shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        return self.imgs[index]


if __name__ == "__main__":
    k = BlenderDataset()
