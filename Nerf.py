import numpy as np
from torch import nn
import torch as tc

import json
import os
import imageio.v2 as imageio


dvc = tc.device("cuda" if tc.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_rays(pose, W, H, focal):
    # TODO : optimize.
    """

    Involved in inference phase.
    Returns
    _______

    rays_o : Tensor
        rays origin.
    rays_d : Tensor
        rays direction, expecting shape (H, W, 3)


    """
    # NOTE : Why divide to focal length ? to scaling. Why don't scale origin ? Unecessary.
    # NOTE : Should is rays_dir (H, W, 3) or (W, H, 3). In there (W, H, 3)

    x_grid, y_grid = tc.meshgrid(
        tc.arange(W, dtype=tc.float32, device=dvc),
        tc.arange(H, dtype=tc.float32, device=dvc),
        indexing="xy",
    )

    direction = tc.stack(
        [
            (x_grid - 0.5 * W) / focal,
            (0.5 * H - y_grid) / focal,
            -tc.ones_like(x_grid),
        ],
        dim=-1,
    )
    # direction shape (W, H, 3)
    rays_d = tc.sum(pose[:3, :3] * direction[..., None, :], dim=-1)
    rays_o = pose[:-1, -1].expand(rays_d.shape)  # NOTE : redundant.

    # TEST: test cuda.
    check = rays_o.is_cuda and rays_d.is_cuda
    if dvc == "cuda" and not check:
        raise Exception("Non-cuda set")

    return rays_o, rays_d


def sample_stratified(rays_o, rays_d, near, far, n_samples):
    """Sample along rays.

    Paprameters
    ___________
    rays_o: Tensor
        rays' origin.
    rays_d: Tensor
        rays' direction.
    near: scalar.
        Bound value to sample, corresponding to
    """
    t_vals = tc.linspace(0.0, 1.0, n_samples, device=dvc)
    z_vals = near * (1 - t_vals) + far * t_vals  # n_samples
    z_vals = z_vals.expand(list(rays_d.shape[:-1]) + [n_samples])  # [W, H, n_samples]

    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
    )  # [W, H, N_samples, 3]
    print(pts.shape)
    return pts, z_vals


class NERF(nn.Module):
    """Nerf model implementation"""

    def __init__(self, d_l_input, d_u_input):
        """The architecture from the source code of the paper."""
        super().__init__()
        # Config.
        self.n_l_net = 8
        self.d_filter = 256
        self.d_l_input = d_l_input
        self.skip = [4]

        self.n_u_net = 3
        self.d_u_filter = int(self.d_filter / 2)
        self.skip_u = []
        self.d_u_input = d_u_input

        self.act = nn.functional.relu
        layers = [nn.Linear(self.d_l_input, self.d_filter)]
        # lower layers
        for i in range(1, self.n_l_net):
            layers.append(
                nn.Linear(self.d_filter + self.d_l_input, self.d_filter)
                if i in self.skip
                else nn.Linear(self.d_filter, self.d_filter)
            )
        self.low_lays = nn.ModuleList(layers)

        # upper layers
        self.vox_unit = nn.Linear(self.d_filter, 1)
        self.feat = nn.Linear(self.d_filter, self.d_filter)
        # assume view dir is 3d.
        layers = [nn.Linear(self.d_filter + self.d_u_input, self.d_u_filter)]
        for i in range(1, self.n_u_net):
            layers.append(nn.Linear(self.d_u_filter, self.d_u_filter))
        self.up_layers = nn.ModuleList(layers)

        self.rgb_unit = nn.Linear(self.d_u_filter, 3)

    # NOTE : whole batch or single rays ? that is whole batch.
    def forward(self, x, view_dir):
        """Forward path.

        Paprameters
        ___________

        x : Tensor
            The encoded coordiantes of the samples on rays, expecting shape (batch_size, d_l_input).
        view_dir : Tensor
            The encoded view direction vector of the sample on rays, expecting shape (batch_size, d_u_input).

        Returns
        _______

        Tensor
            a tensor has rgb, voxel value, expecting shape (batch_size, 3 + 1).

        """

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


class BlenderDataset:
    """Source: http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip

    Only load train set from lego dataset.

    Attributes
    __________

    poses : Tensor
        Poes matrix of images, expecting shape (n_image, 4, 4)
    imgs : Tensor
        Images data that has been scaled, expecting shape (n_images, height, width, 3),
    H : scalar
        Height of the images.
    W : scalar
        Width of the images.
    focal : scalar
        Focal length of the camera.
    """

    def __init__(self):
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
            # DEBUG:  purpose
            break
            # DEBUG: purpose

        self.poses = tc.from_numpy(np.array(poses).astype(np.float32))
        self.imgs = (np.array(imgs) / 255.0).astype(np.float32)
        # white background
        self.imgs = (255.0 * (1 - self.imgs[..., -1:])) + (
            self.imgs[..., :-1] * self.imgs[..., -1:]
        )
        self.imgs = tc.from_numpy(self.imgs)
        self.H, self.W = imgs[0].shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        # TEST: test cuda.
        temp_img = self.imgs[idx].to(device=dvc)
        if dvc == "cuda" and not temp_img.is_cuda:
            raise Exception("Non cuda set")
        return temp_img, *get_rays(self.poses[idx], self.W, self.H, self.focal)


def train():
    # config
    n_iter = 10000
    near = 2
    far = 6
    n_coarse_sample = 6

    # ------

    dataset = BlenderDataset()
    # run train.
    for _ in range(n_iter):
        img_idx = np.random.randint(len(dataset))
        gt_rgb, rays_o, rays_d = dataset[img_idx]
        pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_coarse_sample)


if __name__ == "__main__":
    data = BlenderDataset()
    imgs, rays_o, rays_d = data[0]
    sample_stratified(rays_o, rays_d, 2, 6, 10)
