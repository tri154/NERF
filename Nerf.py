import numpy as np
from torch import nn
import torch as tc

import json
import os
import cv2

import matplotlib.pyplot as plt


dvc = tc.device("cuda" if tc.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# CONFIG

# sampling
n_coarse_samples = 6
n_fine_samples = 5
near = 2
far = 6

# encoding
L_pts = 10
L_viewdir = 4
# coarse model
c_hyperparameter = {
    "n_l_net": 2,
    "d_filter": 256,
    "d_l_input": 3 * (1 + 2 * L_pts),
    "l_skip": [],
    "n_u_net": 1,
    "d_u_filter": 128,
    "d_u_input": 3 * (1 + 2 * L_viewdir),
}


# fine model
f_hyperparameter = {
    "n_l_net": 6,
    "d_filter": 256,
    "d_l_input": 3 * (1 + 2 * L_pts),
    "l_skip": [4],
    "n_u_net": 1,
    "d_u_filter": 128,
    "d_u_input": 3 * (1 + 2 * L_viewdir),
}

# train
n_iter = 10000
chunk_size = 2**4
lr = 5e-4
dis_rate = 25


# Blender data
scale_ratio = 4
# END CONFIG


def get_rays(pose, H, W, focal):
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
    # NOTE : Should rays_dir be (H, W, 3) or (W, H, 3). In there (W, H, 3)

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
    # direction shape (H, H, 3)
    rays_d = tc.sum(pose[:3, :3] * direction[..., None, :], dim=-1)
    rays_d = tc.transpose(rays_d, dim0=0, dim1=1)
    rays_o = pose[:-1, -1].expand(rays_d.shape)  # NOTE : redundant.

    # # TEST: test cuda.
    # check = rays_o.is_cuda and rays_d.is_cuda
    # if dvc == "cuda" and not check:
    #     raise Exception("Non-cuda set")

    return rays_o, rays_d


def sample_stratified(rays_o, rays_d, near, far, n_samples):
    """Sample along rays.

    Parameters
    ___________
    rays_o: Tensor
        rays' origin.
    rays_d: Tensor
        rays' direction.
    near: scalar.
        Bound value to sample, corresponding to

    Returns
    _______
    pts : Tensor
        sampled points.
    z_vals : Tensor
        Sampling scalars range [0, 1].
    """
    t_vals = tc.linspace(0.0, 1.0, n_samples, device=dvc)
    z_vals = near * (1 - t_vals) + far * t_vals  # n_samples
    z_vals = z_vals.expand(*rays_d.shape[:-1], n_samples)  # [H, W, n_samples]
    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
    )  # [H, W, N_samples, 3]
    return pts, z_vals


def sample_herarchical(z_vals, weights, n_h_sample, rays_o, rays_d):
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])  # (W, H, n_sample - 1)

    pdf = (weights + 1e-5) / tc.sum(
        weights + 1e-5, dim=-1, keepdim=True
    )  # (W, H, n_sample)
    cdf = tc.cumsum(pdf, dim=-1)
    cdf = tc.cat((tc.ones_like(cdf[..., :1]), cdf), dim=-1)  # (W, H, n_sample + 1)

    u = tc.linspace(0.0, 1.0, steps=n_h_sample, device=dvc)
    u = u.expand(*cdf.shape[:-1], u.shape[-1])  # (W, H, n_h_sample)
    u = u.contiguous()
    # to find inverse cdf in the new sample.
    inds = tc.searchsorted(cdf, u, right=True)  # (W, H, n_h_sample)

    below = tc.clamp(inds - 1, min=0)
    above = tc.clamp(inds, max=cdf.shape[-1] - 1)
    # create bins.
    inds_g = tc.stack(
        [below, above], dim=-1
    )  # (W, H, n_h_sample, 2) = (800, 800, 10, 2)

    matched_shape = (
        *inds_g.shape[:-1],
        cdf.shape[-1],
    )  # (W, H, n_h_sample, n_sample + 1) =  (800, 800, 10, 7)
    # (800, 800, 10, 2)
    cdf_g = tc.gather(
        cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g
    )  # replace index by value of the cdf, ranging [0, 1].
    bins_g = tc.gather(
        z_vals_mid.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g
    )

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = tc.where(denom < 1e-5, tc.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    new_z_samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    new_z_samples = new_z_samples.detach()
    z_vals_combined, _ = tc.sort(tc.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    # (W, H, n_samples + n_h_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
    return pts, z_vals_combined, new_z_samples


def positional_encoding(pts, L):
    """

    Parameters
    ___________

    pst : Tensor
        points to encode, expecting shape (N_points, 3)
    L : scalar
        a parameter in the formula.

    Returns
    _______

    Tensor
        encoded points, expecting shape (N_points, d_input * (1 + 2L).
    """
    lin = 2.0 ** tc.linspace(0.0, L - 1, L, device=dvc, dtype=pts.dtype)
    encoding = [pts]
    for freq in lin:
        for fn in [tc.sin, tc.cos]:
            encoding.append(fn(freq * pts))
    return tc.cat(encoding, dim=-1)


def prepare_chunks(input, chunks_size):
    """Split to chunks
    Parameters
    __________
    input : Tensor, expecting shape (W * H * n_sample, n_feature)


    Returns
    _______

    list: list of chunks.
    """
    # res = [input[i : i + chunks_size] for i in (0, input.shape[0], chunks_size)]
    res = []
    for i in range(0, input.shape[0], chunks_size):
        res.append(input[i : i + chunks_size])
    return res


def cumprod_exclusive(tensor):
    cumprod = tc.cumprod(tensor, dim=-1)
    cumprod = cumprod.roll(1, -1)
    cumprod[..., 0] = 1
    return cumprod


def toRGB(raw, z_vals, rays_d):
    """From rgb, voxel of samples to rgb of rays."""
    dist = z_vals[..., 1:] - z_vals[..., :-1]
    temp_t = tc.tensor([1e10], dtype=dist.dtype, device=dvc)  # H, W, n_sample
    dist = tc.cat((dist, temp_t.expand(*dist.shape[:-1], 1)), dim=-1)

    dist = dist * tc.norm(rays_d, dim=-1, keepdim=True)  # (H, W n_sample) * (H, W, 1)
    # raw (H, W, n_sample, 4)
    alpha = 1.0 - tc.exp(-nn.functional.relu(raw[..., 3]) * dist)  # (H, W, n_sample)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)  # (H, W, n_sample)

    rgb = tc.sigmoid(raw[..., :3])  # (H, W, n_sample, 3)
    rgb_map = tc.sum(weights[..., None] * rgb, dim=-2)
    acc_map = 1 - tc.sum(weights, -1, keepdim=True)

    # for white backgroundo
    rgb_map = rgb_map + acc_map
    return rgb_map, weights


class NERF(nn.Module):
    """Nerf model implementation"""

    def __init__(self, hyper):
        """The architecture from the source code of the paper."""
        super().__init__()
        self.hyper = hyper

        self.act = nn.functional.relu
        layers = [nn.Linear(hyper["d_l_input"], hyper["d_filter"])]
        # lower layers
        for i in range(1, hyper["n_l_net"]):
            layers.append(
                nn.Linear(hyper["d_filter"] + hyper["d_l_input"], hyper["d_filter"])
                if i in hyper["l_skip"]
                else nn.Linear(hyper["d_filter"], hyper["d_filter"])
            )
        self.low_lays = nn.ModuleList(layers)

        # upper layers
        self.vox_unit = nn.Linear(hyper["d_filter"], 1)
        self.feat = nn.Linear(hyper["d_filter"], hyper["d_filter"])
        # assume view dir is 3d.
        layers = [
            nn.Linear(hyper["d_filter"] + hyper["d_u_input"], hyper["d_u_filter"])
        ]
        for i in range(1, hyper["n_u_net"]):
            layers.append(nn.Linear(hyper["d_u_filter"], hyper["d_u_filter"]))
        self.up_layers = nn.ModuleList(layers)

        self.rgb_unit = nn.Linear(hyper["d_u_filter"], 3)

    # NOTE : whole batch or single rays ? that is whole batch.
    def forward(self, x, view_dir):
        """Forward path.

        Parameters
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
            if i in self.hyper["l_skip"]:
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
            img = cv2.imread(fn)
            img = cv2.resize(img, (int(800 / scale_ratio), int(800 / scale_ratio)))
            imgs.append(img)
            poses.append(np.array(frame["transform_matrix"]))
            # DEBUG:  purpose
            break
            # DEBUG: purpose

        self.poses = tc.from_numpy(np.array(poses).astype(np.float32))
        self.imgs = (np.array(imgs) / 255.0).astype(np.float32)
        # white background
        # self.imgs = (255.0 * (1 - self.imgs[..., -1:])) + (
        #     self.imgs[..., :-1] * self.imgs[..., -1:]
        # )
        self.imgs = tc.from_numpy(self.imgs)
        self.H, self.W = imgs[0].shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)
        self.focal = self.focal / scale_ratio

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        # # TEST: test cuda.
        # temp_img = self.imgs[idx].to(device=dvc)
        # if dvc == "cuda" and not temp_img.is_cuda:
        #     raise Exception("Non cuda set")
        return self.imgs[idx], self.poses[idx].to(device=dvc)


def NERF_forward(pose, focal, H, W, coarse_model, fine_model):
    """Run a forward of the model with input as an image.
    Returns
    _______
    rgb_map : Tensor, expecting shape (H, W, 3), prediction image.
    """
    rays_o, rays_d = get_rays(pose, H, W, focal)  # (H, W, 3)
    pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_coarse_samples)
    # pts (H, W, n_sample, 3) , z_vals (H, W, n_sample)

    pts = pts.view(-1, pts.shape[-1])
    pts = positional_encoding(pts, L_pts)  # (W * H * n_sample, -1)

    temp_viewdir = rays_d / tc.norm(rays_d, dim=-1, keepdim=True)  # (H, W, 3)
    temp_viewdir = positional_encoding(temp_viewdir, L_viewdir)  # (H, W, ?)
    temp_viewdir = temp_viewdir[..., None, :]

    viewdir = temp_viewdir.expand(
        *temp_viewdir.shape[:-2], n_coarse_samples, temp_viewdir.shape[-1]
    )  # (W, H, n_sample, ?)
    viewdir = viewdir.reshape(-1, viewdir.shape[-1])  # (W * H * n_samples, -1)

    # NOTE: Should I encode the whole, or only a needed chunk. That is the whole .
    chunk_pts = prepare_chunks(pts, chunk_size)
    chunk_viewdir = prepare_chunks(viewdir, chunk_size)

    predictions = []
    for xs, viewdirs in zip(chunk_pts, chunk_viewdir):
        predictions.append(coarse_model(xs, viewdirs))
    raw = tc.cat(predictions, dim=0)  # (-1, 4)
    raw = raw.view(W, H, -1, raw.shape[-1])
    rgb_coarse, weights = toRGB(raw, z_vals, rays_d)  # (W, H, 3), (W, H, n_sample)

    # fine model.
    pts, z_vals_combined, new_z_vals = sample_herarchical(
        z_vals, weights[..., 1:-1], n_fine_samples, rays_o, rays_d
    )

    pts = pts.view(-1, pts.shape[-1])
    pts = positional_encoding(pts, L_pts)

    # viewdir1 = rays_d / tc.norm(rays_d, dim=-1, keepdim=True)
    # viewdir1 = positional_encoding(viewdir1, L_viewdir)
    # viewdir1 = viewdir1[..., None, :].expand(
    #     *viewdir1.shape[:-1], n_fine_samples + n_coarse_samples, viewdir1.shape[-1]
    # )
    # viewdir1 = viewdir1.reshape(-1, viewdir1.shape[-1])

    viewdir = temp_viewdir.expand(
        *temp_viewdir.shape[:-2],
        n_coarse_samples + n_fine_samples,
        temp_viewdir.shape[-1],
    )
    viewdir = viewdir.reshape(-1, viewdir.shape[-1])

    chunk_pts = prepare_chunks(pts, chunk_size)
    chunk_viewdir = prepare_chunks(viewdir, chunk_size)

    predictions = []
    for xs, viewdirs in zip(chunk_pts, chunk_viewdir):
        predictions.append(fine_model(xs, viewdirs))
    raw = tc.cat(predictions, dim=0)
    raw = raw.view(W, H, -1, raw.shape[-1])
    rgb_fine, weights = toRGB(raw, z_vals_combined, rays_d)

    return rgb_coarse, rgb_fine

    return


def train():
    # data
    dataset = BlenderDataset()

    # model
    coarse_model = NERF(c_hyperparameter).to(device=dvc)
    fine_model = NERF(f_hyperparameter).to(device=dvc)

    optimizer = tc.optim.Adam(
        list(coarse_model.parameters()) + list(fine_model.parameters()), lr=lr
    )

    # run train.
    coarse_model.train()
    fine_model.train()

    train_psnr = []
    for _ in range(n_iter):
        img_idx = np.random.randint(len(dataset))
        rgb_gt, pose = dataset[img_idx]
        rgb_coarse, rgb_fine = NERF_forward(
            pose, dataset.focal, dataset.H, dataset.W, coarse_model, fine_model
        )

        # Check for any numerical issues.
        for i, v in enumerate([rgb_coarse, rgb_fine]):
            if tc.isnan(v).any():
                print(f"! [Numerical Alert] {i} contains NaN.")
            if tc.isinf(v).any():
                print(f"! [Numerical Alert] {i} contains Inf.")

        # TEST
        print("check, delete")
        print(rgb_gt.shape)
        print(rgb_coarse.shape)
        print(rgb_fine.shape)
        loss = tc.nn.functional.mse_loss(rgb_coarse, rgb_gt)
        loss = loss + tc.nn.functional.mse_loss(rgb_fine, rgb_gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        psnr = -10 * tc.log10(loss)

        train_psnr.append(psnr.item())
        print(f"PSNR: {psnr}")

        if _ % dis_rate == 0:
            plt.imshow(rgb_fine.detach().cpu().numpy())
            plt.show()


if __name__ == "__main__":
    train()
