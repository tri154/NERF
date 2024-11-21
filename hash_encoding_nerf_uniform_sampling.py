import tinycudann as tcnn

import numpy as np
from torch import nn
import torch as tc
import nerfacc

import json
import os
import cv2

from tqdm import trange
from google.colab.patches import cv2_imshow

dvc = tc.device("cuda" if tc.cuda.is_available() else "cpu")

ROOT_DIR = "/content/drive/Othercomputers/My Laptop/Subjects/2.Introduction to Cognitive Intelligence/project"

# CONFIG
# model
config_density_mlp = """{
  	"encoding": {
		"otype": "HashGrid",
		"n_levels": 16,
		"n_features_per_level": 2,
		"log2_hashmap_size": 15,
		"base_resolution": 16,
		"per_level_scale": 1.5
	},
	"network": {
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 64,
		"n_hidden_layers": 2
	}
}"""

config_color_mlp = """{
	"dir_encoding": {
		"otype": "Composite",
		"nested": [
			{
				"n_dims_to_encode": 3,
				"otype": "SphericalHarmonics",
				"degree": 4
			},
			{
				"otype": "Identity"
			}
		]
	},
	"network": {
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 64,
		"n_hidden_layers": 1
	}
}"""


# sampling
n_samples = 64
near = 2
far = 6


# train
n_iter = 10000
chunk_size = 2**18
lr = 1e-4
beta1 = 0.9
beta2 = 0.99
eps = 1e-15
dis_rate = 25


# Blender data
scale_ratio = 4
# END CONFIG


def load_model(coarse_model, fine_model):
    coarse_model.load_state_dict(
        tc.load(os.path.join(ROOT_DIR, "coarse_model.pt"), weights_only=True)
    )
    coarse_model.to(device=dvc)
    fine_model.load_state_dict(
        tc.load(os.path.join(ROOT_DIR, "fine_model.pt"), weights_only=True)
    )
    fine_model.to(device=dvc)


def save_model(coarse_model, fine_model):
    tc.save(coarse_model.state_dict(), os.path.join(ROOT_DIR, "coarse_model.pt"))
    tc.save(fine_model.state_dict(), os.path.join(ROOT_DIR, "fine_model.pt"))


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
    # direction shape (H, W, 3)
    rays_d = tc.sum(pose[:3, :3] * direction[..., None, :], dim=-1)
    rays_d = tc.transpose(rays_d, dim0=0, dim1=1)
    rays_o = pose[:-1, -1].expand(rays_d.shape)  # NOTE : redundant.

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


def prepare_chunks(input, chunks_size):
    """Split to chunks
    Parameters
    __________
    input : Tensor, expecting shape (W * H * n_sample, n_feature)


    Returns
    _______

    list: list of chunks.
    """
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
    return rgb_map


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
            if scale_ratio != 1:
                img = cv2.resize(img, (int(800 / scale_ratio), int(800 / scale_ratio)))
            imgs.append(img)
            poses.append(np.array(frame["transform_matrix"]))
            # DEBUG:  purpose
            # break
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
        return self.imgs[idx], self.poses[idx].to(device=dvc)


def init_model():
    cfg_density_mlp = json.loads(config_density_mlp)
    cfg_color_mlp = json.loads(config_color_mlp)
    density_mlp = tcnn.NetworkWithInputEncoding(
        3,
        16,
        endcoding_config=cfg_density_mlp["encoding"],
        network_config=cfg_density_mlp["network"],
    ).to(device=dvc)
    viewdir_encoder = tcnn.Encoding(3, cfg_color_mlp["dir_encoding"]).to(device=dvc)
    color_mlp = tcnn.Network(32, 3, cfg_color_mlp["network"]).to(device=dvc)
    return density_mlp, color_mlp, viewdir_encoder


class HashNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne1, self.n2, self.e2 = init_model()

    def forward(self, x, viewdir):
        x = self.ne1(x)  # (batch_size, ..., 16)
        log_space_density = x[..., :1]  # (batch_size,..., 1)
        viewdir = self.e2(viewdir)  # (batch_size,..., 16)
        viewdir = tc.cat((viewdir, x), dim=-1)
        viewdir = self.n2(viewdir)  # (batch_size,..., 3)
        return tc.cat((viewdir, tc.exp(log_space_density)), dim=-1)

    def params(self):
        return (
            list(self.ne1.parameters())
            + list(self.n2.parameters())
            + list(self.e2.parameters())
        )


def NERF_forward(pose, focal, H, W, model):
    """Run a forward of the model with input as an image.
    Returns
    _______
    rgb_map : Tensor, expecting shape (H, W, 3), prediction image.
    """
    rays_o, rays_d = get_rays(pose, H, W, focal)  # (H, W, 3)
    rays_o = rays_o.reshape(-1, rays_o.shape[-1])
    rays_d = rays_d.reshape(-1, rays_d.shape[-1])
    pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples)
    # pts (H, W, n_sample, 3) , z_vals (H, W, n_sample)
    pts = pts.reshape(-1, pts.shape[-1])

    viewdir = rays_d / tc.norm(rays_d, dim=-1, keepdim=True)  # (H, W, 3)
    viewdir = viewdir[..., None, :]
    viewdir = viewdir.expand(
        *viewdir.shape[:-2], n_samples, viewdir.shape[-1]
    )  # (W, H, n_sample, ?)
    viewdir = viewdir.reshape(-1, viewdir.shape[-1])  # (W * H * n_samples, -1)

    # NOTE: Should I encode the whole, or only a needed chunk. That is only the batch .
    chunk_pts = prepare_chunks(pts, chunk_size)
    chunk_viewdir = prepare_chunks(viewdir, chunk_size)

    predictions = []
    for xs, viewdirs in zip(chunk_pts, chunk_viewdir):
        predictions.append(model(xs, viewdirs))
    raw = tc.cat(predictions, dim=0)  # (-1, 4)
    raw = raw.view(H, W, -1, raw.shape[-1])
    rgb_predition = toRGB(raw, z_vals, rays_d)  # (W, H, 3), (W, H, n_sample)

    return rgb_predition


def train(dataset, model, optimizer):
    # run train.
    model.train()

    train_psnr = []
    for _ in trange(n_iter):
        img_idx = np.random.randint(len(dataset))
        rgb_gt, pose = dataset[img_idx]
        rgb_gt = rgb_gt.to(device=dvc)
        rgb_prediction = NERF_forward(pose, dataset.focal, dataset.H, dataset.W, model)

        # TEST: Check for any numerical issues.
        if tc.isnan(rgb_prediction).any():
            print("! [Numerical Alert] contains NaN.")
        if tc.isinf(rgb_prediction).any():
            print("! [Numerical Alert] contains Inf.")

        loss = tc.nn.functional.mse_loss(rgb_prediction, rgb_gt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        psnr = -10 * tc.log10(loss)

        train_psnr.append(psnr.item())
        print(f"PSNR: {psnr}")

        if _ % dis_rate == 0:
            # add collab cv imshow
            cv2_imshow(rgb_prediction.detach().cpu().numpy() * 255)


if __name__ == "__main__":
    # data
    data = BlenderDataset()
    # model
    model = HashNeRF()
    # optimizer
    optimizer = tc.optim.Adam(model.params(), lr, (beta1, beta2), eps)

    radius = 1.5
    roi_aabb = tc.ones(6) * radius
    roi_aabb[:3] = roi_aabb[:3] * -1
    nerfacc.OccGridEstimator(roi_aabb, 128, 1)

    train(data, model, optimizer)
