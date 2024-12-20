from functools import update_wrapper
from sys import exception
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
from torch.amp import custom_bwd, custom_fwd

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
		"n_hidden_layers": 1
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
		"n_hidden_layers": 2
	}
}"""


# sampling
n_samples = 1024
radius = 1.5
roi_aabb = tc.ones(6) * radius
roi_aabb[:3] = roi_aabb[:3] * -1


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


class _trunc_exp(tc.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=tc.float32, device_type="cuda")
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return tc.exp(x)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * tc.exp(x.clamp(-15, 15))


trunc_exp = _trunc_exp.apply


# TODO:  modify later
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
    alpha = 1.0 - tc.exp(-raw[..., 3] * dist)  # (H, W, n_sample)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)  # (H, W, n_sample)

    rgb = raw[..., :3]  # (H, W, n_sample, 3)
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
        encoding_config=cfg_density_mlp["encoding"],
        network_config=cfg_density_mlp["network"],
    ).to(device=dvc)
    viewdir_encoder = tcnn.Encoding(3, cfg_color_mlp["dir_encoding"]).to(device=dvc)
    color_mlp = tcnn.Network(32, 3, cfg_color_mlp["network"]).to(device=dvc)
    return density_mlp, color_mlp, viewdir_encoder


class HashNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.ne1, self.n2, self.e2 = init_model()

    def geometry(self, x):
        x = self.ne1(x)
        log_space_density = x[..., :1]
        return trunc_exp(log_space_density)

    def texture(self, x, viewdir):
        x = self.ne1(x)  # (batch_size, ..., 16)
        log_space_density = x[..., :1]  # (batch_size,..., 1)
        viewdir = self.e2(viewdir)  # (batch_size,..., 16)
        viewdir = tc.cat((viewdir, x), dim=-1)
        viewdir = self.n2(viewdir)  # (batch_size,..., 3)

        # apply activation function.
        viewdir = tc.sigmoid(viewdir)
        log_space_density = trunc_exp(log_space_density)

        return viewdir, log_space_density

    def forward(self, x, viewdir):
        x = self.ne1(x)  # (batch_size, ..., 16)
        log_space_density = x[..., :1]  # (batch_size,..., 1)
        viewdir = self.e2(viewdir)  # (batch_size,..., 16)
        viewdir = tc.cat((viewdir, x), dim=-1)
        viewdir = self.n2(viewdir)  # (batch_size,..., 3)

        # apply activation function.
        viewdir = tc.sigmoid(viewdir)
        log_space_density = trunc_exp(log_space_density)

        return tc.cat((viewdir, log_space_density), dim=-1)

    def params(self):
        return (
            list(self.ne1.parameters())
            + list(self.n2.parameters())
            + list(self.e2.parameters())
        )


def NERF_forward(pose, focal, H, W, model, estimator):
    """Run a forward of the model with input as an image.
    Returns
    _______
    rgb_map : Tensor, expecting shape (H, W, 3), prediction image.
    """
    rays_o, rays_d = get_rays(pose, H, W, focal)  # (H, W, 3)
    rays_o = rays_o.reshape(-1, rays_o.shape[-1])  # (H * W, 3)
    rays_d = rays_d.reshape(-1, rays_d.shape[-1])  # (H * W, 3)
    rays_d = rays_d / tc.norm(rays_d, dim=-1, keepdim=True)

    def sigma_fn(t_starts, t_ends, ray_indices):
        # t_starts, t_ends (n_samples, 1)
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]  # (n_sample, 3)
        t_dirs = rays_d[ray_indices]  # (n_sample, 3)
        # TEST:
        if (t_starts + t_ends).ndims != 1:
            raise Exception("dims error")
        positions = (
            t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        )  # (n_samples, 3)
        density = model.geometry(positions)
        return density  # (n_samples, 1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]  # (n_samples, 3)
        # TEST:
        if (t_starts + t_ends).ndims != 1:
            raise Exception("dims error")
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        rgb, density = model.texture(positions, t_dirs)
        return rgb, density

    with tc.no_grad():
        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn,
            near_plane=0.2,
            far_plane=1e4,
            render_step_size=1.732 * 2 * radius / n_samples,
            cone_angle=0.0,
            early_stop_eps=1e-4,
            alpha_thre=0.0,
        )
    # if overflow, it need to be seperated.
    # chunk_pts = prepare_chunks(pts, chunk_size)
    # chunk_viewdir = prepare_chunks(t_dirs, chunk_size)

    # NOTE: can be optimized.
    colors, opacities, depths, extras = nerfacc.rendering(
        t_starts, t_ends, ray_indices, n_rays=rays_o.shape[0], rgb_sigma_fn=rgb_sigma_fn
    )
    return colors, opacities, depths, extras


def train(dataset, model, optimizer, estimator):
    # run train.
    model.train()

    train_psnr = []
    step = 0
    for _ in trange(n_iter):
        img_idx = np.random.randint(len(dataset))
        rgb_gt, pose = dataset[img_idx]
        rgb_gt = rgb_gt.to(device=dvc)
        rgb_gt.view(-1, rgb_gt.shape[-1])
        colors, opacities, depths, extras = NERF_forward(
            pose, dataset.focal, dataset.H, dataset.W, model, estimator
        )

        # TEST: Check for any numerical issues.
        # if tc.isnan(rgb_prediction).any():
        #     print("! [Numerical Alert] contains NaN.")
        # if tc.isinf(rgb_prediction).any():
        #     print("! [Numerical Alert] contains Inf.")

        loss = tc.nn.functional.mse_loss(colors, rgb_gt)

        loss.backward()
        optimizer.step()

        def occ_eval_fn(x):
            density = model.geometry(x)
            return density * (1.732 * 2 * radius / n_samples)

        estimator.update_every_n_steps(step, occ_eval_fn)
        step = step + 1
        optimizer.zero_grad()
        psnr = -10 * tc.log10(loss)

        train_psnr.append(psnr.item())
        print(f"PSNR: {psnr}")

        if _ % dis_rate == 0:
            # add collab cv imshow
            cv2_imshow(colors.detach().cpu().numpy() * 255)


if __name__ == "__main__":
    # data
    data = BlenderDataset()
    # model
    model = HashNeRF()
    # optimizer
    optimizer = tc.optim.Adam(model.params(), lr, (beta1, beta2), eps)

    estimator = nerfacc.OccGridEstimator(roi_aabb, 128, 1)

    train(data, model, optimizer, estimator)
