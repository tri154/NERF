import torch
import json
import os
import cv2
import tinycuda as tcnn
from torch import nn
import numpy as np
from torch.amp import custom_bwd, custom_fwd
from torch.utils.data import DataLoader, dataloader

ROOT_DIR = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#training
chunk_size = 2**14

#opitmizer
lr = 1e-4
betas = (0.9, 0.99)
eps = 1e-15

#sampling
near = 2
far = 6
n_samples = 64

bound = 1

#rendering
density_scale = 1 # test if higher than 1

class _trunc_exp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _trunc_exp.apply


class NeRFNetwork(nn.Module):
    def __init__(
        self,
        encoding="HashGrid",
        encoding_dir="SphericalHarmonics",
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        bound=1,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        # p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return sigma, geo_feat

        
    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params


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

    def __init__(self, scale_ratio=1):
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

        self.poses = torch.from_numpy(np.array(poses).astype(np.float32))
        self.poses = self.poses.to(device)

        self.imgs = (np.array(imgs) / 255.0).astype(np.float32)
        self.imgs = torch.from_numpy(self.imgs)
        self.imgs = self.imgs.to(device)

        self.H, self.W = imgs[0].shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)
        self.focal = self.focal / scale_ratio

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx], self.poses[idx]

    def collate(self, index):
        """ 
        Parameters:
        ___________

        index : list of indexs in image. Default: batch_size=1, only a single index.
        """
        pose = self.poses[index]

        i, j = torch.meshgrid(
                torch.arange(self.W, dtype=torch.float32).to(pose),
                torch.arange(self.H, dtype=torch.float32).to(pose),
                indexing='ij')
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        directions = torch.stack([(i - self.W * .5) / self.focal,
                                    -(j - self.H * .5) / self.focal,
                                    -torch.ones_like(i)
                                ], dim=-1)

        # Apply camera pose to directions
        rays_d = torch.sum(directions[..., None, :] * pose[:3, :3], dim=-1)

        # Origin is same for all directions (the optical center)
        rays_o = pose[:3, -1].expand(rays_d.shape)
        
        res = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays_o,
            'rays_d': rays_d
        }
        return res

    def dataloader(self):
        size = len(self.imgs)
        train_loader = DataLoader(list(range(size)), batch_size=1, shuffle=True, collate_fn=self.collate)
        return train_loader

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
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)
    z_vals = near * (1 - t_vals) + far * t_vals  # n_samples
    z_vals = z_vals.expand(*rays_d.shape[:-1], z_vals.shape[-1])  # [H, W, n_samples]
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
    # res = [input[i : i + chunks_size] for i in (0, input.shape[0], chunks_size)]
    res = []
    for i in range(0, input.shape[0], chunks_size):
        res.append(input[i : i + chunks_size])
    return res


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def train(model, train_loader, optimizer):
    aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound]).to(device)

    for data in train_loader:
        rays_o, rays_d = data['rays_o'], data['rays_d']
        rays_o = rays_o.reshape(-1, rays_o.shape[-1])
        rays_d = rays_d.reshape(-1, rays_d.shape[-1])

        pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples) # already flatten out.

        pts = torch.min(torch.max(pts, aabb[:3]), aabb[3:])

        pts_chunk = prepare_chunks(pts, chunk_size)

        sigma_pred = []
        geo_feat_pred = []
        for chunk in pts_chunk:
            sigma, geo_feat = model.density(chunk)
            sigma_pred.append(sigma)
            geo_feat_pred.append(geo_feat)

        sigmas = torch.cat(sigma_pred, dim=0)
        geo_feats = torch.cat(geo_feat_pred, dim=0)

        #need to do that ?.
        # sigmas = sigmas.reshape(*z_vals.shape[:2], -1) # Restore shape (H, W, n_samples).
        # geo_feats = geo_feats.reshape(*z_vals.shape[:2], -1, geo_feats.shape[-1]) # Restore shape (H, W, n_samples, 3).
        
        with torch.no_grad():
            deltas = z_vals[..., 1:] - z_vals[..., :-1]  # (H, W, n_samples - 1)
            sample_dist = (far - near) / n_samples
            deltas = torch.cat( (deltas, sample_dist * torch.ones_like(deltas[..., :1])) ,dim=-1) # (H, W, n_samples).

            alphas = 1 - torch.exp(-deltas * density_scale *  sigmas) #(H, W, n_samples)
            alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [H, W, n_samples + 1]
            weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # (H, W, n_samples).

            z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [H, W, n_samples - 1]
            new_z_vals = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples).detach()
            
            new_pts =  rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) #already flatten outs.
            new_pts = torch.min(torch.max(new_pts, aabb[:3]), aabb[3:]) 

        
        new_pts_chunk = prepare_chunks(new_pts, chunk_size)
        # what if i flatten out the z_vals, pts .................. J.

        new_sigma_pred = []
        new_geo_feat_pred = []
        for chunk in new_pts_chunk:
            sigma, geo_feat = model.density(chunk)
            new_sigma_pred.append(sigma)
            new_geo_feat_pred.append(geo_feat)
        new_sigmas = torch.cat(new_sigma_pred, dim=0) 
        new_geo_feats = torch.cat(new_geo_feat_pred, dim=0)

        # new_sigmas = new_sigmas.reshape(*new_z_vals.shape[:2], -1)
        # new_geo_feats = new_geo_feats.reshape(*new_z_vals.shape[:2], -1, new_geo_feats.shape[-1])

        # I suddenly felt so bad, I've been sucked. I want to finish that shiet.
        # I hate myself so bad @_@.
        # What I've done ? nothing, feel so bad.
        # I want to say that thing, but I can't.
        # I also want to take a shower, I have to finish that.
        # should I keep this one ?, I forget to write uppercase in this sentence. is my grammar correct ? 
        
        z_vals = torch.cat( (z_vals, new_z_vals), dim=-1 ) # (H * W, n_samples + n_samples)
        z_vals, z_indices = torch.sort(z_vals, dim=-1) # (H * W, n_samples + n_samples)
        
        #Use for what ?? ??????.
        pts = torch.cat( (pts, new_pts), dim=-2) # (H, W, n_samples + n_samples, 3)
        pts = torch.gather(pts, dim=-2, index=z_indices.unsqueeze(-1).expand_as(pts))
        
        sigmas = torch.cat((sigmas, new_sigmas), dim=-1)
        sigmas = torch.gather(sigmas, dim=-1, index=z_indices)

        geo_feats = torch.cat((geo_feats, new_geo_feats), dim=-2)
        geo_feats = torch.gather(geo_feats, dim=-2, index=z_indices.unsqueeze(-1).expand_as(geo_feats))

        #Do I need weights ???????? Test no need that, im in deppression anayawy.
        


if __name__ == '__main__':
    dataset = BlenderDataset()
    dataloader = dataset.dataloader()

    model = NeRFNetwork()

    optimizer = 

    
