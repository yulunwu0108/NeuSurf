import torch
import torch.nn.functional as F
import numpy as np
import mcubes
import tools.feat_utils as feat_utils


# interpolate SDF zero-crossing points
def find_surface_points(sdf, d_all, device='cuda'):
    # shape of sdf and d_all: only inside
    sdf_bool_1 = sdf[...,1:] * sdf[...,:-1] < 0
    # only find backward facing surface points, not forward facing
    sdf_bool_2 = sdf[...,1:] < sdf[...,:-1]
    sdf_bool = torch.logical_and(sdf_bool_1, sdf_bool_2)

    max, max_indices = torch.max(sdf_bool, dim=2)
    network_mask = max > 0
    d_surface = torch.zeros_like(network_mask, device=device).float()

    sdf_0 = torch.gather(sdf[network_mask], 1, max_indices[network_mask][..., None]).squeeze()
    sdf_1 = torch.gather(sdf[network_mask], 1, max_indices[network_mask][..., None]+1).squeeze()
    d_0 = torch.gather(d_all[network_mask], 1, max_indices[network_mask][..., None]).squeeze()
    d_1 = torch.gather(d_all[network_mask], 1, max_indices[network_mask][..., None]+1).squeeze()
    d_surface[network_mask] = (sdf_0 * d_1 - sdf_1 * d_0) / (sdf_0-sdf_1)

    return d_surface, network_mask

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 dataset,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.dataset = dataset
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

        self.feat_ext = feat_utils.FeatExt().cuda()
        self.feat_ext.eval()
        for p in self.feat_ext.parameters():
            p.requires_grad = False

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5


        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
            'mid_z_vals_out' : mid_z_vals
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_color(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    ):

        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        # inv_s in the code == s in the paper == 1 / standard deviation
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        return color

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    model_input = None,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    mid_z_vals_out=None,
                    cos_anneal_ratio=0.0,
                    depth_from_inside_only=None,
                    ):

        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        query_pts = pts.clone()

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        # inv_s in the code == s in the paper == 1 / standard deviation
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        alpha_in = alpha

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        if depth_from_inside_only:
            weights_in = alpha_in * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha_in + 1e-7], -1), -1)[:, :-1]
            weights_in_sum = weights_in.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        if model_input is not None:
            if background_alpha is not None:
                z_final = mid_z_vals_out
            else:
                z_final = mid_z_vals
            if depth_from_inside_only:
                z_final = mid_z_vals

            if depth_from_inside_only:
                dist_map = torch.sum(weights_in / (weights_in.sum(-1, keepdim=True)+1e-10) * z_final, -1)
            else:
                dist_map = torch.sum(weights / (weights.sum(-1, keepdim=True)+1e-10) * z_final, -1)

            sdf_all =  sdf.reshape(batch_size,n_samples).unsqueeze(0)
            d_all = mid_z_vals.unsqueeze(0)
            d_surface, network_mask = find_surface_points(sdf_all, d_all)
            d_surface = d_surface.squeeze(0)
            network_mask = network_mask.squeeze(0)

            object_mask = network_mask

            point_surface = rays_o + rays_d * d_surface[:,None]
            point_surface_wmask = point_surface[network_mask & object_mask]

            points_rendered = rays_o + rays_d * dist_map[:,None]
            sdf_rendered_points = sdf_network(points_rendered)[:, :1]
            sdf_rendered_points_wmask = sdf_rendered_points[object_mask]
            sdf_rendered_points_0 = torch.zeros_like(sdf_rendered_points_wmask)
            pseudo_pts_loss = F.l1_loss(sdf_rendered_points_wmask, sdf_rendered_points_0, reduction='mean')

            return {
                'color': color,
                'sdf': sdf,
                'dists': dists,
                'gradients': gradients.reshape(batch_size, n_samples, 3),
                's_val': 1.0 / inv_s,
                'mid_z_vals': mid_z_vals,
                'weights': weights,
                'cdf': c.reshape(batch_size, n_samples),
                'gradient_error': gradient_error,
                'inside_sphere': inside_sphere,
                'pseudo_pts_loss': pseudo_pts_loss,
                'query_pts': query_pts,
                'point_surface': point_surface_wmask,
                'network_mask': network_mask,
                'object_mask': object_mask,
            }
        else:
            return {
                'color': color,
                'sdf': sdf,
                'dists': dists,
                'gradients': gradients.reshape(batch_size, n_samples, 3),
                's_val': 1.0 / inv_s,
                'mid_z_vals': mid_z_vals,
                'weights': weights,
                'cdf': c.reshape(batch_size, n_samples),
                'gradient_error': gradient_error,
                'inside_sphere': inside_sphere,
                'pseudo_pts_loss': torch.tensor(0.0).float(),
                'query_pts': query_pts,
            }

    def render(self,
               rays_o,
               rays_d,
               near,
               far,
               main_img_idx,
               t,
               random_pcd=None,
               perturb_overwrite=-1,
               background_rgb=None,
               cos_anneal_ratio=0.0,
               model_input=None,
               depth_from_inside_only=False,
               ):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None
        mid_z_vals_out = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)

            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']
            mid_z_vals_out= ret_outside['mid_z_vals_out']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    model_input = model_input,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    mid_z_vals_out= mid_z_vals_out,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    depth_from_inside_only=depth_from_inside_only)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        local_loss = torch.tensor(0).float()
        if model_input is not None:
            output = {
                'color_fine': color_fine,
                's_val': s_val,
                'cdf_fine': ret_fine['cdf'],
                'weight_sum': weights_sum,
                'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
                'gradients': gradients,
                'weights': weights,
                'gradient_error': ret_fine['gradient_error'],
                'inside_sphere': ret_fine['inside_sphere'],
                'pseudo_pts_loss': ret_fine['pseudo_pts_loss'],
                'sdf': ret_fine['sdf'],
                'query_pts': ret_fine['query_pts'],
            }

            point_surface_wmask = ret_fine['point_surface']
            network_mask = ret_fine['network_mask']
            object_mask = ret_fine['object_mask']

            size, center = model_input['size'].unsqueeze(0), model_input['center'].unsqueeze(0)
            size = size[:1]
            center = center[:1]

            cam = model_input['cam'] # 2, 4, 4
            src_cams = model_input['src_cams'] # m, 2, 4, 4
            feat_src = model_input['feat_src']

            if (t % 100 == 0) and (random_pcd is not None):
                ''' unseen view rendering '''
                random_pcd = random_pcd.view(1, -1, 3)
                random_pcd.requires_grad = False

                src_img_idx = model_input['src_idxs'][0]
                rays_o, rays_d = self.dataset.gen_rays_between_from_pts(main_img_idx,
                                                                src_img_idx,
                                                                0.5,
                                                                random_pcd,
                                                                )

                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                batch_size = len(rays_o)
                sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
                z_vals = torch.linspace(0.0, 1.0, self.n_samples)
                z_vals = near + (far - near) * z_vals[None, :]

                z_vals_outside = None
                if self.n_outside > 0:
                    z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

                n_samples = self.n_samples
                perturb = self.perturb

                if perturb_overwrite >= 0:
                    perturb = perturb_overwrite
                if perturb > 0:
                    t_rand = (torch.rand([batch_size, 1]) - 0.5)
                    z_vals = z_vals + t_rand * 2.0 / self.n_samples

                    if self.n_outside > 0:
                        mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                        upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                        lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                        t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                        z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

                if self.n_outside > 0:
                    z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

                background_alpha = None
                background_sampled_color = None
                mid_z_vals_out = None

                # Up sample
                if self.n_importance > 0:
                    with torch.no_grad():
                        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                        sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                        for i in range(self.up_sample_steps):
                            new_z_vals = self.up_sample(rays_o,
                                                        rays_d,
                                                        z_vals,
                                                        sdf,
                                                        self.n_importance // self.up_sample_steps,
                                                        64 * 2**i)
                            z_vals, sdf = self.cat_z_vals(rays_o,
                                                        rays_d,
                                                        z_vals,
                                                        new_z_vals,
                                                        sdf,
                                                        last=(i + 1 == self.up_sample_steps))

                    n_samples = self.n_samples + self.n_importance

                # Background model
                if self.n_outside > 0:
                    z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
                    z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)

                    ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

                    background_sampled_color = ret_outside['sampled_color']
                    background_alpha = ret_outside['alpha']
                    mid_z_vals_out= ret_outside['mid_z_vals_out']

                color = self.render_color(rays_o,
                                            rays_d,
                                            z_vals,
                                            sample_dist,
                                            self.sdf_network,
                                            self.deviation_network,
                                            self.color_network,
                                            background_rgb=background_rgb,
                                            background_alpha=background_alpha,
                                            background_sampled_color=background_sampled_color,
                                            cos_anneal_ratio=cos_anneal_ratio,
                                            )

                us_pose = cam.clone()
                us_pose[0] = feat_utils.gen_camera_between(cam[0].cpu().numpy(), src_cams[0, 0].cpu().numpy(), 0.5)
                us_pose.requires_grad = False
                us_pose = us_pose.unsqueeze(0) # 2, 4, 4

                us_rgb = torch.zeros([1, 3, 768, 1024]).cuda()

                pts_world = random_pcd.view(1, -1, 1, 3, 1)
                pts_world = torch.cat([pts_world, torch.ones_like(pts_world[..., -1:, :])], dim=-2)
                pts_img = feat_utils.idx_cam2img(feat_utils.idx_world2cam(pts_world, us_pose), us_pose).view(1, -1, 3) # 1, N, 3
                us_uv = pts_img[..., :2] / pts_img[..., 2:3]
                us_uv = us_uv.round().long()

                color_mask = ((us_uv[..., 0] > -1) & (us_uv[..., 0] < 1024) & (us_uv[..., 1] > -1) & (us_uv[..., 1] < 768)).squeeze(0)

                us_uv = us_uv[0, color_mask] # M, 2
                color = color[color_mask] # M, 3

                _, cnts = torch.unique(us_uv, sorted=False, return_counts=True, dim=0)
                cnts = torch.cat((torch.tensor([0]).long().cuda(), cnts))
                unique_index = torch.cumsum(cnts, dim=0)
                unique_index = unique_index[:-1]

                us_uv = us_uv[unique_index]
                color = color[unique_index].transpose(0, 1)

                us_rgb[0, :, us_uv[:, 1], us_uv[:, 0]] = color

                us_feat = self.feat_ext(us_rgb)[2]

                local_loss += feat_utils.get_local_loss(random_pcd.reshape(-1, 3), None, us_feat,
                                                      us_pose, feat_src.unsqueeze(0), src_cams.unsqueeze(0),
                                                      2 * torch.ones_like(size).cuda(), torch.zeros_like(center).cuda(),
                                                      color_mask.reshape(-1), color_mask.reshape(-1))

            local_loss += feat_utils.get_local_loss(point_surface_wmask, None, model_input['feat'].unsqueeze(0),
                                        cam.unsqueeze(0), feat_src.unsqueeze(0), src_cams.unsqueeze(0),
                                        size, center, network_mask.reshape(-1),
                                        object_mask.reshape(-1))

            output['local_loss'] = local_loss
            return output

        else:
            return {
                'color_fine': color_fine,
                's_val': s_val,
                'cdf_fine': ret_fine['cdf'],
                'weight_sum': weights_sum,
                'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
                'gradients': gradients,
                'weights': weights,
                'gradient_error': ret_fine['gradient_error'],
                'inside_sphere': ret_fine['inside_sphere'],
                'pseudo_pts_loss': ret_fine['pseudo_pts_loss'],
                'local_loss': local_loss,
                'sdf': ret_fine['sdf'],
                'query_pts': ret_fine['query_pts'],
            }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
