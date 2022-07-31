import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F


def img2mse(x, y): return torch.mean((x - y) ** 2)


def mse2psnr(x):
    device = x.get_device()
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(device)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def saveNumpyImage(img):
    img = np.array(img) * 255
    im = Image.fromarray(img.astype(np.uint8))
    im.save('./logs'+'/white_bkgd_false.jpg')


def make_o_d(img_w, img_h, img_k, pose):

    # make catesian (x. y)
    i, j = torch.meshgrid(torch.linspace(0, img_w - 1, img_w), torch.linspace(0, img_h - 1, img_h))
    i = i.t()
    j = j.t()

    dirs = torch.stack([(i - img_k[0][2])/img_k[0][0],
                        -(j - img_k[1][2])/img_k[1][1],
                        -torch.ones_like(i)], -1)

    rays_d = dirs @ pose[:3, :3].T
    rays_o = pose[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def sample_rays_and_pixel(rays_o, rays_d, target_img, opts):

    img_w, img_h = target_img.size()[:2]
    coords = torch.stack(torch.meshgrid(torch.linspace(0, img_h - 1, img_h), torch.linspace(0, img_w - 1, img_w)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])  # [ HxW , 2 ]

    # 640000 개 중 1024개 뽑기
    selected_idx = np.random.choice(a=coords.size(0), size=opts.batch_size, replace=False)  # default 1024
    selected_coords = coords[selected_idx].long()  # (N_rand, 2)

    # == Sample Rays ==
    rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
    rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]
    # == Sample Pixel ==
    target_img = target_img[selected_coords[:, 0], selected_coords[:, 1]]

    return rays_o, rays_d, target_img  # [1024, 3]


def batchify_rays_and_render_by_chunk(ray_o, ray_d, model, opts, fn_posenc, fn_posenc_d):
    flat_ray_o, flat_ray_d = ray_o.view(-1, 3), ray_d.view(-1, 3)  # [640000, 3], [640000, 3]

    num_whole_rays = flat_ray_o.size(0)
    rays = torch.cat((flat_ray_o, flat_ray_d), dim=-1)

    ret_coarse = []
    ret_fine = []

    for i in range(0, num_whole_rays, opts.chunk):
        rgb_dict = render_rays(rays[i:i+opts.chunk], model, fn_posenc, fn_posenc_d, opts)

        if opts.N_importance > 0:                    # use fine rays
            ret_coarse.append(rgb_dict['coarse'])
            ret_fine.append(rgb_dict['fine'])
        else:                                        # use only coarse rays
            ret_coarse.append(rgb_dict['ret_coarse'])

    if opts.N_importance > 0:
        return torch.cat(ret_coarse, dim=0), torch.cat(ret_fine, dim=0)
    else:
        return torch.cat(ret_coarse, dim=0), None


def render_rays(rays, model, fn_posenc, fn_posenc_d, opts):
    # 1. pre process : make (pts and dirs) (embedded)
    embedded, z_vals, rays_d = pre_process(rays, fn_posenc, fn_posenc_d, opts)

    # ** assign to cuda **
    embedded = embedded.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
    z_vals = z_vals.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
    rays_d = rays_d.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))

    # 2. run model by net_chunk
    chunk = opts.net_chunk
    outputs_flat = torch.cat([model(embedded[i:i+chunk]) for i in range(0, embedded.shape[0], chunk)], 0)  # [net_c
    size = [z_vals.size(0), z_vals.size(1), 4]      # [4096, 64, 4]
    outputs = outputs_flat.reshape(size)

    # 3. post process : render each pixel color by formula (3) in nerf paper
    rgb_map, disp_map, acc_map, weights, depth_map = post_process(outputs, z_vals, rays_d)

    if opts.N_importance > 0:
        # 4. pre precess
        rays = rays.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
        embedded_fine, z_vals_fine, rays_d = pre_process_for_hierarchical(rays, z_vals, weights, fn_posenc, fn_posenc_d, opts)

        # ** assign to cuda **
        # embedded_fine = embedded_fine.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
        # z_vals_fine = z_vals_fine.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
        # rays_d = rays_d.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))

        # 5. run model by net_chunk
        outputs_fine_flat = torch.cat([model(embedded_fine[i:i + chunk]) for i in range(0, embedded_fine.shape[0], chunk)], 0)
        size_fine = [z_vals_fine.size(0), z_vals_fine.size(1), 4]  # [4096, 64, 4]
        outputs_fine = outputs_fine_flat.reshape(size_fine)

        # 6. post process : render each pixel color by formula (3) in nerf paper
        rgb_map_fine, disp_map_fine, acc_map_fine, weights_fine, depth_map_fine = post_process(outputs_fine,
                                                                                               z_vals_fine, rays_d)

        return {'coarse': rgb_map, 'fine': rgb_map_fine}
    return {'coarse': rgb_map}


def pre_process(rays, fn_posenc, fn_posenc_d, opts):

    N_rays = rays.size(0)
    # assert N_rays == opts.chunk, 'N_rays must be same to chunk'
    rays_o, rays_d = rays[:, :3], rays[:, 3:]
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # make normal vector

    near = opts.near * torch.ones([N_rays, 1])
    far = opts.far * torch.ones([N_rays, 1])

    t_vals = torch.linspace(0., 1., steps=opts.N_samples)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, opts.N_samples])
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand([N_rays, opts.N_samples])
    z_vals = lower + (upper-lower) * t_rand

    # rays_o, rays_d : [B, 1, 3], z_vals : [B, 64, 1]
    input_pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    input_pts_flat = input_pts.view(-1, 3)                                # [1024/4096, 64, 3] -> [65536/262144, 3]
    input_pts_embedded = fn_posenc(input_pts_flat)                        # [n_pts, 63]

    input_dirs = viewdirs.unsqueeze(1).expand(input_pts.size())           # [4096, 3] -> [4096, 1, 3]-> [4096, 64, 3]
    input_dirs_flat = input_dirs.reshape(-1, 3)                           # [n_pts, 3]
    input_dirs_embedded = fn_posenc_d(input_dirs_flat)                    # [n_pts, 27]

    embedded = torch.cat([input_pts_embedded, input_dirs_embedded], -1)   # [n_pts, 90]

    # if opts.N_importance > 0:  # it means there are fine samples
    #
    #     z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    #     z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], opts.N_importance, det=(opts.perturb==0.))
    #     z_samples = z_samples.detach()
    #
    #     z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    #     pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [N_rays, N_samples + N_importance, 3]
    #
    #     t_vals_fine = torch.linspace(0., 1., steps=opts.N_importance)  # 128
    #     z_vals_fine = near * (1.-t_vals_fine) + far * (t_vals_fine)
    #     z_vals_fine = z_vals_fine.expand([N_rays, opts.N_importance])
    #     mids_fine = .5 * (z_vals_fine[..., 1:] + z_vals_fine[..., :-1])
    #     upper_fine = torch.cat([mids_fine, z_vals_fine[..., -1:]], -1)
    #     lower_fine = torch.cat([z_vals_fine[..., :1], mids_fine], -1)
    #     t_rand_fine = torch.rand([N_rays, opts.N_importance])
    #     z_vals_fine = lower_fine + (upper_fine - lower_fine) * t_rand_fine
    #
    #     input_pts_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals_fine.unsqueeze(-1)
    #     input_pts_fine_flat = input_pts_fine.view(-1, 3)  # [1024/4096, 64, 3] -> [65536/262144, 3]
    #     input_pts_embedded_fine = fn_posenc(input_pts_fine_flat)  # [n_pts, 63]
    #
    #     input_dirs_fine = viewdirs.unsqueeze(1).expand(input_pts_fine.size())  # [4096, 3] -> [4096, 1, 3]-> [4096, 64, 3]
    #     input_dirs_fine_flat = input_dirs_fine.reshape(-1, 3)  # [n_pts, 3]
    #     input_dirs_embedded_fine = fn_posenc_d(input_dirs_fine_flat)  # [n_pts, 27]
    #
    #     embedded_fine = torch.cat([input_pts_embedded_fine, input_dirs_embedded_fine], -1)  # [n_pts, 90]
    #
    #     return embedded, z_vals, rays_d, embedded_fine, z_vals_fine

    return embedded, z_vals, rays_d


def pre_process_for_hierarchical(rays, z_vals, weights, fn_posenc, fn_posenc_d, opts):

    rays_o, rays_d = rays[:, :3], rays[:, 3:]
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # make normal vector

    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], opts.N_importance, det=(opts.perturb == 0.))
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    # pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
    input_pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    input_pts_flat = input_pts.view(-1, 3)                                # [1024/4096, 64, 3] -> [65536/262144, 3]
    input_pts_embedded = fn_posenc(input_pts_flat)                        # [n_pts, 63]

    input_dirs = viewdirs.unsqueeze(1).expand(input_pts.size())           # [4096, 3] -> [4096, 1, 3]-> [4096, 64, 3]
    input_dirs_flat = input_dirs.reshape(-1, 3)                           # [n_pts, 3]
    input_dirs_embedded = fn_posenc_d(input_dirs_flat)                    # [n_pts, 27]

    embedded = torch.cat([input_pts_embedded, input_dirs_embedded], -1)   # [n_pts, 90]
    return embedded, z_vals, rays_d


def post_process(outputs, z_vals, rays_d):
    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - torch.exp(-act_fn(raw)*dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    device = dists.get_device()
    big_value = torch.Tensor([1e10]).to(device)
    dists = torch.cat([dists, big_value.expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples] 그런데 마지막에는 젤 큰 수 cat
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = outputs[..., :3]
    rgb_sigmoid = torch.sigmoid(rgb)

    alpha = raw2alpha(outputs[..., 3], dists)  # [N_rays, N_samples]

    # Density(alpha) X Transmittance
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb_sigmoid, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    # alpha to real color
    rgb_map = rgb_map + (1.-acc_map.unsqueeze(-1))
    return rgb_map, disp_map, acc_map, weights, depth_map


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    device = cdf.get_device()
    u = u.contiguous().to(device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])
    return samples