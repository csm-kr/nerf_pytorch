import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from IQA_pytorch import SSIM, LPIPSvgg, DISTS


def img2mse(x, y): return torch.mean((x - y) ** 2)


def mse2psnr(x):
    device = x.get_device()
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(device)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def saveNumpyImage(img):
    img = np.array(img) * 255
    im = Image.fromarray(img.astype(np.uint8))
    im.save('./logs'+'/white_bkgd_false.jpg')


def getSSIM(pred, gt):
    SSIM_ = SSIM(channels=3)
    # [W,H,3]->[1,3,W,H]
    return SSIM_(pred.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0), as_loss=False)


def getLPIPS(pred, gt):
    device = pred.get_device()
    LPIPS_ = LPIPSvgg(channels=3).to(device)
    LPIPS_.weights = [(t1, t2.to(device)) for (t1, t2) in LPIPS_.weights]
    # loss_lpips = LLPIS_(rgb.permute(2, 0, 1).unsqueeze(0), test_imgs[i].permute(2, 0, 1).unsqueeze(0))
    # LPIPS_ = lpips.LPIPS(net='vgg')
    return LPIPS_(pred.permute(2, 0, 1), gt.permute(2, 0, 1))


def make_o_d(img_w, img_h, img_k, pose):

    # make catesian (x. y)
    i, j = torch.meshgrid(torch.linspace(0, img_w - 1, img_w, device=pose.get_device()),
                          torch.linspace(0, img_h - 1, img_h, device=pose.get_device()))
    i = i.t()
    j = j.t()

    dirs = torch.stack([(i - img_k[0][2])/img_k[0][0],
                        -(j - img_k[1][2])/img_k[1][1],
                        -torch.ones_like(i)], -1)

    rays_d = dirs @ pose[:3, :3].T
    rays_o = pose[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def sample_rays_and_pixels(rays_o, rays_d, target_img, opts):

    img_w, img_h = target_img.size()[:2]
    coords = torch.stack(torch.meshgrid(torch.linspace(0, img_h - 1, img_h), torch.linspace(0, img_w - 1, img_w)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])  # [ HxW , 2 ]

    # 640000 개 중 1024개 뽑기
    selected_idx = np.random.choice(a=coords.size(0), size=opts.batch_size, replace=False)  # default 1024
    selected_coords = coords[selected_idx].long()  # (N_rand, 2)

    # == Sample Rays ==
    rays_o = rays_o[selected_coords[:, 1], selected_coords[:, 0]]
    rays_d = rays_d[selected_coords[:, 1], selected_coords[:, 0]]
    # == Sample Pixel ==
    target_img = target_img[selected_coords[:, 1], selected_coords[:, 0]]

    return rays_o, rays_d, target_img  # [1024, 3]

# 점찍기
def visualize_points_and_rays(flat_ray_o, flat_ray_d):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(flat_ray_o[:, 0],
               flat_ray_o[:, 1],
               flat_ray_o[:, 2])
    ax.scatter(flat_ray_o[:, 0] + flat_ray_d[:, 0],
               flat_ray_o[:, 1] + flat_ray_d[:, 1],
               flat_ray_o[:, 2] + flat_ray_d[:, 2])
    num_points = flat_ray_o.size(0)
    for i in range(num_points):
        ax.plot(
            [flat_ray_o[i, 0], flat_ray_o[i, 0] + flat_ray_d[i, 0]],
            [flat_ray_o[i, 1], flat_ray_o[i, 1] + flat_ray_d[i, 1]],
            [flat_ray_o[i, 2], flat_ray_o[i, 2] + flat_ray_d[i, 2]]
        )
    plt.show()


def render_by_chunk(ray_o, ray_d, model, fn_posenc, fn_posenc_d, H, W, K, opts):
    flat_ray_o, flat_ray_d = ray_o.view(-1, 3), ray_d.view(-1, 3)  # train : [batch_size, 3] / test :  [640000, 3]
    if opts.vis_points_rays:
        visualize_points_and_rays(flat_ray_o, flat_ray_d)

    # FIXME only llff dataset
    if opts.data_type == 'llff':
        flat_ray_o, flat_ray_d = ndc_rays(H, W, K[0][0], 1., flat_ray_o, flat_ray_d)
        if opts.vis_points_rays:
            visualize_points_and_rays(flat_ray_o, flat_ray_d)

    num_whole_rays = flat_ray_o.size(0)
    rays = torch.cat((flat_ray_o, flat_ray_d), dim=-1)

    ret_coarse = []
    ret_coarse_disp = []
    ret_fine = []
    ret_fine_disp = []

    for i in range(0, num_whole_rays, opts.chunk):
        rgb_dict = render_rays(rays[i:i+opts.chunk], model, fn_posenc, fn_posenc_d, opts)

        if opts.N_importance > 0:                    # use fine rays
            ret_coarse.append(rgb_dict['coarse'])
            ret_coarse_disp.append(rgb_dict['disp_coarse'])
            ret_fine.append(rgb_dict['fine'])
            ret_fine_disp.append(rgb_dict['disp_fine'])
        else:                                        # use only coarse rays
            ret_coarse.append(rgb_dict['coarse'])
            ret_coarse_disp.append(rgb_dict['disp_coarse'])

    if opts.N_importance > 0:
        # RGB, DIST / RGB, DIST
        return torch.cat(ret_coarse, dim=0), torch.cat(ret_coarse_disp, dim=0), \
               torch.cat(ret_fine, dim=0), torch.cat(ret_fine_disp, dim=0)
    else:
        return torch.cat(ret_coarse, dim=0), torch.cat(ret_coarse_disp, dim=0), None, None


def render_rays(rays, model, fn_posenc, fn_posenc_d, opts):
    # 1. pre process : make (pts and dirs) (embedded)
    embedded, z_vals, rays_d = pre_process(rays, fn_posenc, fn_posenc_d, opts)

    # 2. run model by net_chunk
    net_chunk = opts.net_chunk
    outputs_flat = torch.cat([model(embedded[i:i+net_chunk]) for i in range(0, embedded.shape[0], net_chunk)], 0)  # [n_pts, 4]
    size = [z_vals.size(0), z_vals.size(1), 4]      # [bs, n_samples, 4]
    outputs = outputs_flat.reshape(size)            # [bs, n_samples, 4]

    # 3. post process : render each pixel color by formula (3) in nerf paper
    rgb_map, disp_map, acc_map, weights, depth_map = post_process(outputs, z_vals, rays_d)

    if opts.N_importance > 0:

        # 4. pre precess
        rays = rays.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
        embedded_fine, z_vals_fine, rays_d = pre_process_for_hierarchical(rays, z_vals, weights, fn_posenc, fn_posenc_d, opts)

        # 5. run model by net_chunk
        outputs_fine_flat = torch.cat([model(embedded_fine[i:i + net_chunk], is_fine=True) for i in range(0, embedded_fine.shape[0], net_chunk)], 0)
        size_fine = [z_vals_fine.size(0), z_vals_fine.size(1), 4]  # [4096, 64 + 128, 4]
        outputs_fine = outputs_fine_flat.reshape(size_fine)

        # 6. post process : render each pixel color by formula (3) in nerf paper
        rgb_map_fine, disp_map_fine, acc_map_fine, weights_fine, depth_map_fine = post_process(outputs_fine, z_vals_fine, rays_d)

        return {'coarse': rgb_map, 'disp_coarse': disp_map, 'fine': rgb_map_fine, 'disp_fine': disp_map_fine}
    return {'coarse': rgb_map, 'disp_coarse': disp_map}


def pre_process(rays, fn_posenc, fn_posenc_d, opts):

    N_rays = rays.size(0)
    # assert N_rays == opts.chunk, 'N_rays must be same to chunk'
    rays_o, rays_d = rays[:, :3], rays[:, 3:]

    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # make normal vector

    # FIXME only llff

    near = opts.near * torch.ones([N_rays, 1], device=torch.device(f'cuda:{opts.gpu_ids[opts.rank]}'))
    far = opts.far * torch.ones([N_rays, 1], device=torch.device(f'cuda:{opts.gpu_ids[opts.rank]}'))

    t_vals = torch.linspace(0., 1., steps=opts.N_samples, device=torch.device(f'cuda:{opts.gpu_ids[opts.rank]}'))
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, opts.N_samples])
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    if opts.is_test:
        torch.manual_seed(opts.seed)
    t_rand = torch.rand([N_rays, opts.N_samples], device=torch.device(f'cuda:{opts.gpu_ids[opts.rank]}'))
    z_vals = lower + (upper-lower) * t_rand

    # rays_o, rays_d : [B, 1, 3], z_vals : [B, 64, 1]
    input_pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    input_pts_flat = input_pts.view(-1, 3)                                # [1024/4096, 64, 3] -> [65536/262144, 3]
    input_pts_embedded = fn_posenc(input_pts_flat)                        # [n_pts, 63]

    input_dirs = viewdirs.unsqueeze(1).expand(input_pts.size())           # [bs, 3] -> [bs, 1, 3]-> [bs, n_samples, 3]
    input_dirs_flat = input_dirs.reshape(-1, 3)                           # [n_pts, 3]
    input_dirs_embedded = fn_posenc_d(input_dirs_flat)                    # [n_pts, 27]

    embedded = torch.cat([input_pts_embedded, input_dirs_embedded], -1)   # [n_pts, 90]
    return embedded, z_vals, rays_d


def pre_process_for_hierarchical(rays, z_vals, weights, fn_posenc, fn_posenc_d, opts):

    rays_o, rays_d = rays[:, :3], rays[:, 3:]
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # make normal vector

    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], opts.N_importance, det=(opts.perturb == 0.), opts=opts)
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
def sample_pdf(bins, weights, N_samples, det=False, opts=None):

    # assert 가 맞으면 넘어감
    assert opts is not None

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
        if opts.is_test:
            torch.manual_seed(opts.seed)
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])


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


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


class GetterRayBatchIdx(object):
    def __init__(self, rays_rgb):
        self.rays_rgb = rays_rgb
        self.epoch = 0
        self.i_batch = 0

    def shuffle_ray_idx(self, batch_size):
        print("Shuffle data after an epoch!")
        rand_idx = torch.randperm(self.rays_rgb.shape[0])
        self.rays_rgb = self.rays_rgb[rand_idx]
        self.i_batch = batch_size
        self.epoch += 1

    def __call__(self, batch_size):
        self.i_batch += batch_size
        if self.i_batch >= self.rays_rgb.shape[0]:
            self.shuffle_ray_idx(batch_size)
        return self.i_batch, self.rays_rgb, self.epoch


