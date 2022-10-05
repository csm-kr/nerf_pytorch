import os
import torch
import numpy as np

import visdom
import imageio
from tqdm import tqdm, trange

# data
from blender import load_blender
from llff import load_llff_data

# model
from PE import get_positional_encoder
from model import NeRFs
from utils import to8b, batchify_rays_and_render_by_chunk, make_o_d


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


# theta, -30, 4 : shperical to cartesian
# theta : [-180, -171, -162 ..., 171] (40)
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def get_render_pose(n_angle=1, single_angle=-1, phi=-30.0):
    if not n_angle == 1 and single_angle == -1:
        render_poses = torch.stack([pose_spherical(angle, phi, 4.0)
                                    for angle in np.linspace(-180, 180, n_angle+1)[:-1]], 0)
    else:
        render_poses = pose_spherical(single_angle, phi, 4.0).unsqueeze(0)
    return render_poses


def render(i, hwk, model, fn_posenc, fn_posenc_d, opts, n_angle=40, single_angle=-1, render_poses=None):

    print('Start Rendering for idx'.format(i))

    # make render poses for blender
    if render_poses is None and opts.data_type == 'blender':
        render_poses = get_render_pose(n_angle=n_angle, single_angle=single_angle, phi=opts.phi)

    # load check points
    checkpoint = torch.load(os.path.join(opts.log_dir, opts.name, opts.name+'_{}.pth.tar'.format(i)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # make save dir
    save_render_dir = os.path.join(opts.log_dir, opts.name, opts.name+'_{}'.format(i), 'render_result')
    os.makedirs(save_render_dir, exist_ok=True)

    render_poses = torch.Tensor(render_poses).to(torch.device(f'cuda:{opts.gpu_ids[opts.rank]}'))
    img_h, img_w, img_k = hwk

    rgbs = []
    disps = []
    # depths = []
    # accs = []

    with torch.no_grad():
        for i, render_pose in enumerate(tqdm(render_poses)):

            print('RENDERING... idx: {}'.format(i))
            rays_o, rays_d = make_o_d(img_w, img_h, img_k, render_pose[:3, :4])  # [1]
            _, _, pred_rgb, pred_disp = batchify_rays_and_render_by_chunk(rays_o, rays_d, model, fn_posenc, fn_posenc_d, img_h, img_w, img_k, opts)

            # save test image rgb
            rgb = torch.reshape(pred_rgb, [img_h, img_w, 3])
            rgb_np = rgb.cpu().numpy()
            rgbs.append(rgb_np)

            # save test image disp
            disp = torch.reshape(pred_disp, [img_h, img_w, 1])
            disp_np = disp.cpu().numpy()
            disps.append(disp_np)

            if not single_angle == -1:
                imageio.imwrite(os.path.join(save_render_dir, '{}_{}_rgb.png'.format(opts.single_angle, str(opts.phi))), to8b(rgb_np))
                imageio.imwrite(os.path.join(save_render_dir, '{}_{}_disp.png'.format(opts.single_angle, str(opts.phi))), to8b(disp_np/np.nanmax(disp_np)))

        rgbs = np.stack(rgbs, 0)
        disp_nps = np.stack(disps, 0)

    if single_angle == -1:
        imageio.mimwrite(os.path.join(save_render_dir, "{}_rgb.mp4".format(opts.name)), to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(save_render_dir, "{}_rgb.gif".format(opts.name)), to8b(rgbs), duration=0.04)

        imageio.mimwrite(os.path.join(save_render_dir, "{}_disp.mp4".format(opts.name)), to8b(disp_nps/np.nanmax(disp_nps)), fps=30, quality=8)
        imageio.mimwrite(os.path.join(save_render_dir, "{}_disp.gif".format(opts.name)), to8b(disp_nps/np.nanmax(disp_nps)), duration=0.04)


def render_worker(rank, opts):

    if opts.data_type == 'blender':
        images, poses, hwk, i_split, render_poses = load_blender(opts.half_res, opts.testskip, opts.white_bkgd, opts)
    elif opts.data_type == 'llff':
        images, poses, hwk, i_split, render_poses = load_llff_data(opts=opts)

    device = torch.device('cuda:{}'.format(opts.gpu_ids[opts.rank]))
    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)
    model = NeRFs(D=8, W=256, input_ch=63, input_ch_d=27, skips=[4]).to(device)
    render('best', hwk, model, fn_posenc, fn_posenc_d, opts, render_poses=render_poses)


if __name__ == '__main__':

    # parser
    import configargparse
    from config import get_args_parser

    parser = configargparse.ArgumentParser('nerf testing', parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4
    print(opts)

    render_worker(0, opts)