import torch
import visdom

# model
from model import NeRFs

# dataset
from blender import load_blender
from llff import load_llff_data
from PE import get_positional_encoder

# scheduler
from scheduler import CosineAnnealingWarmupRestarts

# train and test
from train import train_each_iters
from test import test_and_eval
from render import render

# numpy get_rays
import numpy as np
from rays import get_rays_np
from utils import GetterRayBatchIdx


def main_worker(rank, opts):
    # 1. parser
    print(opts)

    # 2. visdom
    vis = None
    if opts.visdom_port:
        vis = visdom.Visdom(port=opts.visdom_port)

    # 2. device
    device = torch.device('cuda:{}'.format(opts.gpu_ids[opts.rank]))

    # 3. dataset
    if opts.data_type == 'blender':
        images, poses, hwk, i_split, render_poses = load_blender(opts.half_res, opts.testskip, opts.white_bkgd, opts)
    elif opts.data_type == 'llff':
        images, poses, hwk, i_split, render_poses = load_llff_data(opts=opts)

    i_train, i_val, i_test = i_split

    # 4. model and PE
    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)
    model = NeRFs(D=8, W=256, input_ch=63, input_ch_d=27, skips=[4]).to(device)

    # 5. loss
    criterion = torch.nn.MSELoss()

    # 6. optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opts.lr, betas=(0.9, 0.999))

    # 7. scheduler
    num_steps = int(opts.N_iters)
    warmup_steps = int(opts.warmup_iters)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=num_steps,
        cycle_mult=1.,
        max_lr=opts.lr,
        min_lr=5e-5,
        warmup_steps=warmup_steps
        )

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = 0
    start = start + 1

    result_best = {'i': 0, 'loss': 0, 'psnr': 0}
    result_best_test = {'i': 0, 'loss': 0, 'psnr': 0, 'ssim': 0, 'lpips': 0}

    rays_rgb = None

    if opts.global_batch:
        img_h, img_w, k = hwk
        print('>> [Global Batching] Random Ray for all images')                          # num_img, [ray_o, ray_d], hwf
        rays = np.stack([get_rays_np(img_h, img_w, k, p) for p in poses[:, :3, :4]], 0)  # [20, 2, 378, 504, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        np.random.shuffle(rays_rgb)
        rays_rgb = torch.Tensor(rays_rgb)

    # rays_rgb batch getter for global batch
    getter_ray_batch_idx = GetterRayBatchIdx(rays_rgb)

    for i in range(start, opts.N_iters):
        # train
        result_best = train_each_iters(i, i_train, images, poses, hwk, model, fn_posenc, fn_posenc_d,
                                       vis, optimizer, criterion, result_best, opts, getter_ray_batch_idx)

        # test and render
        if i % opts.save_step == 0 and i > 0:
            # render images
            result_best_test = test_and_eval(i, i_test, images, poses,
                                             hwk, model, fn_posenc, fn_posenc_d,
                                             vis, criterion, result_best_test, opts)
            # render gif and mp4
            render(i, hwk, model, fn_posenc, fn_posenc_d, opts, render_poses=render_poses)
        scheduler.step()

    # test and render at best index
    test_and_eval('best', i_test, images, poses, hwk, model, fn_posenc, fn_posenc_d, vis, criterion, result_best_test, opts)
    render('best', hwk, model, fn_posenc, fn_posenc_d, opts, render_poses=render_poses)


if __name__ == '__main__':
    import configargparse
    from config import get_args_parser

    parser = configargparse.ArgumentParser('nerf training', parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4
    main_worker(0, opts)