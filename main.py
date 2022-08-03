import torch
import visdom
import argparse

# model
from model import NeRFs

# dataset
from blender import load_blender
from PE import get_positional_encoder

# train and test
from train import train_each_iters
from test import test_and_eval


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
    images, poses, hwk, i_split = load_blender(opts.root, opts.name, opts.half_res, opts.testskip, opts.white_bkgd)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.N_iters, eta_min=5e-5)

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = 0
    start = start + 1

    result_best = {'i': 0, 'loss': 0, 'psnr': 0}
    result_best_test = {'i': 0, 'loss': 0, 'psnr': 0}

    for i in range(start, opts.N_iters):

        result_best = train_each_iters(i, i_train, images, poses, hwk, model, fn_posenc, fn_posenc_d,
                                       vis, optimizer, criterion, result_best, opts)
        if i % opts.save_step == 0 and i > 0:
            result_best_test = test_and_eval(i, i_test, images, poses, hwk, model, fn_posenc, fn_posenc_d,
                                             vis, criterion, result_best_test, opts)

        scheduler.step()

    test_and_eval('best', i_test, images, poses, hwk, model, fn_posenc, fn_posenc_d, vis, criterion, result_best_test, opts)


if __name__ == '__main__':
    from config import get_args_parser
    parser = argparse.ArgumentParser('nerf lego training', parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4
    main_worker(0, opts)