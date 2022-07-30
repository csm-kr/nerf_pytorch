import os
import time
import torch
import imageio
import numpy as np
from tqdm import tqdm
# dataset
from blender import load_blender
# model
from model import NeRF
from PE import get_positional_encoder
from utils import mse2psnr, to8b, make_o_d, batchify_rays_and_render_by_chunk, img2mse

LOG_DIR = './logs'
device = 0


def test_and_eval(i, i_test, images, poses, hwk, model, fn_posenc, fn_posenc_d, vis, criterion, result_best_test, opts):

    print('Start Testing for idx'.format(i))
    model.eval()
    checkpoint = torch.load(os.path.join(LOG_DIR, opts.name, opts.name+'_{}.pth.tar'.format(i)))
    model.load_state_dict(checkpoint['model_state_dict'])

    save_test_dir = os.path.join(LOG_DIR, opts.name, opts.name+'_{}'.format(i), 'test_result')
    os.makedirs(save_test_dir, exist_ok=True)

    img_h, img_w, img_k = hwk

    losses = []
    psnrs = []

    test_imgs =torch.from_numpy(images[i_test])
    test_poses = torch.from_numpy(poses[i_test])

    with torch.no_grad():

        for i, test_pose in enumerate(tqdm(test_poses)):

            rays_o, rays_d = make_o_d(img_w, img_h, img_k, test_pose[:3][:4])  # [1]
            _, pred_rgb = batchify_rays_and_render_by_chunk(rays_o, rays_d, model, opts, fn_posenc, fn_posenc_d)  # ** hierachicle sampling **
            # https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf.py#L403

            # SAVE test image
            rgb = torch.reshape(pred_rgb, [img_h, img_w, 3])
            rgb_np = rgb.cpu().numpy()

            rgb8 = to8b(rgb_np)
            savefilename = os.path.join(save_test_dir, '{:03d}.png'.format(i))
            imageio.imwrite(savefilename, rgb8)

            # GET loss & psnr
            target_img_flat = torch.reshape(test_imgs[i], [-1, 3]).to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
            img_loss = criterion(pred_rgb, target_img_flat)

            loss = img_loss
            psnr = mse2psnr(img_loss)
            losses.append(img_loss)
            psnrs.append(psnr)
            print('idx : {} | Loss : {} | PSNR : {}'.format(i, img_loss, psnr))

            # save best result
            if result_best_test['psnr'] < psnr:
                result_best_test['i'] = i
                result_best_test['loss'] = loss
                result_best_test['psnr'] = psnr

    print('BEST Result for Testing) idx : {} , LOSS : {} , PSNR : {}'.format(
        result_best_test['i'], result_best_test['loss'], result_best_test['psnr']))

    f = open(os.path.join(save_test_dir, "_result.txt"), 'w')
    for i in range(len(losses)):
        line = 'idx:{}\tloss:{}\tpsnr:{}\n'.format(i, losses[i], psnrs[i])
        f.write(line)
    f.close()

    return result_best_test


def main_worker(rank, cfg):

    images, poses, hwk, i_split = load_blender(cfg.root, cfg.name, cfg.half_res, cfg.white_bkgd)
    i_train, i_val, i_test = i_split
    img_h, img_w, img_k = hwk

    vis=None

    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)
    # output_ch = 5 if cfg.model.n_importance > 0 else 4

    skips = [4]
    model = NeRF(D=8, W=256,
                 input_ch=input_ch, input_ch_d=input_ch_d, skips=skips).to(device)

    criterion = torch.nn.MSELoss()
    result_best_test = {'i': 0, 'loss': 0, 'psnr': 0}

    # test_and_eval(idx='best',
    #              fn_posenc=fn_posenc,
    #              fn_posenc_d=fn_posenc_d,
    #              model=model,
    #              test_imgs=torch.Tensor(images[i_test]),
    #              test_poses=torch.Tensor(poses[i_test]),
    #              hwk=hwk,
    #              cfg=cfg)

    test_and_eval('best', i_test, images, poses, hwk, model, fn_posenc, fn_posenc_d, vis, criterion, result_best_test, opts)

    # render(idx='best',
    #        fn_posenc=fn_posenc,
    #        fn_posenc_d=fn_posenc_d,
    #        model=model,
    #        hwk=hwk,
    #        cfg=cfg)


if __name__ == '__main__':
    import argparse
    from config import get_args_parser
    parser = argparse.ArgumentParser('nerf lego training', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    print(opts)
    main_worker(0, opts)
