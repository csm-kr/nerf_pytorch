import os
import time
import torch
import imageio
import numpy as np
from tqdm import tqdm

# parser
import configargparse
from config import get_args_parser
# dataset
from blender import load_blender
from llff import load_llff_data
# model
from model import NeRFs
from PE import get_positional_encoder
from utils import mse2psnr, to8b, make_o_d, render_by_chunk, img2mse, getSSIM, getLPIPS
# render
from render import render


def test_and_eval(i, i_test, images, poses, hwk, model, fn_posenc, fn_posenc_d, vis, criterion, result_best_test, opts):

    print('Start Testing for idx'.format(i))
    # opts.is_test = True - 같은 testset 에서 같은 성능이 나오도록

    # load check points
    checkpoint = torch.load(os.path.join(opts.log_dir, opts.name, opts.name+'_{}.pth.tar'.format(i)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # make save dir
    save_test_dir = os.path.join(opts.log_dir, opts.name, opts.name+'_{}'.format(i), 'test_result')
    os.makedirs(save_test_dir, exist_ok=True)

    img_h, img_w, img_k = hwk

    losses = []
    psnrs = []
    ssims = []
    lpipses = []

    test_imgs = torch.from_numpy(images[i_test]).to(torch.device(f'cuda:{opts.gpu_ids[opts.rank]}'))
    test_poses = torch.from_numpy(poses[i_test]).to(torch.device(f'cuda:{opts.gpu_ids[opts.rank]}'))

    with torch.no_grad():

        for i, test_pose in enumerate(tqdm(test_poses)):

            rays_o, rays_d = make_o_d(img_w, img_h, img_k, test_pose[:3][:4])  # [1]
            _, _, pred_rgb, pred_disp = render_by_chunk(rays_o, rays_d, model, fn_posenc, fn_posenc_d, img_h, img_w, img_k, opts)  # ** hierachicle sampling **
            # https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf.py#L403

            # SAVE test image
            rgb = torch.reshape(pred_rgb, [img_h, img_w, 3])
            rgb_np = rgb.cpu().numpy()

            disp = torch.reshape(pred_disp, [img_h, img_w, 1])
            disp_np = disp.cpu().numpy()
            disp_np = disp_np / np.max(disp_np)

            rgb8 = to8b(rgb_np)
            dist8 = to8b(disp_np)
            savefilename_rgb = os.path.join(save_test_dir, '{}_{:03d}.png'.format(opts.name, i))
            imageio.imwrite(savefilename_rgb, rgb8)
            savefilename_dist = os.path.join(save_test_dir, '{}_{:03d}_dist.png'.format(opts.name, i))
            imageio.imwrite(savefilename_dist, dist8)

            # GET loss & psnr
            target_img_flat = torch.reshape(test_imgs[i], [-1, 3]).to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
            img_loss = criterion(pred_rgb, target_img_flat)

            loss = img_loss
            psnr = mse2psnr(img_loss)

            target_img_hwc = target_img_flat.reshape(img_h, img_w, -1).type(torch.float32)
            ssim = getSSIM(target_img_hwc, rgb)
            lpips = getLPIPS(target_img_hwc, rgb)

            # tensor to float
            losses.append(img_loss.cpu().item())
            psnrs.append(psnr.cpu().item())
            ssims.append(ssim.cpu().item())
            lpipses.append(lpips.cpu().item())

            print('idx : {} | Loss : {:.6f} | PSNR : {:.4f} | SSIM : {:.4f} | LPIPS : {:.4f}'.format(i, img_loss.item(),
                                                                                                     psnr.item(),
                                                                                                     ssim.item(),
                                                                                                     lpips.item()))

            # save best result
            if result_best_test['psnr'] < psnr:
                result_best_test['i'] = i
                result_best_test['loss'] = loss.item()
                result_best_test['psnr'] = psnr.item()
                result_best_test['ssim'] = ssim.item()
                result_best_test['lpips'] = lpips.item()

    losses = np.array(losses)
    psnrs = np.array(psnrs)
    ssims = np.array(ssims)
    lpipses = np.array(lpipses)

    print('(MEAN Result for Testing) LOSS : {:.6f} , PSNR : {:.4f} , SSIM : {:.4f}, LPIPS : {:.4f}'.format(losses.mean(),
                                                                                                           psnrs.mean(),
                                                                                                           ssims.mean(),
                                                                                                           lpipses.mean()))
    f = open(os.path.join(save_test_dir, "_result.txt"), 'w')
    for i in range(len(losses)):
        line = 'idx:{}\t loss:{:.6f}\t psnr:{:.4f}\t ssim:{:.4f}\t lpips:{:.4f}\n'.format(i, losses[i], psnrs[i], ssims[i], lpipses[i])
        f.write(line)
    line = '*mean*   loss:{:.6f}\t psnr:{:.4f}\t ssim:{:.4f}\t lpips:{:.4f}\n'.format(losses.mean(), psnrs.mean(),
                                                                                      ssims.mean(), lpipses.mean())
    f.write(line)
    f.close()

    # opts.is_test = False
    return result_best_test


def test_worker(rank, opts):

    if opts.data_type == 'blender':
        images, poses, hwk, i_split, render_poses = load_blender(opts.half_res, opts.testskip, opts.white_bkgd, opts)
    elif opts.data_type == 'llff':
        images, poses, hwk, i_split, render_poses = load_llff_data(opts=opts)

    i_train, i_val, i_test = i_split
    device = torch.device('cuda:{}'.format(opts.gpu_ids[opts.rank]))
    vis = None

    fn_posenc, input_ch = get_positional_encoder(L=10)
    fn_posenc_d, input_ch_d = get_positional_encoder(L=4)

    model = NeRFs(D=8, W=256, input_ch=63, input_ch_d=27, skips=[4]).to(device)
    criterion = torch.nn.MSELoss()
    result_best_test = {'i': 0, 'loss': 0, 'psnr': 0, 'ssim': 0, 'lpips': 0}

    # # test for 20 dataset
    # i_train_test = np.arange(20)
    # test_and_eval(50000, i_train, images, poses, hwk, model, fn_posenc, fn_posenc_d, vis, criterion, result_best_test, opts)

    test_and_eval('best', i_test, images, poses, hwk, model, fn_posenc, fn_posenc_d, vis, criterion, result_best_test, opts)
    render('best', hwk, model, fn_posenc, fn_posenc_d, opts, n_angle=40, single_angle=-1, render_poses=render_poses)


if __name__ == '__main__':

    parser = configargparse.ArgumentParser('nerf lego testing', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    print(opts)
    test_worker(0, opts)
