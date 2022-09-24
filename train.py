import os
import time
import torch
import numpy as np
from utils import make_o_d, sample_rays_and_pixel, batchify_rays_and_render_by_chunk, mse2psnr, getSSIM, getLPIPS


def train_each_iters(i, i_train, images, poses, hwk, model, fn_posenc, fn_posenc_d, vis, optimizer, criterion,
                     result_best, opts, getter_ray_batch_idx=None):
    model.train()

    epoch = 0

    # ray_o, ray_d is on the cpu
    # target_img is on the gpu

    if getter_ray_batch_idx is not None and opts.global_batch:
        # batch가 size를 넘어가면 shuffle
        i_batch, rays_rgb, epoch = getter_ray_batch_idx(opts.batch_size)
        # Random over all images
        batch = rays_rgb[i_batch - opts.batch_size:i_batch]  # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_img = batch[:2], batch[2]
        rays_o, rays_d = batch_rays
        img_h, img_w, img_k = hwk

    else:
        # sample train index
        i_img = np.random.choice(i_train)
        target_img = torch.from_numpy(images[i_img]).type(torch.float32)
        target_pose = torch.from_numpy(poses[i_img, :3, :4])

        img_h, img_w, img_k = hwk

        # make rays_o and rays_d
        rays_o, rays_d = make_o_d(img_w, img_h, img_k, target_pose)                           # [400, 400, 3]
        rays_o, rays_d, target_img = sample_rays_and_pixel(rays_o, rays_d, target_img, opts)  # [1024,3]

    # ** assign target_img to cuda **
    target_img = target_img.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))

    pred_rgb_c, pred_rgb_f = batchify_rays_and_render_by_chunk(rays_o, rays_d, model, fn_posenc, fn_posenc_d, img_h, img_w, img_k, opts)  # [1024,4]

    # update optimizer
    optimizer.zero_grad()
    img_loss_f = criterion(pred_rgb_f, target_img)
    psnr_f = mse2psnr(img_loss_f)

    # getting ssim and lpips for image -> at test.py
    # ssim = getSSIM()
    # lpips = getLPIPS()

    img_loss_c = criterion(pred_rgb_c, target_img)
    psnr_c = mse2psnr(img_loss_c)

    loss = img_loss_c + img_loss_f

    loss.backward()
    optimizer.step()

    psnr = psnr_f

    for param_group in optimizer.param_groups:
        lr = param_group['lr']

    # for each steps
    if i % opts.vis_step == 0:

        # print
        print('Epoch: {0}\t'
              'Step: [{1}/{2}]\t'
              'Loss_c: {loss_c:.4f}\t'
              'Loss_f: {loss_f:.4f}\t'
              'Loss: {loss:.4f}\t'
              'PSNR_c: {psnr_c:.4f}\t'
              'PSNR_f: {psnr_f:.4f}\t'
              'Learning rate: {lr:.7f} s \t'
              .format(epoch,
                      i,
                      opts.N_iters,
                      loss_c=img_loss_c,
                      loss_f=img_loss_f,
                      loss=loss,
                      psnr_c=psnr_c.item(),
                      psnr_f=psnr_f.item(),
                      lr=lr))

        # visdom
        if vis is not None:
            vis.line(X=torch.ones((1, 2)) * i,
                     Y=torch.Tensor([loss, psnr.item()]).unsqueeze(0),
                     update='append',
                     win='training_loss',
                     opts=dict(x_label='step',
                               y_label='loss',
                               title='loss',
                               legend=['total_loss', 'psnr']))
    # save checkpoint
    checkpoint = {'idx': i,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    save_path = os.path.join(opts.log_dir, opts.name)
    os.makedirs(save_path, exist_ok=True)

    if i % opts.save_step == 0 and i > 0:
        torch.save(checkpoint, os.path.join(save_path, opts.name + '_{}.pth.tar'.format(i)))

    if result_best['psnr'] < psnr:
        result_best['i'] = i
        result_best['loss'] = loss
        result_best['psnr'] = psnr
        torch.save(checkpoint, os.path.join(save_path, opts.name + '_best.pth.tar'))

    return result_best