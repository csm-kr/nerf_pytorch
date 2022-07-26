import os
import time
import torch
import numpy as np
from utils import make_o_d, sample_rays_and_pixel, batchify_rays_and_render_by_chunk, mse2psnr


def train_each_iters(i, i_train, images, poses, hwk, model, fn_posenc, fn_posenc_d, vis, optimizer, criterion,
                     result_best, opts):

    # sample train index
    i_img = np.random.choice(i_train)
    target_img = torch.from_numpy(images[i_img])
    target_pose = torch.from_numpy(poses[i_img, :3, :4])

    img_h, img_w, img_k = hwk

    # make rays_o and rays_d
    rays_o, rays_d = make_o_d(img_w, img_h, img_k, target_pose)  # [800, 800, 3]
    rays_o, rays_d, target_img = sample_rays_and_pixel(rays_o, rays_d, target_img, opts)  # [1024,3]
    pred_rgb = batchify_rays_and_render_by_chunk(rays_o, rays_d, model, opts, fn_posenc, fn_posenc_d)  # [1024,4]

    # ** assign target_img to cuda **
    target_img = target_img.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))

    # update optimizer
    optimizer.zero_grad()
    img_loss = criterion(pred_rgb, target_img)

    loss = img_loss
    psnr = mse2psnr(img_loss)

    loss.backward()
    optimizer.step()

    toc = time.time()

    for param_group in optimizer.param_groups:
        lr = param_group['lr']

    # for each steps
    if i % opts.vis_step == 0:
        # print
        print('Step: [{0}/{1}]\t'
              'Loss: {loss:.4f}\t'
              'PSNR: {psnr:.4f}\t'
              'Learning rate: {lr:.7f} s \t'
              .format(i,
                      opts.N_iters,
                      loss=loss,
                      psnr=psnr.item(),
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

    save_path = os.path.join('./logs', opts.name)
    os.makedirs(save_path, exist_ok=True)

    if i % opts.save_step == 0 and i > 0:
        torch.save(checkpoint, os.path.join(save_path, opts.name + '_{}.pth.tar'.format(i)))

    if result_best['psnr'] < psnr:
        result_best['i'] = i
        result_best['loss'] = loss
        result_best['psnr'] = psnr
        torch.save(checkpoint, os.path.join(save_path, opts.name + '_best.pth.tar'))

    return result_best