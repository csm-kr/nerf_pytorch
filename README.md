# NeRF_Pytorch_

Pytorch re-implementation of [NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields)

## Environment

## Training

- batch : 1024 (gtx 1080ti)
- loss : MSE
- dataset : Realistic Synthetic 360
- iterations : 300k
- optimizer : Adam
- init lr : 5e-4
- scheduler : CosineAnnealingLR (upto 5e-5)


## Results

- quantitative results 

| data  | exp   | model              | Batch size     | resolution |  PSNR   |  SSIM  | LPIPS  |   Loss   | 
|-------|-------|--------------------|----------------|------------|---------|--------|--------|----------|
| lego  | 0803  | Hierarchical       | 1024           | 400 x 400  | 31.0766 | 0.9597 | 0.0434 | 0.000817 |

- qualitative results

![](./figures/000.png)


### TODO LIST

- [x] README.md
- [x] Make coarse + fine network 
- [x] Demonstrate render processing
- [x] scheduler
- [x] Half version 
- [x] Measure performance ssim, psnr and lpips
- [ ] Rendering
- [ ] Quick start 



