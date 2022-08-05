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

- quantitative results (official)

| data          |  model             | Batch size     | resolution |  PSNR   |  SSIM  | LPIPS  |   Loss   | 
|---------------|--------------------|----------------|------------|---------|--------|--------|----------|
| drums(f)      | Hierarchical       | 4096           | 800 x 800  | 25.01   | 0.925  | 0.091  | -        |
| hotdog(f)     | Hierarchical       | 4096           | 800 x 800  | 36.18   | 0.974  | 0.121  | -        |
| lego(f)       | Hierarchical       | 4096           | 800 x 800  | 32.54   | 0.961  | 0.050  | -        |
| materials(f)  | Hierarchical       | 4096           | 800 x 800  | 29.62   | 0.949  | 0.063  | -        |

- quantitative results (this repo)

| data          | model              | Batch size     | resolution |  PSNR   |  SSIM  | LPIPS  |   Loss   | 
|---------------|--------------------|----------------|------------|---------|--------|--------|----------|
| drums         | Hierarchical       | 1024           | 400 x 400  | 25.6536 | 0.9292 | 0.0769 | 0.002917 |
| hotdog        | Hierarchical       | 1024           | 400 x 400  | - | - | - | - |
| lego          | Hierarchical       | 1024           | 400 x 400  | 31.0766 | 0.9597 | 0.0434 | 0.000817 |
| materials     | Hierarchical       | 1024           | 400 x 400  | 29.8234 | 0.9570 | 0.0535 | 0.001700 |




- qualitative results

![](./figures/000.png)
![lego_gif](./figures/lego.gif)

![material_rgb](./figures/materials_000.png)
![material_gif](./figures/materials_rgb.gif)

![drums_rgb](./figures/drums_000.png)
![drums_gif](./figures/drums_rgb.gif)


### TODO LIST

- [x] README.md
- [x] Make coarse + fine network 
- [x] Demonstrate render processing
- [x] scheduler
- [x] Half version 
- [x] Measure performance ssim, psnr and lpips
- [x] Rendering
- [ ] Quick start 



