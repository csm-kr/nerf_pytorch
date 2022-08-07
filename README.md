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

| data          |  model             | Batch size     | resolution |  PSNR   |  SSIM  | LPIPS  | 
|---------------|--------------------|----------------|------------|---------|--------|--------|
| Chair         | Hierarchical       | 4096           | 800 x 800  | 33.0000 | 0.9670 | 0.0460 | 
| Drums         | Hierarchical       | 4096           | 800 x 800  | 25.0100 | 0.9250 | 0.0910 | 
| Ficus         | Hierarchical       | 4096           | 800 x 800  | 30.1300 | 0.9640 | 0.0440 | 
| Hotdog        | Hierarchical       | 4096           | 800 x 800  | 36.1800 | 0.9740 | 0.1210 | 
| Lego          | Hierarchical       | 4096           | 800 x 800  | 32.5400 | 0.9610 | 0.0500 | 
| Materials     | Hierarchical       | 4096           | 800 x 800  | 29.6200 | 0.9490 | 0.0630 | 
| Mic           | Hierarchical       | 4096           | 800 x 800  | 32.9100 | 0.9800 | 0.0280 | 
| Ship          | Hierarchical       | 4096           | 800 x 800  | 28.6500 | 0.8560 | 0.2060 | 

- quantitative results (this repo)

| data          | model              | Batch size     | resolution |  PSNR   |  SSIM  | LPIPS  | 
|---------------|--------------------|----------------|------------|---------|--------|--------|
| drums         | Hierarchical       | 1024           | 400 x 400  | 25.6536 | 0.9292 | 0.0769 | 
| hotdog        | Hierarchical       | 1024           | 400 x 400  | 37.3367 | 0.9809 | 0.0294 | 
| lego          | Hierarchical       | 1024           | 400 x 400  | 31.7081 | 0.9636 | 0.0386 | 
| materials     | Hierarchical       | 1024           | 400 x 400  | 29.8234 | 0.9570 | 0.0535 | 

- qualitative results

![drums_rgb](./figures/drums_000.png)
![drums_gif](./figures/drums_rgb.gif)

![hotdog_rgb](./figures/hotdog_000.png)
![hotdog_gif](./figures/hotdog_rgb.gif)

![lego_rgb](./figures/000.png)
![lego_gif](./figures/lego.gif)

![material_rgb](./figures/materials_000.png)
![material_gif](./figures/materials_rgb.gif)


### TODO LIST

- [x] README.md
- [x] Make coarse + fine network 
- [x] Demonstrate render processing
- [x] scheduler
- [x] Half version 
- [x] Measure performance ssim, psnr and lpips
- [x] Rendering
- [ ] Other dataset [LIFF]
- [ ] Quick start 



