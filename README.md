# NeRF_Pytorch_

Pytorch re-implementation of [NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields)

## Environment

## Training

- batch : 1024 (gtx 1080ti)
- loss : MSE
- dataset : Realistic Synthetic 360
- iterations : 200k
- optimizer : Adam
- init lr : 5e-4
- scheduler : ExponentialLR (upto 5e-5)
- scheduler : CosineAnnealingLR (upto 5e-5)


## Results

- quantitative results 
| data  | exp   | model                 | Batch size     | res        |  PSNR   |  SSIM  | LPIPS  | Loss   | 
|-------|-------|-----------------------|----------------|------------|---------|--------|--------|--------|
| lego  | 0803  | c(64) + 192(c+f)      | 1024           |400 x 400   | 31.0703 | 0.9597 | 0.0435 |0.000818|

- qualitative results
left top to right down 1000/ 50000/ 100000/ 150000 step

![](./figures/results_1000_50000_100000_1500000.JPG)

### TODO LIST

- [x] README.md
- [x] Make coarse + fine network 
- [x] Demonstrate render processing
- [x] scheduler
- [x] Half version 
- [x] Measure performance ssim and psnr
- [ ] Quick start 
- [ ] Rendering


