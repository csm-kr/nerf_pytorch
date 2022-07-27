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

| date  | exp   | model                 | Batch size     | # params   | PSNR   | SSIM   | loss   | Time | 
|-------|-------|-----------------------|----------------|------------|--------|--------|--------| -----|
| 07.27 | 1     | coarse(64)            | 1024           |6235582     |0.9163  |0.2613  | 0.1133 | 12s  |
| 07.28 | 2     | coarse(64) + fine(64) | 4096           |6235402     |0.9097  |0.2757  | 0.1133 | 12s  |
| 07.29 | 3     | coarse(64) + fine(128)| 4096           |6235402     |0.9097  |0.2757  | 0.1133 | 12s  |

- qualitative results

left top to right down 1000/ 50000/ 100000/ 150000 step

![](./figures/results_1000_50000_100000_1500000.JPG)

### TODO LIST

- [x] README.md
- [x] Make coarse + fine network 
- [x] Demonstrate render processing
- [x] scheduler
- [ ] Quick start 
- [ ] Measure performance ssim and psnr
- [ ] Rendering

