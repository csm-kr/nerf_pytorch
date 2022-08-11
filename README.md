# NeRF_Pytorch_

Pytorch re-implementation of [NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields)

We have referenced the following repos - ofiiclal [code](https://github.com/bmild/nerf) and pytorch nerf [code](https://github.com/yenchenlin/nerf-pytorch)


## Environment

## Training

- batch : 1024 (gtx 1080ti) due to 
- loss : MSE
- dataset : Realistic Synthetic 360
- iterations : 300k
- optimizer : Adam
- init lr : 5e-4
- scheduler : CosineAnnealingLR (upto 5e-5) + warmup(10k)
- network init : xavier for stability 

## Results

- quantitative results (official)

| data          |  model             | Batch size     | resolution |  PSNR   |  SSIM  | LPIPS  | 
|---------------|--------------------|----------------|------------|---------|--------|--------|
| chair         | Hierarchical       | 4096           | 800 x 800  | 33.0000 | 0.9670 | 0.0460 | 
| drums         | Hierarchical       | 4096           | 800 x 800  | 25.0100 | 0.9250 | 0.0910 | 
| ficus         | Hierarchical       | 4096           | 800 x 800  | 30.1300 | 0.9640 | 0.0440 | 
| hotdog        | Hierarchical       | 4096           | 800 x 800  | 36.1800 | 0.9740 | 0.1210 | 
| lego          | Hierarchical       | 4096           | 800 x 800  | 32.5400 | 0.9610 | 0.0500 | 
| materials     | Hierarchical       | 4096           | 800 x 800  | 29.6200 | 0.9490 | 0.0630 | 
| mic           | Hierarchical       | 4096           | 800 x 800  | 32.9100 | 0.9800 | 0.0280 | 
| ship          | Hierarchical       | 4096           | 800 x 800  | 28.6500 | 0.8560 | 0.2060 | 
| mean          | Hierarchical       | 4096           | 800 x 800  | 31.0100 | 0.9470 | 0.0810 | 

- quantitative results (this repo)

| data          | model              | Batch size     | resolution |  PSNR   |  SSIM  | LPIPS  | Link | 
|---------------|--------------------|----------------|------------|---------|--------|--------|------|
| chair         | Hierarchical       | 1024           | 400 x 400  | 34.8707 | 0.9790 | 0.0280 | [link](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/ETsPfnY_ohFIt5okXA9of4wBdmviTiU2mMMxS44Loz85ew) |  
| drums         | Hierarchical       | 1024           | 400 x 400  | 25.6536 | 0.9292 | 0.0769 | [link](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/Ect9iuZVm2xEkRiKhiegxwEBp0vz0LGBF5tMabLe8EUy4w) | 
| ficus         | Hierarchical       | 1024           | 400 x 400  | 29.3877 | 0.9643 | 0.0447 | [link](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EYhf5_lSY3lHtResPgI9940BEUdzZeU04_M6RJkDRrGyYA) | 
| hotdog        | Hierarchical       | 1024           | 400 x 400  | 37.3367 | 0.9809 | 0.0294 | [link](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EURUZTliqixLmNArQ5FQfWMB7eodTTXDzlpqhsPh7toi9A) | 
| lego          | Hierarchical       | 1024           | 400 x 400  | 31.7081 | 0.9636 | 0.0386 | [link](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EUjqsS-vTmNOrOs-PO8uOPoBXgGVoW5-VOeCkn986iZOpQ) | 
| materials     | Hierarchical       | 1024           | 400 x 400  | 29.8181 | 0.9571 | 0.0534 | [link](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EYnayR9P5a1No8mctOjWUgwBW0CLak30IeEXXmW7mATzvw) | 
| mic           | Hierarchical       | 1024           | 400 x 400  | 33.8926 | 0.9803 | 0.0239 | [link](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EcWVEi5Al3FGnMM18nE8p-0BLTVTOaIqP0HE0txACXHo1w) | 
| ship          | Hierarchical       | 1024           | 400 x 400  | 29.6258 | 0.8762 | 0.1342 | [link](https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EUiZG2RitnpCpUIo2S-eAmgB1Pho2c4Fq3QXLRUy_hGtsg) | 
| mean          | Hierarchical       | 1024           | 400 x 400  | 31.5367 | 0.9538 | 0.0536 | -    | 

- For most classes, our repo performed slightly better that original papers.

- we think it because the differences which are network init, resolution(batch size), and warm up training

- qualitative results

![chair_rgb](./figures/chair_000.png)
![chair_gif](./figures/chair_rgb.gif)

![drums_rgb](./figures/drums_000.png)
![drums_gif](./figures/drums_rgb.gif)

![drums_rgb](./figures/ficus_000.png)
![drums_gif](./figures/ficus_rgb.gif)

![hotdog_rgb](./figures/hotdog_000.png)
![hotdog_gif](./figures/hotdog_rgb.gif)

![lego_rgb](./figures/000.png)
![lego_gif](./figures/lego.gif)

![material_rgb](./figures/materials_000.png)
![material_gif](./figures/materials_rgb.gif)

![mic_rgb](./figures/mic_000.png)
![mic_gif](./figures/mic_rgb.gif)

![ship_rgb](./figures/ship_000.png)
![ship_gif](./figures/ship_rgb.gif)

### Quick start

1 - download synthetic files at [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) to your root 

2 - download pth files at upper link

3 - make path 'logs' and put it in like this:

```
figures
logs
    |-- chair
    ...
    |-- lego
        |-- lego_best.pth.tar
    ...
    |-- ship
main.py
blender.py
...
utils.py
```

4 - run test.py script with your dataset root and name and 'testing' args 

```
usage: test.py --root 'your root' --name 'the classs you wnat' --testing 
e.g.) python test.py --root ./data/nerf_synthetic --name lego --testing
```

5 - than you can test image and rendering .mp4 and .gif

![mic_rgb](./figures/lego_004.png)

### TODO LIST

- [x] README.md
- [x] Make coarse + fine network 
- [x] Demonstrate render processing
- [x] scheduler
- [x] Half version 
- [x] Measure performance ssim, psnr and lpips
- [x] Rendering
- [x] Quick start 
- [ ] Other dataset [LIFF]

### Reference

https://github.com/bmild/nerf

https://github.com/yenchenlin/nerf-pytorch
