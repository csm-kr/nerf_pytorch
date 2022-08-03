import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # visualization
    parser.add_argument('--visdom_port', type=int, default=2023)

    # etc
    parser.add_argument('--vis_step', type=int, default=100)
    parser.add_argument('--save_step', type=int, default=50000)

    # save
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # dataset
    # FIXME
    parser.add_argument('--root', type=str, default=r'./data/nerf_synthetic/ficus')
    parser.add_argument('--name', type=str, default='ficus')
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1024)  # 2 ^ 10
    parser.add_argument('--chunk', type=int, default=4096)       # 2 ^ 12
    parser.add_argument('--net_chunk', type=int, default=65536)  # 2 ^ 16
    parser.add_argument('--near', type=int, default=2)
    parser.add_argument('--far', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=0)

    # training
    parser.add_argument('--N_iters', type=int, default=300001)
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128, help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", type=bool, default=True, help='use full 5D input instead of 3D')
    parser.add_argument("--white_bkgd", type=bool, default=True, help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", type=bool, default=True, help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--testskip", type=int, default=8, help='among 200 test dataset, skip number e.g. 8 -> 200//8 = 25')
    # parser.add_argument("--precrop_iters", type=int, default=500)

    # testing
    parser.add_argument('--seed', type=int, default=0)
    parser.set_defaults(is_test=False)
    parser.add_argument('--testing', dest='is_test', action='store_true')

    # rendering
    parser.add_argument('--n_angle', type=int, default=40)
    parser.add_argument('--single_angle', type=int, default=-1)
    parser.add_argument('--phi', type=float, default=-30.0)

    # for multi-gpu
    parser.add_argument('--gpu_ids', nargs="+", default=['0'])   # usage : --gpu_ids 0, 1, 2, 3
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('nerf lego training', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4
    print(opts)

