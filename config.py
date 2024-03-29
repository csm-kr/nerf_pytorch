import configargparse


def get_args_parser():
    parser = configargparse.ArgumentParser(add_help=False)
    # config
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--name', type=str)

    # visdom
    parser.add_argument('--visdom_port', type=int, default=2023)

    # etc
    parser.add_argument('--vis_step', type=int, default=100)
    parser.add_argument('--save_step', type=int, default=50000)

    # save
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # dataset
    parser.add_argument('--data_type', type=str, help='llff or blender')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--llffhold', type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument('--near', type=float)
    parser.add_argument('--far', type=float)

    # training
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1024)  # 4096 (official)
    parser.add_argument('--chunk', type=int, default=4096)       # 32768 (32 * 1024) (official)
    parser.add_argument('--net_chunk', type=int, default=65536)  # 65536 (64 * 1024) (official)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.set_defaults(vis_points_rays=False)
    parser.add_argument('--vis_points_rays_ture', dest='vis_points_rays', action='store_true')

    # global batch
    parser.set_defaults(global_batch=True)
    parser.add_argument('--global_batch_false', dest='global_batch', action='store_false')
    parser.add_argument('--N_iters', type=int)
    parser.add_argument('--warmup_iters', type=int, default=2500)
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128, help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')

    parser.set_defaults(use_viewdirs=True)
    parser.add_argument("--white_bkgd", type=bool, default=True, help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.set_defaults(half_res=True)
    parser.add_argument('--full_res', dest='half_res', action='store_false')

    parser.add_argument("--testskip", type=int, default=8, help='among 200 test dataset, skip number e.g. 8 -> 200//8 = 25')

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
    parser = configargparse.ArgumentParser('nerf lego training', parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4
    print(opts)

