HPARAMS_REGISTRY = {}

class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


# We only use CIFAR-10 dataset
cifar10 = Hyperparams()
cifar10.dataset = 'cifar10'
cifar10.lr = 0.0002
cifar10.wd = 0.01
cifar10.n_batch = 32
cifar10.ema_rate =  0.9998
cifar10.warmup_iters = 100
cifar10.skip_threshold = 400.0
cifar10.max_iters = 1407     # training ends up based on which is longer between max_iters & epoch.
cifar10.num_epochs = 1
HPARAMS_REGISTRY['cifar10'] = cifar10


def parse_args_and_update_hparams(H, parser, s=None):
    args = parser.parse_args(s)
    valid_args = set(vars(args).keys())

    hps = HPARAMS_REGISTRY['cifar10']
    for k in hps:
        if k not in valid_args:
            raise ValueError(f"{k} not in default args")
    parser.set_defaults(**hps)
    args = parser.parse_args(s)
    H.update(vars(args))


# we manage all parameters here except for model structure.
def add_vae_arguments(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--data_root', type=str, default='../content')

    parser.add_argument('--desc', type=str, default='test')
    parser.add_argument('--restore_path', type=str, default=None,
                        help="checkpoint prefix를 지정하면 그 지점부터 학습을 복원!")        # default = './saved_models/test/latest'
    parser.add_argument('--restore_ema_path', type=str, default=None)                        # default='./saved_models/test/latest'
    parser.add_argument('--restore_log_path', type=str, default=None)                        # default='./saved_models/test/latest-log.jsonl'
    parser.add_argument('--restore_optimizer_path', type=str, default=None)                  # default='./saved_models/test/latest-opt.th'
    parser.add_argument('--dataset', type=str, default='cifar10')

    parser.add_argument('--ema_rate', type=float, default=0.999)

    parser.add_argument('--test_eval', action="store_true")
    parser.add_argument('--warmup_iters', type=float, default=0)

    parser.add_argument('--grad_clip', type=float, default=200.0)
    parser.add_argument('--skip_threshold', type=float, default=400.0)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--lr_prior', type=float, default=0.00015)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--wd_prior', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=10)                 # 10000 (maximum)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.9)

    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--iters_per_ckpt', type=int, default=25000)
    parser.add_argument('--iters_per_print', type=int, default=1000)
    parser.add_argument('--iters_per_save', type=int, default=1500)          # 10000
    parser.add_argument('--iters_per_images', type=int, default=10000)
    parser.add_argument('--epochs_per_eval', type=int, default=10)            # number of epoch
    parser.add_argument('--epochs_per_probe', type=int, default=None)
    parser.add_argument('--epochs_per_eval_save', type=int, default=20)
    parser.add_argument('--num_images_visualize', type=int, default=10)
    parser.add_argument('--num_variables_visualize', type=int, default=6)
    parser.add_argument('--num_temperatures_visualize', type=int, default=3)
    parser.add_argument('--max_iters', type=int, default=3125)                # number of maximum iterations
    return parser
