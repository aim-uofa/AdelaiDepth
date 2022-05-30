import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--backbone', type=str, default='resnet50', help='Select backbone type, resnet50 or resnext101')
        parser.add_argument('--batchsize', type=int, default=2, help='Batch size')
        parser.add_argument('--base_lr', type=float, default=0.001, help='Initial learning rate')
        parser.add_argument('--load_ckpt', help='Checkpoint path to load')
        parser.add_argument('--resume', action='store_true', help='Resume to train')
        parser.add_argument('--epoch', default=50, type=int, help='Total training epochs')
        parser.add_argument('--dataset_list', default=None, nargs='+', help='The names of multiple datasets')
        parser.add_argument('--loss_mode', default='_vnl_ssil_ranking_', help='Select loss to supervise, joint or ranking')
        parser.add_argument('--lr_scheduler_multiepochs', default=[10, 25, 40], nargs='+', type=int, help='Learning rate scheduler step')

        parser.add_argument('--val_step', default=5000, type=int, help='Validation steps')
        parser.add_argument('--snapshot_iters', default=5000, type=int, help='Checkpoint save iters')
        parser.add_argument('--log_interval', default=10, type=int, help='Log print iters')
        parser.add_argument('--output_dir', type=str, default='./output', help='Output dir')
        parser.add_argument('--use_tfboard', action='store_true', help='Tensorboard to log training info')

        parser.add_argument('--dataroot', default='./datasets', required=True, help='Path to images')
        parser.add_argument('--dataset', default='multi', help='Dataset loader name')
        parser.add_argument('--scale_decoder_lr', type=float, default=1, help='Scale learning rate for the decoder')
        parser.add_argument('--thread', default=0, type=int, help='Thread for loading data')
        parser.add_argument('--start_step', default=0, type=int, help='Set start training steps')
        parser.add_argument('--sample_ratio_steps', default=10000, type=int, help='Step for increasing sample ratio')
        parser.add_argument('--sample_start_ratio', default=0.1, type=float, help='Start sample ratio')
        parser.add_argument('--local_rank', type=int, default=0, help='Rank ID for processes')
        parser.add_argument('--nnodes', type=int, default=1, help='Amount of nodes')
        parser.add_argument('--node_rank', type=int, default=0, help='Rank of current node')
        parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:22', help='URL specifying how to initialize the process group')

        # parser.add_argument('--optim', default='SGD', help='Select optimizer, SGD or Adam')
        # parser.add_argument('--start_epoch', default=0, type=int, help='Set training epochs')
        # parser.add_argument('--results_dir', type=str, default='./evaluation', help='Output dir')
        # parser.add_argument('--diff_loss_weight', default=1, type=int, help='Step for increasing sample ratio')

        self.initialized = True
        return parser

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        self.opt = parser.parse_args()

        return self.opt
def print_options(opt, logger=None):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    logger.info(message)