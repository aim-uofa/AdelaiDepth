from tools.parse_arg_base import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='train', help='Training flag')
        parser.add_argument('--phase_anno', type=str, default='train', help='Annotations file name')
        self.isTrain = True
        return parser
