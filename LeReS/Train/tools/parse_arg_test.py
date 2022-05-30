from tools.parse_arg_base import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='test flag')
        parser.add_argument('--phase_anno', type=str, default='test', help='eigen/eigen_test, Annotations file name')
        return parser
