from tools.parse_arg_base import BaseOptions

class ValOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='val', help='Validation flag')
        parser.add_argument('--phase_anno', type=str, default='val', help='Annotations file name')
        return parser
