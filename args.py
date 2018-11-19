import argparse

class ArgsConfig(object):
    """Docstring for ArgsConfig. """
    def __init__(self, model_names):
        self._model_names = model_names

        self.parser = argparse.ArgumentParser(description='Pytorch CNN')
        self.subparsers = self.parser.add_subparsers(title='running mode',\
                help='running mode')
        self.subparsers.required = True

        self.train = self.subparsers.add_parser('train', help='train mode help')
        self.train.set_defaults(mode='train')
        self.val   = self.subparsers.add_parser('val',   help='validate mode help')
        self.val.set_defaults(mode='val')
        self.test  = self.subparsers.add_parser('test',  help='test mode help')
        self.test.set_defaults(mode='test')

        self.train_args()
        self.val_args()
        self.test_args()
        self.commom_args()

        self.args = self.parser.parse_args()
        
    def train_args(self):
        parser = self.train
        parser.add_argument('--epochs', metavar='',\
                default=90, type=int,\
                help='number of total epochs to run (default: 90)')
        parser.add_argument('--start-epoch', metavar='',\
                default=0, type=int,\
                help='start epoch (default: 0)')
        parser.add_argument('--lr', metavar='',\
                default=0.01, type=float,\
                help='initial learning rate (default: 0.01)')
        parser.add_argument('--lr-decay', metavar='',\
                default=0.1, type=float,\
                help='learning rate decay rate (default: 0.1)')
        parser.add_argument('--lr-decay-freq', metavar='',\
                default=30, type=float,\
                help='learning rate decay frequency (epochs) (default: 30)')
        parser.add_argument('--momentum', metavar='',\
                default=0.9, type=float,\
                help='momentum param (default: 0.9)')
        parser.add_argument('--wd', metavar='',\
                default=1e-4, type=float,\
                help='weight decay (default: 1e-4)')
        parser.add_argument('--print-freq', default=10, type=int,
                metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--val-freq', metavar='',\
                default=5, type=int,\
                help='save model frequency (epochs) (default: 5)')
        parser.add_argument('--pretrained', action='store_true',\
                help='use pre-trained model')
        parser.add_argument('--logdir', default='train_log', type=str, 
                help='logdir for tensorboard')

    def val_args(self):
        parser = self.val

    def test_args(self):
        parser = self.test

    def commom_args(self):
        for parser in self.subparsers.choices.values():
            parser.add_argument('--seed', metavar='',\
                    default=233, type=int)
            parser.add_argument('-a', '--arch', metavar='',\
                    default='resnet18', choices=self._model_names,\
                    help='{}: {} (default: resnet18)'.format(\
                    'model architecture', ' | '.join(self._model_names)))
            parser.add_argument('-j', '--workers', metavar='',\
                    default=0, type=int,\
                    help='number of data loaders (default: 0)')
            parser.add_argument('-b', '--batch-size', metavar='',
                    default=64, type=int,
                    help='batch size (default: 64)')
            parser.add_argument('--resume', metavar='PATH',\
                    default=None, type=str, 
                    help='path of checkpoint (default: None)')
            parser.add_argument('--cuda', action='store_true',
                    help='use GPU')
            parser.add_argument('--gpu', default=None, help='GPU ID')

# }}}
