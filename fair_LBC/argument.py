import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Fairness')
    parser.add_argument('--data-dir', default='./data/',
                        help='data directory (default: ./data/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--device', default=0, type=int, help='cuda device number')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--modelpath', default=None)

    parser.add_argument('--evalset', default='all', choices=['all', 'train', 'test'])

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', '--epoch', default=30, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', default=4, type=int, help='mini batch size')
    parser.add_argument('--seed', default=0, type=int, help='seed for randomness') #
    parser.add_argument('--subject-seed', default=0, type=int, help='seed for subject id shuffle') #
    parser.add_argument('--date', default='dummy', type=str, help='experiment date')
    parser.add_argument('--data', default='mri', type=str, choices=['mri'],
                        help='data to train')

    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['SGD',
                                 'SGD_momentum_decay',
                                 'Adam'],
                        help='(default=%(default)s)')

    parser.add_argument('--parallel', default=False, action='store_true', help='data parallel')
    parser.add_argument('--num-workers', default=4, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')
    parser.add_argument('--weight-decay', default=0.001, type=float, help='weight decay')
    parser.add_argument('--milestones', nargs="+", default=[int(15),int(24)], type=int, help="for lr scheduler")
    parser.add_argument('--pos-weight', default=1., type=float)
    parser.add_argument('--with-mci', default=False, action='store_true', help='mci label added')
    parser.add_argument('--use-single-img', default=False, action='store_true', help='use only last visit image')

    parser.add_argument('--network', default='mobilenetv2', type=str, choices=['mobilenetv2','shufflenetv2','resnet10','resnet18'], help='network to train')
    parser.add_argument('--method', default='None', type=str, choices=['None','WCE','oversampling','LBC'], help='fairness method')

    parser.add_argument('--lbc-eta', default=1, type=float)
    parser.add_argument('--lbc-iteration', default=100, type=int, help='training iteration for label bias correcting method')
    parser.add_argument('--lbc-reweighting-target-criterion', default='eopp', type=str, choices=['eo','eopp'])
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.mode == 'eval' and args.modelpath is None:
        raise Exception('Model path to load is not specified!')

    return args
