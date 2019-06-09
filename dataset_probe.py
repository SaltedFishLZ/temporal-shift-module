# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

# from tensorboardX import SummaryWriter

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)


    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5


    val_dataset = TSNDataSet(
                    args.root_path, args.val_list, num_segments=args.num_segments,
                    new_length=data_length,
                    modality=args.modality,
                    image_tmpl=prefix,
                    random_shift=False,
                    transform=torchvision.transforms.Compose([
                        GroupScale(int(256)),
                        GroupCenterCrop(224),
                        Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                        ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                        normalize,
                   ]),
                   )

    print(val_dataset.video_list)

if __name__ == '__main__':
    main()
