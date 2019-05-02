# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Notice that this file has been modified to support ensemble testing

import argparse
import time
import copy
import pickle

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F

__ddebug__ = False

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default=25)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')

args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def aggregate_corrects(correct, maxk=1):
    '''
    - correct : [K][Sample]
    - return : [K][Sample]
    '''
    res = np.empty(correct.shape)
    for _k in range(maxk):
        hit = np.array(correct.shape[1] * [0,])
        k = _k + 1
        for _i in range(k):
            hit = np.logical_or(hit, correct[_i])
        res[_k] = (hit)
    return(res)

def hits_to_acc(hits):
    '''
    - hits : [K][Sample]
    - return : list[K]
    '''
    res = []
    for hit_k in hits:
        correct_k = hit_k.sum(0)
        hit_rate = correct_k * (100.0 / hits.shape[1])
        # print("Hit Rate: {}".format(hit_rate))
        res.append(hit_rate)
    return(res)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.numpy()  
    
    hits = aggregate_corrects(correct, maxk)
    accs = hits_to_acc(hits)
    
    # assemble to a list
    res = []
    for k in topk:
        res.append(accs[k-1])

    return(res)

class accuracy_multichance(object):
    
    def __init__(self, topk=(1,)):
        self.inited = False
        self.topk = copy.deepcopy(topk)
        self.maxk = max(topk)
        # historical corrects for independent predictions
        # list[Chance] -> ndarray[Sample]
        self.id_corrects = []
        # historical corrects for mc predictions
        self.mc_corrects = []
        # historical accuracy for independent predictions
        # list[Chance] -> ndarray[K]
        self.id_accuracies = []
        # historical accuracy for multi-chance prediction
        self.mc_accuracies = []


    def get_targets(self, target):
        # targets is a list of all samples
        # [Sample]
        self.target = copy.deepcopy(target)
        self.inited = True

    def __call__(self, output):
        '''
        - outputs : outputs of all batches in a single test
        [Batch * Sample Per Batch]
        '''

        # temporary correct arrary for this 1 evaluation
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        id_correct = pred.eq(self.target.view(1, -1).expand_as(pred))
        id_correct = id_correct.numpy()  

        # log data
        self.id_corrects.append(id_correct)

        # logical or
        if (len(self.mc_corrects) > 0):
            mc_correct = np.logical_or(
                self.mc_corrects[-1], id_correct)
        else:
            mc_correct = id_correct
        self.mc_corrects.append(mc_correct)

        id_hits = aggregate_corrects(id_correct, self.maxk)
        mc_hits = aggregate_corrects(mc_correct, self.maxk)

        id_accs = hits_to_acc(id_hits)
        mc_accs = hits_to_acc(mc_hits)

        self.id_accuracies.append(id_accs)
        self.mc_accuracies.append(mc_accs)

        res = []
        for k in self.topk:
            res.append(mc_accs[k-1])
        return(res)



def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None



def eval_video(video_data, net, this_test_segments, modality):
    net.eval()
    with torch.no_grad():
        i, data, label = video_data
        batch_size = label.numel()
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if modality == 'RGB':
            length = 3
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+ modality)

        data_in = data.view(-1, length, data.size(2), data.size(3))
        if is_shift:
            data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
        rst = net(data_in)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        rst = rst.data.cpu().numpy().copy()

        if net.module.is_shift:
            rst = rst.reshape(batch_size, num_class)
        else:
            rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

        return i, rst, label



# ---------------------------------------------
#               Arg Parse
# ---------------------------------------------

weights_list = args.weights.split(',')
test_segments_list = [int(s) for s in args.test_segments.split(',')]
assert len(weights_list) == len(test_segments_list)
if args.coeff is None:
    coeff_list = [1] * len(weights_list)
else:
    coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
    test_file_list = args.test_list.split(',')
else:
    test_file_list = [None] * len(weights_list)

data_iter_list = []
net_list = []
modality_list = []


this_weights = weights_list[0]
this_test_segments = test_segments_list[0]
test_file = test_file_list[0]
is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)

if 'RGB' in this_weights:
    modality = 'RGB'
else:
    modality = 'Flow'
this_arch = this_weights.split('TSM_')[1].split('_')[2]
modality_list.append(modality)
num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                        modality)
print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
net = TSN(num_class, this_test_segments if is_shift else 1, modality,
          base_model=this_arch,
          consensus_type=args.crop_fusion_type,
          img_feature_dim=args.img_feature_dim,
          pretrain=args.pretrain,
          is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
          non_local='_nl' in this_weights,
          )
if 'tpool' in this_weights:
    from ops.temporal_shift import make_temporal_pool
    make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel
checkpoint = torch.load(this_weights)
checkpoint = checkpoint['state_dict']
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)
net.load_state_dict(base_dict)
input_size = net.scale_size if args.full_res else net.input_size
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(input_size),
    ])
elif args.test_crops == 3:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupFullResSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 5:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size, flip=False)
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))
data_loader = torch.utils.data.DataLoader(
        TSNDataSet(root_path, test_file if test_file is not None else val_list, num_segments=this_test_segments,
                   new_length=1 if modality == "RGB" else 5,
                   modality=modality,
                   image_tmpl=prefix,
                   test_mode=True,
                   remove_missing=len(weights_list) == 1,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                       GroupNormalize(net.input_mean, net.input_std),
                   ]),
                   dense_sample=args.dense_sample,
                   random_sample=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
)

net = torch.nn.DataParallel(net, device_ids=args.gpus).cuda()
net.eval()

# ---------------------------------------------


proc_start_time = time.time()


targets_of_allsamples = np.zeros((0))
predics_of_allsamples = np.zeros((0, num_class))


mcacc_meter = accuracy_multichance(topk=(1,5))



CHANCES = 20

for chance in range(CHANCES):
    print("Chance [{}]".format(chance))
    
    targets_of_allsamples = np.zeros((0))
    predics_of_allsamples = np.zeros((0, num_class))
    
    i = 0
    for data_label_pairs in data_loader:
        with torch.no_grad():
            data, label = data_label_pairs
            rst = eval_video((i, data, label), net, this_test_segments, modality)
            predict = rst[1]

            predics_of_allsamples = np.append(predics_of_allsamples, predict, axis=0)
            targets_of_allsamples = np.append(targets_of_allsamples, label, axis=0)

            # if (i % 20 == 0):
            #     print("[{}]: Prediction Blob Shape ".format(i), predics_of_allsamples.shape)
            i += 1

    targets_of_allsamples = (torch.Tensor(targets_of_allsamples)).long()
    predics_of_allsamples = torch.Tensor(predics_of_allsamples)

    if (__ddebug__):
        predicts_list = predics_of_allsamples.tolist()
        fname = "chance-{}.pred".format(chance)
        f = open(fname, "w")
        for r in predicts_list:
            f.write(str(r))
        f.close()


    if (False == mcacc_meter.inited):
        mcacc_meter.get_targets(targets_of_allsamples)

    running_accurary = (mcacc_meter(predics_of_allsamples))
        
    print("ID Acc : {}".format(mcacc_meter.id_accuracies[-1]))
    print("MC Acc : {}".format(mcacc_meter.mc_accuracies[-1]))




# dump results
weight_name = this_weights.split('/')[-1]

fname = weight_name.split('.')[0] + ".idacc"
f = open(fname, "wb")
pickle.dump(mcacc_meter.id_accuracies, f)
f.close()

fname = weight_name.split('.')[0] + ".mcacc"
f = open(fname, "wb")
pickle.dump(mcacc_meter.mc_accuracies, f)
f.close()
