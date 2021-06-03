

import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import glob
import warnings
import time
from cs_data_loader import *

from skimage import io, transform
import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule
#from model_mapping import rename_weight_for_head

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset
        parser.add_argument('--model', type=str, default='encnet',
                            help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='ade20k',
                            help='dataset name (default: pascal12)')
        parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=520,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=480,
                            help='crop image size')
        parser.add_argument('--train-split', type=str, default='train',
                            help='dataset train split (default: train)')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--se-loss', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2,
                            help='SE-loss weight (default: 0.2)')
        parser.add_argument('--batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--verify', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating mIoU')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--acc-bn', action='store_true', default= False,
                            help='Re-accumulate BN statistics')
        parser.add_argument('--test-val', action='store_true', default= False,
                            help='generate masks on val set')
        parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        # multi grid dilation option
        parser.add_argument("--multi-grid", action="store_true", default=False,
                            help="use multi grid dilation policy")
        parser.add_argument('--multi-dilation', nargs='+', type=int, default=None,
                            help="multi grid dilation list")
        parser.add_argument('--os', type=int, default=8,
                            help='output stride default:8')
        parser.add_argument('--no-deepstem', action="store_true", default=False,
                    help='backbone without deepstem')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args

time_total = 0.
batch_size = 1
img_size = (256, 512)
device = 'cuda:0'

def test(args):
    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    test_set = CSDataset('test2.csv', transform=transforms.Compose([Rescale(img_size), CSToTensor()]))


    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}

    test_data = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

    # model
    pretrained = args.resume is None and args.verify is None

    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=pretrained)
        model.base_size = args.base_size
        model.crop_size = args.crop_size

    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, aux = args.aux,
                                       se_loss=args.se_loss,
                                       norm_layer=torch.nn.BatchNorm2d if args.acc_bn else SyncBatchNorm,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       multi_grid=args.multi_grid,
                                       multi_dilation=args.multi_dilation,
                                       os=args.os,
                                       no_deepstem=args.no_deepstem)

    # resuming checkpoint
    #print("=={}".format(os.path.isfile(args.resume)))
    if torch.cuda.is_available():
        model.cuda()

    checkpoint = torch.load('checkpoint.pth.tar')
    weights = checkpoint['state_dict']
    model.load_state_dict(weights)


    print(model)



    #scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \

    model.eval()
    #metric = utils.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)

    for i, temp_batch in enumerate(tbar):
        temp_rgb = temp_batch['rgb'].float().to(device).cuda()
        temp_foregd = temp_batch['foregd'].long().squeeze(1).to(device).cuda()
        temp_partial_bkgd = temp_batch['partial_bkgd'].float().to(device).cuda()

        with torch.set_grad_enabled(False):
            # pre-processing the input and target on the fly
            foregd_idx = (temp_foregd.float() > 0.5).float()

            time_start = time.time()

            outputs1,outputs2,outputs3,outputs4,outputs5,outputs6 = model(temp_rgb)








            fore_middle_msk = F.interpolate((outputs1 > 0.5).float(), scale_factor=1).int()
            fore_middle_msk = fore_middle_msk.to('cpu').numpy().squeeze()
            fore_middle_msk_color = fore_middle_msk * 255

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave('outdir1/color'+str(i)+'.png', fore_middle_msk.astype(np.uint8))
            io.imsave('outdir/color'+str(i)+'.png', fore_middle_msk_color.astype(np.uint8))


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)
