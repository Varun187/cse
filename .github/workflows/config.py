
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch


from utils.attr_dict import AttrDict


__C = AttrDict()
cfg = __C
__C.EPOCH = 0
__C.CLASS_UNIFORM_PCT = 0.0
__C.BATCH_WEIGHTING = False

__C.BORDER_WINDOW = 1
__C.REDUCE_BORDER_EPOCH = -1
__C.STRICTBORDERCLASS = None


__C.DATASET = AttrDict()
__C.DATASET.CITYSCAPES_DIR = 'http://idd:insaan.iiit.ac.in/dataset/download/'
__C.DATASET.CITYSCAPES_AUG_DIR = 'http://idd:insaan.iiit.ac.in/dataset/download/'
__C.DATASET.MAPILLARY_DIR = 'http://idd:insaan.iiit.ac.in/dataset/download/'
__C.DATASET.KITTI_DIR = 'http://idd:insaan.iiit.ac.in/dataset/download/'
__C.DATASET.KITTI_AUG_DIR = 'http://idd:insaan.iiit.ac.in/dataset/download/'
__C.DATASET.CAMVID_DIR = 'http://idd:insaan.iiit.ac.in/dataset/download/'
__C.DATASET.CV_SPLITS = 3


__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
__C.MODEL.BNFUNC = None

def assert_and_infer_cfg(args, make_immutable=True, train_mode=True):


    if hasattr(args, 'syncbn') and args.syncbn:
        if args.apex:
            import apex
            __C.MODEL.BN = 'apex-syncnorm'
            __C.MODEL.BNFUNC = apex.parallel.SyncBatchNorm
        else:
            raise Exception('No Support for SyncBN without Apex')
    else:
        __C.MODEL.BNFUNC = torch.nn.BatchNorm2d
        print('Using regular batch norm')

    if not train_mode:
        cfg.immutable(True)
        return
    if args.class_uniform_pct:
        cfg.CLASS_UNIFORM_PCT = args.class_uniform_pct

    if args.batch_weighting:
        __C.BATCH_WEIGHTING = True

    if args.jointwtborder:
        if args.strict_bdr_cls != '':
            __C.STRICTBORDERCLASS = [int(i) for i in args.strict_bdr_cls.split(",")]
        if args.rlx_off_epoch > -1:
            __C.REDUCE_BORDER_EPOCH = args.rlx_off_epoch

    if make_immutable:
        cfg.immutable(True)
