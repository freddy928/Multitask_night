import os
from yacs.config import CfgNode as CN


_C = CN()

_C.DATA_DIRECTORY_TARGET = '/root/autodl-tmp/Dark_Zurich/rgb_anon'
_C.DATA_LIST_PATH_TARGET = '/root/autodl-tmp/Dark_Zurich/corresp/train/night/zurich_dn_pair_train.csv'
_C.INPUT_SIZE_TARGET = 640
_C.DATA_SCALE_FACTOR = 0.25
_C.DATA_ROT_FACTOR = 10
_C.DATA_TRANSLATE = 0.1
_C.DATA_SHEAR = 0.0

_C.LRDISCR= 0.0001
_C.BATCH_SIZE= 8     # batch size
_C.MAX_EPOCH = 60

_C.POWER = 0.9