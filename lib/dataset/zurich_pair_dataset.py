import os.path as osp
import numpy as np
from torch.utils import data
import cv2
import random
import math
from lib.dataset.transforms import *
import torchvision.transforms as standard_transforms


def random_perspective_zurich(combination, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    img_d, img_n = combination
    height = img_d.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img_n.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img_n.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_n.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img_n = cv2.warpPerspective(img_n, M, dsize=(width, height), borderValue=(114, 114, 114))
            img_d = cv2.warpPerspective(img_d, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img_n = cv2.warpAffine(img_n, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            img_d = cv2.warpAffine(img_d, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    combination = (img_n, img_d)
    return combination


def letterbox(combination, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """Resize the input image and automatically padding to suitable shape :https://zhuanlan.zhihu.com/p/172121380"""
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    img_n, img_d = combination
    shape = img_n.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img_n = cv2.resize(img_n, new_unpad, interpolation=cv2.INTER_LINEAR)
        img_d = cv2.resize(img_d, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_n = cv2.copyMakeBorder(img_n, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    img_n = cv2.copyMakeBorder(img_d, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # add border
    # print(img.shape)

    combination = (img_n, img_d)
    return combination, ratio, (dw, dh)


class zurich_pair_DataSet(data.Dataset):
    def __init__(self, cfg, root, list_path, max_iters=None, set='val', joint_transform=None):
        self.cfg= cfg
        self.root = root
        self.list_path = list_path
        self.is_train= True if set=='train' else False
        self.inputsize= cfg.INPUT_SIZE_TARGET

        train_input_transform = []
        train_input_transform += [standard_transforms.ToTensor(),
                                  standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        self.target_transform = extended_transforms.MaskToTensor()
        self.transform = standard_transforms.Compose(train_input_transform)

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        for pair in self.img_ids:
            night, day = pair.split(",")
            img_night = osp.join(self.root, "%s" % (night)+"_rgb_anon.png")
            img_day = osp.join(self.root, "%s" % (day)+"_rgb_anon.png")
            self.files.append({
                "img_night": img_night,
                "img_day": img_day,
                "name": night+"_rgb_anon.png"
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image_n = cv2.imread(datafiles["img_night"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image_d = cv2.imread(datafiles["img_day"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = image_n.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image_n = cv2.resize(image_n, (int(w0 * r), int(h0 * r)), interpolation=interp)
            image_d = cv2.resize(image_d, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = image_d.shape[:2]

        (image_n, image_d), ratio, pad = letterbox((image_n, image_d), resized_shape, auto=True, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        if self.is_train:
            combination = (image_n, image_d)
            (image_n, image_d) = random_perspective_zurich(
                combination=combination,
                degrees=self.cfg.DATA_ROT_FACTOR,
                translate=self.cfg.DATA_TRANSLATE,
                scale=self.cfg.DATA_SCALE_FACTOR,
                shear=self.cfg.DATA_SHEAR
            )
        size = image_n.shape
        name = datafiles["name"]
        if self.transform is not None:
            image_n = self.transform(image_n)
            image_d = self.transform(image_d)

        return image_n, image_d, np.array(size), name

