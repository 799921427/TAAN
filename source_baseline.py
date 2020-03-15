from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from reid import datasets
from reid import models
import torchvision
from reid.dist_metric import DistanceMetric
from reid.loss.CrossTriplet import CrossTriplet as TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import CamRandomIdentitySampler as RandomIdentitySampler
from reid.utils.data.sampler import CamSampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from utlis import RandomErasing, WarmupMultiStepLR,CrossEntropyLabelSmooth

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    model = models.create("baseline_wo_D", num_classes=295, num_features=2048, attention_mode=1)
    # print(model)
    checkpoint_s = load_checkpoint('/home/fan/cross_reid/sysu_logs_5000_mode1/model_best.pth.tar')

    model_dict = model.state_dict()
    state_dict = {k:v for k,v in checkpoint_s.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    # model = model.cuda()
    model.eval()
    img_path = '0001.jpg'
    img = cv2.imread(img_path)
    # img = mpimg.imread(img_path)
    # print(img.shape)

    size = (128, 256)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imshow("img", img)
    img = torchvision.transforms.functional.to_tensor(img)
    img = img.unsqueeze(0)
    # img.cuda()
    _, _, att_feat = model(img)

    plt.figure("image")

    for i, fea in enumerate(att_feat):
        plt.subplot(1,4,i+1)
        fea = fea.detach().numpy()
        img = fea[0, 0, :, :]
        fmin = img.min()
        fmax = img.max()
        img = ((img - fmin) / (fmax - fmin + 0.00001)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = img[..., ::-1]
        im = plt.imshow(img)

    # position = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    # plt.show()
    plt.show()
    print(img.shape)
    cv2.waitKey(0)
