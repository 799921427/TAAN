
# encoding: utf-8

from __future__ import print_function, absolute_import
import os.path as osp
import os
import numpy as np
import torch
from torch import nn
import scipy.io as sio
from reid import models
from reid.models.newresnet import *
from reid.utils.serialization import load_checkpoint
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from reid.utils.data import transforms as T

img_to_tensor = transforms.ToTensor()

features = 2048

model_path = './0.5_rgb_0.8_ir/'
model = models.create('two_pipe', num_classes = 395, num_features= features, attention_mode = 1)
model_s = models.create('two_pipe', num_classes = 395, num_features = features, attention_mode = 1)
model_ir = models.create('two_pipe', num_classes = 395, num_features = features, attention_mode = 1)

checkpoint_I = load_checkpoint(model_path + 'model_best.pth.tar')
checkpoint_s = load_checkpoint(model_path + 's_model_best.pth.tar')
checkpoint_ir = load_checkpoint(model_path + 'ir_model_best.pth.tar')

model.load_state_dict(checkpoint_I['model'])
model_s.load_state_dict(checkpoint_s['model'])
model_ir.load_state_dict(checkpoint_ir['model'])

model = model.cuda()
model_s = model_s.cuda()
model_ir = model_ir.cuda()
model.eval()
model_s.eval()
model_ir.eval()

normalizer = transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])

test_transformer = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    normalizer,
])

print('model instaled \n')

path_list = ['./data/sysu/SYSU-MM01/cam1/', './data/sysu/SYSU-MM01/cam2/', './data/sysu/SYSU-MM01/cam3/', './data/sysu/SYSU-MM01/cam4/', './data/sysu/SYSU-MM01/cam5/', './data/sysu/SYSU-MM01/cam6/']
pic_num = [333, 333, 533, 533, 533, 333]
for index, path in enumerate(path_list):
    print(index)
    cams = torch.LongTensor([index])
    sub = ((cams == 2).long() + (cams == 5).long()).cuda()
    count = 1
    array_list = []
    person_id_list = []
    dict_person = {}
    tot_num = pic_num[index]
    array_list_to_array = [[] for _ in range(tot_num)]
    for fpathe, dirs, fs in os.walk(path):
        person_id = fpathe.split('/')[-1]
        if(person_id == ''):
            continue
        cam_name = fpathe[-9: -5]
        fs.sort()
        person_id_list.append(person_id)
        dict_person[person_id] = fs
    person_id_list.sort()
    for person in person_id_list:
        temp_list = []
        for imagename in dict_person[person]:
            filename = path + str(person) + '/' + imagename
            img=Image.open(filename)
            img=test_transformer(img)
            img=img.view([1,3,256,128])
            img=img.cuda()
            _, result_y = model(img)
            # result_y = torch.nn.functional.normalize(result_y, dim=1, p=2)
            # print(result_y.size())
            # if index == 2 or index == 5:
            #     _, result_x = model_ir(img)
               # result_x = torch.nn.functional.normalize(result_x, dim=1, p=2)
            # else:
            #     _, result_x = model_s(img)
               # result_x = torch.nn.functional.normalize(result_x, dim=1, p=2)
            # result_y = torch.cat([result_y, result_x], 1)
            # result_y = (result_y + result_x) / 2
            result_y = torch.nn.functional.normalize(result_y, dim=1, p=2)
            # print(result_y.size())
            result_y = result_y.view(-1, features)
            # print(result_y.size())
            result_y = result_y.squeeze()
            result_npy = result_y.data.cpu().numpy()
            result_npy = result_npy.astype('double')
            temp_list.append(result_npy)
        temp_array = np.array(temp_list)
        array_list_to_array[int(person)-1] = temp_array
    array_list_to_array = np.array(array_list_to_array)
    sio.savemat(model_path + cam_name + '.mat', {'feature_test':array_list_to_array})

print("end")
