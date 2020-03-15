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


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
img_to_tensor = transforms.ToTensor()  

features = 2048

model = models.create('two_pipe', num_classes=395, num_features=features, attention_mode=1)

checkpoint_I = load_checkpoint('/home/fan/cross_reid_new/tri_256_128_2000epoch_instance_4_64_0.9_ir_1.0_rgb/model_best.pth.tar')
# model_dict = model.state_dict()
# state_dict = {k: v for k, v in checkpoint_I.items() if k in model_dict.keys()}
# model_dict.update(state_dict)
# # for key in checkpoint_I.keys():
# # 	for item in checkpoint_I[key].keys():
# # 		print(item)
model.load_state_dict(checkpoint_I['model'])

# model.model.fc = nn.Sequential()
# model.classifier = nn.Sequential()
# 
model = model.cuda()
model.eval()
# model_generator_IR.eval()

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

test_transformer = T.Compose([
	T.Resize((256,128)),
	T.ToTensor(),
	normalizer,
])

scaler = transforms.Scale((256,128))


print('model initialed \n')

# content = sio.loadmat('./test.mat')
# print (content)
cam_name = ''
# path_list = ['/home/steam/Workspace/SYSU-MM01/cam_gray/cam1/','/home/steam/Workspace/SYSU-MM01/cam_gray/cam2/','/home/steam/Workspace/SYSU-MM01/cam_gray/cam3/', \
#          '/home/steam/Workspace/SYSU-MM01/cam_gray/cam4/','/home/steam/Workspace/SYSU-MM01/cam_gray/cam5/','/home/steam/Workspace/SYSU-MM01/cam_gray/cam6/']

path_list = ['./data/sysu/SYSU-MM01/cam1/','./data/sysu/SYSU-MM01/cam2/','./data/sysu/SYSU-MM01/cam3/', \
             './data/sysu/SYSU-MM01/cam4/','./data/sysu/SYSU-MM01/cam5/','./data/sysu/SYSU-MM01/cam6/']
# path_list = ['/home/steam/Workspace/SYSU-MM01/cam/cam6/']
pic_num = [333,333,533,533,533,333]
for index, path in enumerate(path_list):

	print(index)
	cams = torch.LongTensor([index])
	sub = ((cams == 2).long() + (cams == 5).long()).cuda()
	# print(sub)
	# if(index == 2 or 5):
	# 	model = model_generator_IR
	# else:
	# 	model = model_generator_I
	count = 1
	array_list = []
	person_id_list = []
	dict_person = {}
	tot_num = pic_num[index]
	array_list_to_array = [[] for _ in range(tot_num)] 
	#print(path)
	for fpathe,dirs,fs in os.walk(path):
		# print (dirs)
		# print (fpathe[-9:-5])
		person_id = fpathe.split('/')[-1]
		# print (person_id)
		if(person_id == ''):
			continue
		cam_name = fpathe[-9:-5]
		# print (cam_name)
		# print (fs)
		fs.sort()
		person_id_list.append(person_id)
		dict_person[person_id] = fs
	person_id_list.sort()
	#print(person_id_list)
	for person in person_id_list:
		temp_list = []
		for imagename in dict_person[person]:
			if imagename.__len__() > 8 : continue

			filename = path + str(person) + '/' + imagename
			if filename.__len__() > 40:
				print (filename)
			img=Image.open(filename)  
			img=test_transformer(img)
			# img=img.resize((128,256))
			# img = img_to_tensor(img)
			# img = normalize(img)
			# img = Variable(img.unsqueeze(0))
			# img = img.cuda()   
			#_, result_y=model(img)
			img=img.view([1,3,256,128])
			img=img.cuda()
			# print(img.size())
			_, result_y = model(img)
			# print(result_x)
			#print(result_y)
			result_y = result_y.view(-1, features)
			result_y = result_y.squeeze()
			# result_y = result_y.view(-1, 512)
			result_npy=result_y.data.cpu().numpy()
			result_npy=result_npy.astype('double')
			temp_list.append(result_npy)
		temp_array = np.array(temp_list)
		#normalize
		temp_array = torch.nn.functional.normalize(torch.from_numpy(temp_array), dim=1, p=2)
		array_list_to_array[int(person) - 1] = np.array(temp_array)
	#print(array_list_to_array.size())
	array_list_to_array = np.array(array_list_to_array)
	# if path == '/home/steam/Workspace/SYSU-MM01/cam_gray/cam6/':
	# if path == '/home/steam/Workspace/SYSU-MM01/cam/cam6/':
	#	array_list_to_array = array_list_to_array.reshape((-1, 20, 2048))
		# array_list_to_array = array_list_to_array.reshape((-1, 20, 512))

	sio.savemat(cam_name + '.mat', {'feature_test':array_list_to_array})

