import os
import tensorflow as tf
import numpy as np
import random

# from util import wsort,readCSV
from keras.preprocessing import image
import keras
from keras import backend as K
from pathlib import Path
from keras.models import Model
from keras_vggface import utils
from keras.layers import Flatten,Dense,Input,Lambda,Dropout, Conv3D,Permute
from tensorflow import optimizers
from msn69 import MSN
#import psutil
import sys

os.environ['CUDA_VISIBLE_DEVICES']='0'

NUMBER_CLASSES = 8631
DROPOUT_HOLDER = 0.5
WEIGHT_OF_LOSS_WEIGHT = 7e-7
_SAVER_MAX_TO_KEEP = 10
_MOMENTUM = 0.9
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
BATCH = 20
NUM_RGB_CHANNELS = 3
NUM_FRAMES = 16
CHANNELS = 3
_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}


directory = 'D:\\Depression\\processed_images\\'
label_dir20220117 = 'D:\\Depression\\labels\\'
train_dirl = ['train/']
label_dirl = ['Frame_Labels/PSPI/']
train_common_dirl = ['042-ll042/', '043-jh043/','047-jl047/','048-aa048/','049-bm049/',
              '052-dr052/','059-fn059/','064-ak064/','066-mg066/','080-bn080/',
              '092-ch092/','095-tv095/','096-bg096/','097-gf097/','101-mg101/',
              '103-jk103/','106-nm106/','107-hs107/','108-th108/','109-ib109/']
test_common_dirl = ['115-jy115/', '120-kz120/', '121-vw121/', '123-jh123/', '124-dn124/']
test_dirl = ['test']
dirLabels = [
    'Frame_Labels/PSPI/042-ll042/ll042t1aaaff']

dirDevelopment = directory + 'cropImages/Development/'
dirDev = ['Freeform/', 'Northwind/']

dirTesting = directory + 'cropImages/Testing/'
dirTest = ['Freeform/', 'Northwind/']
dirLabelsTest = '/DEPRESSION_DATASET/Depression/AVEC2014/AVEC2014_Labels_Testset/Testing/DepressionLabels/'


def val_generator():
	while True:
		X=np.zeros((BATCH,NUM_FRAMES,112,112,3))
		Y=[0,1,2,3,4,5,6,7,5,4,3,2,1,5,2,0,1,2,1,0]
		images = []
		modality = np.random.randint(2,size=BATCH)
		usuario = np.random.randint(50,size=BATCH)
		for m in range(BATCH):
			# users=os.listdir(dirDevelopment+dirDev[modality[m]])
			for p in Path(directory + test_dirl[0] + test_common_dirl[m]).iterdir():
				for s in p.rglob('*.png'):
					# yield s
					images.append(s)
			numImages=len(images)

			imagens = np.random.randint(numImages)
			indice=0
			for j in range(imagens,imagens+128,8):
				imagem = image.load_img(images[(j) % numImages], target_size=(112, 112))
				imagem=image.img_to_array(imagem)
				# here you put your function to subtract the mean of vggface2 dataset
				imga = utils.preprocess_input(imagem,version=2) #subtract the mean of vggface dataset
				X[m,indice,:,:,:]=imga
				indice=indice+1
		# 	for p in Path(directory + label_dirl[0] + test_common_dirl[m]).iterdir():
		# 		for s in p.rglob('*.txt'):
		# 			with open(s, "r") as f:  # 打开文件
		# 				label = f.read()  # 读取文件
		# 				Y.append(float(label))
		# 	Y.append(label)
		#
		# Y=np.array(Y)
		yield X,Y

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


#一次生成batch大小的数据
def generator():
	while True:

		modality = np.random.randint(20, size=BATCH)  # 返回24个0-4的整数
		usuario = np.random.randint(50, size=BATCH)  # 返回24个0-50的整数
		# BATCH = 24
		X = np.zeros((BATCH, NUM_FRAMES, 112, 112, 3))


		Y2 = []
		images = []
		for i in range(BATCH):
			# users = os.listdir(directory + dirl[0])  # 训练文件夹四选一
			for p in Path(directory + train_dirl[0] + train_common_dirl[i]).iterdir():
				for s in p.rglob('*.png'):
					# yield s
					images.append(s)
			# Y = []
			# print(images)
			# images = get_img_file(directory+dirl[0]+'/')
			# print(images)
			numImages = len(images)

			imagens = np.random.randint(numImages)
			indice = 0

			valor = np.random.random()
			if valor < 0.25:
				flagFlip = 1
			elif valor < 0.50 and valor >= 0.25:
				flagFlip = 2
			elif valor < 0.75 and valor >= 0.50:
				flagFlip = 3
			else:
				flagFlip = 4
			Y = []
			for j in range(imagens, imagens + 128, 8):  # start,stop,step
				imagem = image.load_img(images[(j) % numImages], target_size=(112, 112))
				# print(images[(j) % numImages])
				strdir1, strdir2, strdir3, subdir = str(images[(j) % numImages]).split('\\', 3)
				label_file = label_dir20220117+subdir[:-4]+'_facs.txt'
				with open(label_file, "r") as f:  # 打开文件
					label = f.read()  # 读取文件
					Y.append(float(label))
				imagem = image.img_to_array(imagem)
				# here you put your function to subtract the mean of vggface2 dataset
				imga = utils.preprocess_input(imagem, version=2)  # subtract the mean of vggface dataset

				if flagFlip == 1:
					X[i, indice, :, :, :] = np.flip(imga, axis=1)
				elif flagFlip == 2:
					X[i, indice, :, :, :] = image.apply_affine_transform(imga, theta=30, channel_axis=2,
																		 fill_mode='nearest', cval=0., order=1)
				elif flagFlip == 3:
					X[i, indice, :, :, :] = np.flip(imga, axis=0)
				else:
					X[i, indice, :, :, :] = imga

				indice = indice + 1
			# labels = get_label_file(directory + train_dirl[0] + common_dirl[i])
			#
			# for l in range(len(labels)):
			#     with open(labels[l], "r") as f:  # 打开文件
			#         data = f.read()  # 读取文件
			#         if float(data) != 0.0:
			#             print(float(data))


			# for p in Path(directory + label_dirl[0] + train_common_dirl[i]).iterdir():
			# 	for s in p.rglob('*.txt'):
			# 		for i in range(20):
			# 			with open(s, "r") as f:  # 打开文件
			# 				label = f.read()  # 读取文件
			# 				Y.append(float(label))
		# sets = dirl[modality[i]].split('/')[1]
		# sets = 'Training'
		# # You can train the model using Training and Development sets
		# if sets == 'Training':
		# label = readCSV(dirLabels[0] + '/' + users[usuario[i]] + '_Depression.csv')
		# # else:
		# #     label = readCSV(dirLabels[1] + '/' + users[usuario[i]] + '_Depression.csv')
		# Y.append(label)
		print(Y)
		Y=np.array(Y)
		
		yield X,Y


if __name__ == '__main__':
	
	np.random.seed(42)

	rgb_model = MSN(input_shape=(NUM_FRAMES,FRAME_HEIGHT,FRAME_WIDTH,NUM_RGB_CHANNELS),classes=NUMBER_CLASSES)

	last_layer = rgb_model.get_layer('flatten').output
	
	#--FC Layer
	hidden1 = Dense(512,activation='relu',name='hidden1')(last_layer)
	hidden1 = Dropout(0.5)(hidden1)
	
	#--FC Layer
	hidden2 = Dense(512,activation='relu',name='hidden2')(hidden1)
	hidden2 = Dropout(0.5)(hidden2)

	#--Regression Layer
	out = Dense(1,activation='linear',name='classifier')(hidden2)

	custom_vgg_model = Model(rgb_model.input,out)
	adam = optimizers.Adam(lr=0.0001, decay=0.0005)
	custom_vgg_model.compile(loss='mse',optimizer=adam)
	custom_vgg_model.fit(generator(),steps_per_epoch=50,validation_data=val_generator(),validation_steps=10,epochs=5,verbose=1)
	#batch_size = 数据集大小/steps_per_epoch
	#--Here you read the label


	# Y=readCSV(dirLabelsTest+'_Depression.csv')
	# buf=0
	# numberOfFrames = NUM_FRAMES #
	# while (buf < numberOfFrames):
	# 	#Insert here the directory of the images of test set
	# 	imagem = image.load_img(dirTesting,target_size=(FRAME_HEIGHT,FRAME_WIDTH))
	# 	imagem = image.img_to_array(imagem)
	# 	#Subtract the mean of VGGFace2 dataset
	# 	#---put your function here
	# 	imga = utils.preprocess_input(imagem,version=2) #here it is the mean value of VGGFace dataset///////RESNET50?
	# 	X.append(imga)
	# X = np.array(X)
	# X = np.expand_dims(X,axis=0)
	# prediction = custom_vgg_model.predict(X)
