import os
import tensorflow as tf
import numpy as np
import random

# from util import wsort,readCSV
from keras.preprocessing import image
import keras
from keras import backend as K

from keras.models import Model
from keras_vggface import utils
from keras.layers import Flatten, Dense, Input, Lambda, Dropout, Conv3D, Permute
from tensorflow import optimizers
from msn69 import MSN
# import psutil
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUMBER_CLASSES = 8631
DROPOUT_HOLDER = 0.5
WEIGHT_OF_LOSS_WEIGHT = 7e-7
_SAVER_MAX_TO_KEEP = 10
_MOMENTUM = 0.9
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
BATCH = 24
NUM_RGB_CHANNELS = 3
NUM_FRAMES = 16
CHANNELS = 3
_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}

directory = 'D:\\Depression\\MSN-master\\DEPRESSION_DATASET\\'

dirl = ['cropImages/Training/Freeform/', 'cropImages/Training/Northwind/', 'cropImages/Development/Freeform/',
        'cropImages/Development/Northwind/']

dirLabels = [
    'Frame_Labels/PSPI/042-ll042/ll042t1aaaff']

dirDevelopment = directory + 'cropImages/Development/'
dirDev = ['Freeform/', 'Northwind/']

dirTesting = directory + 'cropImages/Testing/'
dirTest = ['Freeform/', 'Northwind/']
dirLabelsTest = '/DEPRESSION_DATASET/Depression/AVEC2014/AVEC2014_Labels_Testset/Testing/DepressionLabels/'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist
def get_label_file(file_name):
    labellist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.txt')):
                labellist.append(os.path.join(parent, filename))
        return labellist

if __name__ == '__main__':

    modality = np.random.randint(4, size=BATCH)  # 返回24个0-4的整数
    usuario = np.random.randint(50, size=BATCH)  # 返回24个0-50的整数
    # BATCH = 24
    X = np.zeros((BATCH, NUM_FRAMES, 112, 112, 3))
    Y = []
    Y2 = []
    for i in range(BATCH):
        users = os.listdir(directory + dirl[modality[i]])  # 训练文件夹四选一
        images = get_img_file(directory+dirl[modality[i]]+users[0]+'/')
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

        for j in range(imagens, imagens + 128, 8):  # start,stop,step
            imagem = image.load_img(images[(j) % numImages], target_size=(112, 112))

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
            print(indice)
        # sets = dirl[modality[i]].split('/')[1]
        # sets = 'Training'
        # # You can train the model using Training and Development sets
        # if sets == 'Training':
            label = readCSV(dirLabels[0] + '/' + users[usuario[i]] + '_Depression.csv')
        # else:
        #     label = readCSV(dirLabels[1] + '/' + users[usuario[i]] + '_Depression.csv')
        Y.append(label)
    # label_dir = directory + dirLabels[0]
    #
    # labels = get_label_file(label_dir)
    # for l in range(len(labels)):
    #     with open(labels[l], "r") as f:  # 打开文件
    #         data = f.read()  # 读取文件
    #         if float(data) != 0.0 :
    #             print(float(data))