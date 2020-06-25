# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:54:14 2020

@author: ENVY13
"""

from __future__ import division

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import random
import os
import glob
import scipy.ndimage.interpolation as inter
from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from sklearn.preprocessing import LabelBinarizer 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.optimizers import rmsprop
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing

class SBU_dataset():
    def __init__(self, dir):
        print ('loading data from:', dir)
        
        self.pose_paths = glob.glob(os.path.join(dir, 's*', '*','*','*.txt'))
        self.pose_paths.sort()

    def get_data(self):
        
        datapos = {} 
        datapath={}
        
        def read_txt(pose_path):
            a = pd.read_csv(pose_path,header=None).T
            a = a[1:]
            return a.values
        
        for i in range(1,9):
            datapos[i] = []
            datapath[i] = []

        for pose_path in self.pose_paths:
            pose = read_txt(pose_path)         
            datapos[int(pose_path.split('\\')[7])].append(pose)
            datapath[int(pose_path.split('\\')[7])].append(pose_path)

        return datapos,datapath

def read_skeleton_file(file_path):
    skeleton_file = open(file_path)  
    lines = skeleton_file.readlines()  
    frame_count = len(lines) 
    pos = np.zeros((frame_count, 30, 3)) 
    for i in range(len(lines)):  
        pos[i] = np.array(list(map(float, lines[i].split(',')[1:]))).reshape(30, 3)
    return pos

    
def one_obj(frame_l=28, joint_n=30, joint_d=3):
    input_joints = Input(name='joints', shape=(frame_l, joint_n, joint_d))
    input_joints_diff = Input(name='joints_diff', shape=(frame_l, joint_n, joint_d))

    # 考虑到数据集的大小，我们相应地简化了图3中的网络架构。
    # 具体来说，conv1，conv2，conv3，conv5，conv6和fc7的输出通道
    # 分别减少为32、16、16、32、64和64。
    # 并删除co​​nv4层。此外，所有输入序列的长度标准化为16帧而不是32帧。

    # padding: 卷积方式。可选方法有"SAME"、"VALID"。
    # 其中，VALID方式以（1,1）作为卷积的左上角起始点。
    # SAME则以（1,1）作为卷积的中心起始点。
    # 即，SAME方法得到的卷积矩阵要大于VALID。

    ##########branch 1##############
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_joints)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(filters=16, kernel_size=(3, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
   
    # 将x的123维度转换为132
    x = Permute((1, 3, 2))(x)
    
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
     ##########branch 2##############Temporal difference
    x_d = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_joints_diff)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)

    x_d = Conv2D(filters=16, kernel_size=(3, 1), padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
   
    x_d = Permute((1, 3, 2))(x_d)

    x_d = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x_d)
    x_d = BatchNormalization()(x_d)
    x_d = LeakyReLU()(x_d)
    
    ##########branch 2##############

    x = concatenate([x, x_d], axis=-1)
    
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Dropout(0.3)(x)
    
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Dropout(0.2)(x)
    
    x = Dense(64,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(8,activation='softmax')(x)
    
    #model = Model(input_joints, x)
    
    ##########branch 1##############

   

    
    # 在多输入或多输出模型的情况下，你也可以使用列表：
    #model = Model(inputs=[a1, a2], outputs=[b1, b3, b3])

    model = Model([input_joints, input_joints_diff], x)
    
    return model

if __name__ == '__main__':
    
    lb = LabelBinarizer()
    SBU = SBU_dataset(dir="C:\Python专用\临时工作区\SBU\data\SBU")
    datapos,datapath = SBU.get_data()

    pos = {}
    label = []
    for i in range(1, 9):
        pos[i] = []
        for ins in datapath[i]:
            pos[i].append(read_skeleton_file(ins))
            label.append(i)

    for i in range(1, 9):
        for j in range(len(pos[i])):
            ind = np.sort(np.random.randint(0, len(pos[i][j]) - 1, 28))
            pos[i][j] = pos[i][j][ind]
    
    b=list()
    for i in range(1, 9):
        for j in range(len(pos[i])):
            t=pos[i][j]
            b.append(t[np.newaxis,:])
    
    X = np.concatenate(b,axis=0)
    Y = lb.fit_transform(label)
    
    
    
            
    
    model = one_obj()
    model.summary()
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
'''
rs=ShuffleSplit(n_splits=3,test_size=.3,random_state=0)
rs.get_n_splits(X)
for train_index,test_index in rs.split(X,Y):
    xtrain,xtest=X[train_index],X[test_index]
    ytrain,ytest=Y[train_index],Y[test_index]
'''

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2,random_state=0)

x_d=xtrain.copy()
for i in range(225):
    for j in range(28):
        if j!=27:
            x_d[i][j]=xtrain[i][j+1]-xtrain[i][j]
x_t=xtest.copy()
for i in range(57):
    for j in range(28):
        if j!=27:
            x_t[i][j]=xtest[i][j+1]-xtest[i][j]

'''
def norm(d):
    m=np.mean(d)
    mx=np.max(d)
    mi=np.min(d)
    print(m)
    d=d/m
    return d

for i in range(225):
    for j in range(28):
            x_d[i][j]=norm(x_d[i][j])

for i in range(57):
    for j in range(28):
            x_t[i][j]=norm(x_t[i][j])
'''

history=model.fit([xtrain,x_d],ytrain,epochs=150,validation_data=([xtest,x_t],ytest),batch_size=32)

plt.plot(history.epoch,history.history.get('loss'),label='train loss')
plt.plot(history.epoch,history.history.get('val_loss'),label='test loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()



'''
print("------测试网络------")
predictions = model.predict(xtest, batch_size=32)
print(classification_report(ytest.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))
'''
