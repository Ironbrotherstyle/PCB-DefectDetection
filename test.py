#!/usr/bin/env python
# _*_ coding: utf-8 _*_

from keras.models import Model,Sequential
from keras.layers import Dense,Convolution2D,Conv2D,Flatten,MaxPooling2D,Activation,Dropout,BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import  plot_model, model_to_dot
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import time
import os

# 训练的batch_size
batch_size = 16
# 训练的epoch
epochs = 1

s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

# 图像Generator，用来构建输入数据
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # vertical_flip=True,
        # rotation_range=True,
        # featurewise_center=True,
        # featurewise_std_normalization=True
        )

# 从文件中读取数据，目录结构应为train下面是各个类别的子目录，每个子目录中为对应类别的图像
train_generator = train_datagen.flow_from_directory('/Users/apple/Desktop/classificiation/test/28_new_data_test/train',
                                                    target_size = (28, 28),
                                                    batch_size = batch_size,
                                                    )

# 训练图像的数量
train_image_numbers = train_generator.samples

# 输出类别信息
print (train_generator.class_indices)

# 生成测试数据
val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        #featurewise_center=True,
        #featurewise_std_normalization=True
         )
validation_generator = val_datagen.flow_from_directory('/Users/apple/Desktop/classificiation/test/28_new_data_test/val',
                                        target_size = (28, 28),
                                        batch_size = batch_size,
                                        )

val_image_numbers = validation_generator.samples

test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=True,
        featurewise_std_normalization=True)
test_generator = test_datagen.flow_from_directory('/Users/apple/Desktop/classificiation/test/28_new_data_test/test',
                                        target_size = (28, 28),
                                        batch_size = batch_size)

test_image_numbers = test_generator.samples

# #使用ResNet的结构，不包括最后一层，且加载ImageNet的预训练参数
# base_model = ResNet50(weights = 'imagenet', include_top = False, pooling = 'avg')
# #构建网络的最后一层，3是自己的数据的类别
# predictions = Dense(6, activation='softmax')(base_model.output)
#
# #定义整个模型
# base_model = Model(inputs=base_model.input, outputs=predictions)


base_model = Sequential()
#卷积层 12 × 120 × 120 大小
base_model.add(Conv2D(
        nb_filter = 16,
        nb_row = 5,
        nb_col = 5,
        border_mode = 'valid',
        dim_ordering = 'tf',
        input_shape = (28,28,3)))
# base_model.add(BatchNormalization())
base_model.add(Activation('relu'))#激活函数使用修正线性单元
base_model.add(Dropout(0.25))
#池化层12 × 60 × 60

base_model.add(Conv2D(
        nb_filter = 32,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'valid',
        dim_ordering = 'tf'))
# base_model.add(BatchNormalization())
base_model.add(Activation('relu')) #激活函数使用修正线性单元
#池化层12 × 60 × 60
base_model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
base_model.add(Dropout(0.25))

base_model.add(Conv2D(
        nb_filter = 64,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'valid',
        dim_ordering = 'tf'))
# base_model.add(BatchNormalization())
base_model.add(Activation('relu'))#激活函数使用修正线性单元
#池化层12 × 60 × 60
base_model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
base_model.add(Dropout(0.25))
base_model.add(Flatten())
base_model.add(Dense(128))
base_model.add(Activation('relu'))
base_model.add(Dropout(0.5))
base_model.add(Dense(6))
base_model.add(Activation('softmax'))


'''
base_model = Sequential()
#卷积层 12 × 120 × 120 大小
base_model.add(Conv2D(
        nb_filter = 32,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'valid',
        dim_ordering = 'tf',
        input_shape = (28,28,3)))
# base_model.add(BatchNormalization())
base_model.add(Activation('relu'))#激活函数使用修正线性单元
base_model.add(Dropout(0.25))
#池化层12 × 60 × 60
base_model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
base_model.add(Conv2D(
        nb_filter = 64,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'valid',
        dim_ordering = 'tf',
        input_shape = (28,28,3)))
# base_model.add(BatchNormalization())
base_model.add(Activation('relu'))#激活函数使用修正线性单元
base_model.add(Dropout(0.25))
#池化层12 × 60 × 60
base_model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
base_model.add(Flatten())
base_model.add(Dense(64))
base_model.add(Activation('relu'))
base_model.add(Dropout(0.5))
base_model.add(Dense(6))
base_model.add(Activation('softmax'))
'''
'''
base_model = Sequential()
#卷积层 12 × 120 × 120 大小
base_model.add(Convolution2D(
        nb_filter = 64,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'valid',
        dim_ordering = 'tf',
        input_shape = (28,28,3)))
# base_model.add(BatchNormalization())
base_model.add(Activation('relu'))#激活函数使用修正线性单元
#池化层12 × 60 × 60
base_model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
base_model.add(Flatten())
base_model.add(Dense(64))
base_model.add(Activation('relu'))
base_model.add(Dense(6))
base_model.add(Activation('softmax'))
'''
'''
base_model=Sequential()
base_model.add(Convolution2D(32,(3,3),input_shape=(28,28,3)))
base_model.add(Activation('relu'))
base_model.add(MaxPooling2D(pool_size=(2,2)))

base_model.add(Convolution2D(32,(3,3)))
base_model.add(Activation('relu'))
base_model.add(MaxPooling2D(pool_size=(2,2)))

base_model.add(Flatten())
base_model.add(Dense(64))
base_model.add(Activation('relu'))
base_model.add(Dropout(0.5))
base_model.add(Dense(6)) #six-class
base_model.add(Activation('softmax')) #Softmax
'''
print (base_model.summary())

# 编译模型，loss为交叉熵损失
base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#logs file path
logs_path = './log_pcb%s'%(s_time)
try:
   os.makedirs(logs_path)
except:
   pass

#record loos,acc,val_loss,val_acc to tensorboard
tensorboard=TensorBoard(log_dir=logs_path, histogram_freq=0,write_graph=True)

# 训练模型
history=base_model.fit_generator(train_generator,
                         steps_per_epoch = train_image_numbers // batch_size,
                         epochs = epochs,
                         verbose=1,
                         validation_data = validation_generator,
                         #validation_steps = batch_size
                         validation_steps = val_image_numbers // batch_size,
                         callbacks=[tensorboard]
                         )
with open('log_keras_test_test.txt','w') as f:
    f.write(str(history.history))

#top_layer = base_model.layers[0]

# the weight[0] has the data of all kernels in this layer, weight[1] is  a nb_filter*1 array
#print(len(top_layer.get_weights()))
# print(type(top_layer.get_weights()[0]))
#print(top_layer.get_weights()[0].shape)
#print(top_layer.get_weights()[1].shape)
# print(top_layer.get_weights()[1])
#print(top_layer.get_weights()[0][:,:,:,0].squeeze())

'''
plt.subplot(2,2,1)
plt.axis('off')
plt.imshow (top_layer.get_weights()[0][:,:,:,0].squeeze()[0], cmap='gray')
plt.subplot(2,2,2)
plt.axis('off')
plt.imshow (top_layer.get_weights()[0][:,:,:,5].squeeze()[0], cmap='gray')
plt.subplot(2,2,3)
plt.axis('off')
plt.imshow (top_layer.get_weights()[0][:,:,:,10].squeeze()[0], cmap='gray')
plt.subplot(2,2,4)
plt.axis('off')
plt.imshow (top_layer.get_weights()[0][:,:,:,15].squeeze()[0], cmap='gray')
'''
#plt.imshow (top_layer.get_weights()[0][:,0].squeeze()[0], cmap='gray')
# plt.savefig("1.png")
#plt.show()

scores = base_model.evaluate_generator(test_generator)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

plot_model(base_model, to_file='base_model_test_test.png', show_shapes=True)

#保存训练得到的模型
base_model.save_weights('weights.h5')
