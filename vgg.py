#!/usr/bin/env python
# _*_ coding: utf-8 _*_

from keras.models import Model,Sequential
from keras.layers import Dense,Convolution2D,Flatten,MaxPooling2D,Activation,Dropout,BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import  VGG16
from keras.utils.vis_utils import  plot_model, model_to_dot
import matplotlib.pyplot as plt

# 训练的batch_size
batch_size = 16
# 训练的epoch
epochs = 1

# 图像Generator，用来构建输入数据
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=True,
        featurewise_center=True,
        featurewise_std_normalization=True
        )

# 从文件中读取数据，目录结构应为train下面是各个类别的子目录，每个子目录中为对应类别的图像
train_generator = train_datagen.flow_from_directory('/Users/apple/Desktop/classificiation/test/28_new_data_test/train', target_size = (224, 224), batch_size = batch_size)

# 训练图像的数量
image_numbers = train_generator.samples

# 输出类别信息
print (train_generator.class_indices)

# 生成测试数据
val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=True,
        featurewise_std_normalization=True)
validation_generator = val_datagen.flow_from_directory('/Users/apple/Desktop/classificiation/test/28_new_data_test/val',
                                        target_size = (224, 224),
                                        batch_size = batch_size)

val_image_numbers = validation_generator.samples

test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=True,
        featurewise_std_normalization=True)
test_generator = test_datagen.flow_from_directory('/Users/apple/Desktop/classificiation/test/28_new_data_test/test',
                                        target_size = (224, 224),
                                        batch_size = batch_size)

test_image_numbers = test_generator.samples


base_model = VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3))
for layer in base_model.layers[:-1]:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
fc = Dense(64)(x)
output = Dense(6,activation="softmax")(fc)

model = Model(inputs=base_model.inputs,outputs=output)


print (model.summary())

# 编译模型，loss为交叉熵损失
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 训练模型
history=model.fit_generator(train_generator,
                         steps_per_epoch = image_numbers // batch_size,
                         epochs = epochs,
                         validation_data = validation_generator,
                         validation_steps = val_image_numbers // batch_size
                         )
with open('log_keras_vgg16.txt','w') as f:
    f.write(str(history.history))

print(validation_generator)

# top_layer = base_model.layers[1]
# plt.imshow (top_layer.get_weights()[0][:,:,:,10].squeeze()[0], cmap='gray')
# plt.show()

plot_model(base_model, to_file='base_model_vgg16.png', show_shapes=False)


scores = base_model.evaluate(test_generator)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
#保存训练得到的模型
model.save_weights('weights.h5')
