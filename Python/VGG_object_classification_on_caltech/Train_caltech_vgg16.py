import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import initializers
from keras import backend as K

K.image_data_format() == 'channels_first'

import os
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.transpose(np.array([img, img, img]), (2, 0, 1))
    return img

path = "/media/suzhuo/Elements/mediaCal/data/caltech-101"
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)
imgs = []
labels = []
# LOAD ALL IMAGES 
for i, category in enumerate(categories):
    iter = 0
    for f in os.listdir(path + "/" + category):
        if iter == 0:
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_exts:
                continue
            fullpath = os.path.join(path + "/" + category, f)
            img = scipy.misc.imresize(imread(fullpath), [128,128, 3])
            img = img.astype('float32')
            img[:,:,0] -= 123.68
            img[:,:,1] -= 116.78
            img[:,:,2] -= 103.94
            imgs.append(img) # NORMALIZE IMAGE 
            label_curr = i
            labels.append(label_curr)
        #iter = (iter+1)%10;
# print ("Num imgs: %d" % (len(imgs))) # 9144
# print ("Num labels: %d" % (len(labels)) ) #9144
# print ("Labels: ",labels)
# print ('Number of categories: ',ncategories) # 102
# print ('Type of imgs: ',type(imgs)) # type is list
# print ('Len of imgs: ',len(imgs)) # 9144
# print ('Sample img', imgs[1])
# print ('Type of sample img: ',type(imgs[1])) # numpy.ndarray
# print ('Size of sample img: ',np.size(imgs[1])) # 49152 is this the orginal ? 3*2^14
seed = 7
np.random.seed(seed)
import pandas as pd
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size = 0.1) # Change the proportion of training and test
X_train = np.stack(X_train, axis=0)
y_train = np.stack(y_train, axis=0)
X_test = np.stack(X_test, axis=0)
y_test = np.stack(y_test, axis=0)
# print ("Num train_imgs: %d" % (len(X_train))) # 8229 when test_size = 0.1
# print ("Num test_imgs: %d" % (len(X_test))) # 915 when test_size = 0.1
# # one hot encode outputs - change the single value into a vector with (number of categories) dimensions
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes= y_test.shape[1]

# print('Shape of y_test: ',y_test.shape) # (915,102)
# print('Shape of y_train: ',y_train.shape) # (8229,102)
# print('Sample of X_train: ',X_train[1,1,1,:]) # [25.32 -19.779999 -20.940002]? value of three channels? why it be negative?
# # print('Sample of y_train: ', y_train[1])
# # normalize inputs from 0-255 to 0.0-1.0
# print('Former Shape of X_train: ',X_train.shape) # (8229,128,128,3) - (batch_size,w,h,channel)
# print('Former Shape of X_test: ',X_test.shape) # (915,128,128,3) -
# # X_train = X_train.transpose(0, 3, 1, 2) #
# # X_test = X_test.transpose(0, 3, 1, 2) #
# print('After Shape of X_train: ',X_train.shape) # (8229,3,128,128)
# print('After Shape of X_test: ',X_test.shape) # (915,3,128,128)

import scipy.io as sio
data = {}
data['categories'] = categories
data['X_train'] = X_train
data['y_train'] = y_train
data['X_test'] = X_test
data['y_test'] = y_test
sio.savemat('caltech_del.mat', data)

from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
# Create the model
model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), input_shape=(3, 128, 128), padding='same', activation='relu')) # original
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) #error here

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile mode
epochs = 300 #
lrate = 0.0001
decay = lrate/epochs
#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = SGD(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

np.random.seed(seed)
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
          epochs=epochs, batch_size=56, shuffle=True, callbacks=[earlyStopping])
#hist = model.load_weights('./64.15/model.h5');
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['train','test'])
plt.title('loss')
plt.savefig("loss7(vgg16).png",dpi=300,format="png")
plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['train','test'])
plt.title('accuracy')
plt.savefig("accuracy7(vgg16).png",dpi=300,format="png")
model_json = model.to_json()
with open("model7(vgg16).json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5r
model.save_weights("model7(vgg16).h5")
print("Saved model to disk")
