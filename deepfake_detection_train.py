import os
import cv2
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
import math

input_shape = ( 128, 128, 3)
data_dir = 'dataset'#'../df/input/cropped-faces-10-v2/train'
TRAIN_DATA_SIZE = len(os.listdir(data_dir+'/train'))*2
VALID_DATA_SIZE = len(os.listdir(data_dir+'/valid'))*2
# real_data = [f for f in os.listdir(data_dir+'/real')[0:DATA_SIZE] if f.endswith('.png')]
# fake_data = [f for f in os.listdir(data_dir+'/fake')[0:DATA_SIZE] if f.endswith('.png')]

# import random
# train_dataset = []
# for img in real_data:
#     dataset.append([img, 1])
# for img in fake_data:
#     dataset.append([img, 0])
# random.shuffle(dataset)

# def gen(dir, batch_size):
    
#     while True:
#         for path, target in dataset:
#             if(target == 1):
#                 yield (img_to_array(load_img(data_dir+'/real/'+path)).flatten() / 255.0).reshape(1, 128,128,3), target
#             else:
#                 yield (img_to_array(load_img(data_dir+'/fake/'+path)).flatten() / 255.0).reshape(1, 128, 128,3), target

        # X.append(img_to_array(load_img(data_dir+'/real/'+img)).flatten() / 255.0)
        # Y.append(1)
        # X.append(img_to_array(load_img(data_dir+'/fake/'+img)).flatten() / 255.0)
        # Y.append(0)

#Y_val_org = Y

#Normalization
#X = np.array(X)
#Y = to_categorical(Y, 2)
1
#Reshape
#X = X.reshape(-1, 224, 2240, 3)

#Train-Test split
#X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


model = tf.keras.models.load_model('deepfake-detection-model.h5')
# googleNet_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
# googleNet_model.trainable = True
# model = Sequential()
# model.add(googleNet_model)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(units=2, activation='softmax'))
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
#               metrics=['accuracy'])
# model.summary()

#Currently not used
early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=2,
                               verbose=0, mode='auto')
EPOCHS = 1000
BATCH_SIZE = 90

gen = keras.preprocessing.image.ImageDataGenerator()
train_batches = gen.flow_from_directory(data_dir+'/train', class_mode="categorical",\
        shuffle=True, batch_size=BATCH_SIZE)
# valid_batches = gen.flow_from_directory(data_dir, class_mode="categorical",\
#         shuffle=True, batch_size=BATCH_SIZE)

#early_stopping_callback = EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve)
checkpoint_callback = keras.callbacks.ModelCheckpoint('deepfake-detection-model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

history = model.fit_generator(train_batches, steps_per_epoch = math.floor(TRAIN_DATA_SIZE / BATCH_SIZE), epochs = EPOCHS,\
    #validation_data=valid_batches, validation_steps = math.floor(VALID_DATA_SIZE / BATCH_SIZE),  \
    verbose = 1, 
    callbacks=[checkpoint_callback])
#history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (X_val, Y_val), verbose = 1)
model.save('deepfake-detection-model.h5')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
t = f.suptitle('Pre-trained InceptionResNetV2 Transfer Learn with Fine-Tuning & Image Augmentation Performance ', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,EPOCHS+1))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, EPOCHS+1, 1))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch #')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, EPOCHS+1, 1))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch #')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

#Output confusion matrix
def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print('True positive = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True negative = ', cm[1][1])
    print('\n')
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.ylabel('Actual label', size = 20)
    plt.xlabel('Predicted label', size = 20)
    plt.xticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.yticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.ylim([2, 0])
    plt.show()
    
print_confusion_matrix(Y_val_org, model.predict_classes(X))
