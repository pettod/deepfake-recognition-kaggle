import math, json, os, sys
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Input, Conv2D, \
    BatchNormalization, Activation, AveragePooling2D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
from keras.regularizers import l2

DATA_DIR = "../cropped_faces/resnet_data"
TRAIN_DIR = DATA_DIR + "/train"
VALID_DIR = DATA_DIR + "/train"
SIZE = (224, 224)
BATCH_SIZE = 4


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def main():
    with tf.device("/device:GPU:0"):
        num_train_samples = sum(
            [len(files) for r, d, files in os.walk(TRAIN_DIR)])
        num_valid_samples = sum(
            [len(files) for r, d, files in os.walk(VALID_DIR)])

        num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
        num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

        gen = keras.preprocessing.image.ImageDataGenerator()
        val_gen = keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True, vertical_flip=True)

        batches = gen.flow_from_directory(
            TRAIN_DIR, target_size=SIZE, class_mode="categorical", shuffle=True,
            batch_size=BATCH_SIZE)
        val_batches = val_gen.flow_from_directory(
            VALID_DIR, target_size=SIZE, class_mode="categorical", shuffle=True,
            batch_size=BATCH_SIZE)
        a = batches.next()
        model = keras.applications.resnet50.ResNet50(weights=None, classes=2)
        input_shape = tuple(list(SIZE) + [3])
        finetuned_model = resnet_v2(input_shape, 56, num_classes=2)
        finetuned_model.compile(
            optimizer=Adam(lr=0.0001), loss="binary_crossentropy",
            metrics=["accuracy"])

        early_stopping = EarlyStopping(patience=10)
        checkpointer = ModelCheckpoint(
            "resnet50_best.h5", verbose=1, save_best_only=True)

        finetuned_model.fit_generator(
            batches, steps_per_epoch=num_train_steps, epochs=1000,
            callbacks=[early_stopping, checkpointer],
            validation_data=val_batches,
            validation_steps=num_valid_steps)
        finetuned_model.save("resnet50_final.h5")


main()
