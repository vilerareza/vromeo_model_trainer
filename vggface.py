import os
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras import layers, optimizers
import tensorflow as tf


class VGGFace():

    model = None

    def __init__(self, nClasses, modelWeightsFile = "files/vgg_face_weights.h5"):
        self.modelWeightsFile = modelWeightsFile
        self.model = self.create_model(nClasses)

    def create_model(self, nClasses):
        baseModel = self.baseModel()
        self.load_weights (baseModel, self.modelWeightsFile)
        # for layer in baseModel.layers:
        #     layer.trainable = False
        model = self.add_layers(baseModel, nClasses)
        return model
        
    def baseModel(self):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        return model

    def load_weights(self, model, weightFile):
        if os.path.isfile(weightFile):
            model.load_weights(weightFile)
            print ("Weights loaded to model")
        else:
            print ("File not found. Nothing to load")

    def add_layers(self, baseModel, nClasses):
        conv1 = layers.Conv2D(2048, 1) (baseModel.layers[-4].output)
        conv2 = layers.Conv2D(1024, 1) (conv1)
        conv3 = layers.Conv2D(512, 1) (conv2)
        conv4 = layers.Conv2D(nClasses, 1) (conv3)
        flat1 = layers.Flatten()(conv4)
        #fc = Dense(1024, activation='relu')(flat)
        out = layers.Activation('softmax')(flat1)
        model = Model(inputs=baseModel.layers[0].input, outputs=out)
        return model


