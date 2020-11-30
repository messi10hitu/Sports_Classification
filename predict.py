#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class sports:
    def __init__(self, filename):
        self.filename = filename

    def predictionsports(self):
        # load model
        model = keras.models.load_model('TF_Sports_resnet50.h5')

        # summarize model
        # model.summary()
        imagename = self.filename

        # load an image from file
        image = load_img(imagename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model.predict(image)
        # print(yhat)
        # print(np.argmax(yhat[0]))

        classification = [np.argmax(yhat[0])]
        # print(classification)
        if classification == [0]:
            prediction = 'badminton'
            return [{"image": prediction}]
        elif classification == [1]:
            prediction = 'baseball'
            return [{"image": prediction}]
        elif classification == [2]:
            prediction = 'basketball'
            return [{"image": prediction}]
        elif classification == [3]:
            prediction = 'boxing'
            return [{"image": prediction}]
        elif classification == [4]:
            prediction = 'chess'
            return [{"image": prediction}]
        elif classification == [5]:
            prediction = 'cricket'
            return [{"image": prediction}]
        elif classification == [6]:
            prediction = 'fencing'
            return [{"image": prediction}]
        elif classification == [7]:
            prediction = 'football'
            return [{"image": prediction}]
        elif classification == [8]:
            prediction = 'formula1'
            return [{"image": prediction}]
        elif classification == [9]:
            prediction = 'gymnastics'
            return [{"image": prediction}]
        elif classification == [10]:
            prediction = 'hockey'
            return [{"image": prediction}]
        elif classification == [11]:
            prediction = 'ice_hockey'
            return [{"image": prediction}]
        elif classification == [12]:
            prediction = 'kabaddi'
            return [{"image": prediction}]
        elif classification == [13]:
            prediction = 'motogp'
            return [{"image": prediction}]
        elif classification == [14]:
            prediction = 'shooting'
            return [{"image": prediction}]
        elif classification == [15]:
            prediction = 'swimming'
            return [{"image": prediction}]
        elif classification == [16]:
            prediction = 'table_tennis'
            return [{"image": prediction}]
        elif classification == [17]:
            prediction = 'tennis'
            return [{"image": prediction}]
        elif classification == [18]:
            prediction = 'volleyball'
            return [{"image": prediction}]
        elif classification == [19]:
            prediction = 'weight_lifting'
            return [{"image": prediction}]
        elif classification == [20]:
            prediction = 'wrestling'
            return [{"image": prediction}]
        elif classification == [21]:
            prediction = 'wwe'
            return [{"image": prediction}]
