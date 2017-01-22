import os
from pathlib import Path
import pandas as pd
import numpy as np
import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle
from keras.models import model_from_json
import json
from sklearn.model_selection import train_test_split
import random


df = pd.read_csv("driving_log.csv", header=None, skipinitialspace=True, names=["center", "left", "right",  "steering", "throttle", "brake", "speed"])
df.shape

# Center Data
center = df[['center', 'steering']]
center.columns = ['image', 'steering']

# Left Data
left = df[['left', 'steering']]
left.columns = ['image', 'steering'] 
left.loc[:, "steering"] = left.steering.apply(lambda x: x+0.35)

# Right Data
right = df[['right', 'steering']]
right.columns = ['image', 'steering']
right.loc[:, "steering"] = right.steering.apply(lambda x: x-0.35)


frames = [left, center, right]
# Combine data
data = pd.concat(frames, axis = 0, ignore_index = True)
        
        
X = data.image
Y = data.steering
Y_train = Y.astype(np.float32)

# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20, random_state=1)
X_train = X.reset_index(drop=True)
Y_train = Y.reset_index(drop=True)
# X_val = X_val.reset_index(drop=True)
# Y_val = Y_val.reset_index(drop=True)

print("Training shape is {}".format(X_train.shape))
# print("Validation shape is {}".format(X_val.shape))





img_cols, img_rows = 200, 66

def read_image(image_path, steering):
    image = mpimg.imread(image_path)
    shape = image.shape
    image = image[int(shape[0]/3):int(shape[0]-25), 0:shape[1]]
    image = cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
    return image, steering

def flip(image, steering):
    if np.random.randint(2)==0:
        image_flip = cv2.flip(image, 1)
        steering *= (-1.)
    else:
        image_flip = image
    return image_flip, steering

def change_brightness(image, steering):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,2] = image[:,:,2] * np.random.uniform(0.4, 1.2)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image, steering
    
def shift(image, steering):
    trans_range = 150
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steering + tr_x/trans_range*2*.4
    tr_y = 10*np.random.uniform()-10/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(200,66))

    return image_tr, steer_ang











img_rows, img_cols = 66, 200
epoch_size = 3000
batch_size_train = 100
batch_size_val = 20

def image_preprocess(img_path, steering):
    image, steering = read_image(img_path, steering)
    image, steering = flip(image, steering)
    image, steering = change_brightness(image, steering)
    image, steering = shift(image,steering)
   
    image = np.array(image).astype("float32")
    image = image.reshape((1, img_rows, img_cols, 3))
    return image, steering


def batchgen_train(X, Y):
    
    start = 0
    features = np.ndarray(shape = (batch_size_train, img_rows, img_cols, 3))
    labels = np.ndarray(shape = (batch_size_train,))
    
    while 1:
        ind = np.random.choice(range(X.shape[0]), batch_size_train)
        X_batch = X[ind]
        Y_batch = Y[ind]
        for i in range(start, start+batch_size_train):
            image, label = image_preprocess(X_batch.iloc[i%batch_size_train], Y_batch.iloc[i%batch_size_train])
            features[i%batch_size_train] = image
            label = np.array([[label]])
            labels[i%batch_size_train] = label
        start = start + batch_size_train
        yield (features, labels)

def batchgen_val(X, Y):
    
    start = 0
    features = np.ndarray(shape = (batch_size_val, img_rows, img_cols, 3))
    labels = np.ndarray(shape = (batch_size_val,))
    
    while 1:
        ind = np.random.choice(range(X.shape[0]), batch_size_val)
        X_batch = X[ind]
        Y_batch = Y[ind]
        for i in range(start, start+batch_size_val):
            image, label = read_image(X.iloc[i%batch_size_val], Y.iloc[i%batch_size_val])
            features[i%batch_size_val] = image
            label = np.array([[label]])
            labels[i%batch_size_val] = label
        start = start + batch_size_val
        yield (features, labels)


        
def save_model(fileModelJSON,fileWeights):
    #print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON,'w' ) as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)
    




# Build the network

from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Dropout, MaxPooling2D, Convolution2D, Lambda, ELU, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam


input_shape = (66, 200, 3)
pool_size = (2, 3)
dropout = 0.4
samples_per_epoch = epoch_size - epoch_size%batch_size_train
nb_epoch = 1
nb_val_samples = samples_per_epoch/20


if Path("model_7.json").is_file():
    with open("model_7.json", 'r') as jfile:
       model = model_from_json(json.load(jfile))
    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam, loss="mse")
    model.load_weights("model_7.h5")
    print("Loaded model from disk:")
    model.summary()            
 
else:
    model = Sequential()
    
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(dropout))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(dropout))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(dropout))
    
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(dropout))
    
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(dropout))
    
    model.add(Flatten())
    model.add(ELU())
    
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(dropout))
    
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(dropout))
    
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dropout(dropout))
    
    model.add(Dense(1))
    
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss="mse")
    model.summary()


history = model.fit_generator(batchgen_train(X_train, Y_train),
                    samples_per_epoch=samples_per_epoch, 
                    nb_epoch=nb_epoch,
#                     validation_data=batchgen_val(X_val, Y_val),
#                     nb_val_samples=nb_val_samples,          
                    verbose=1)

fileModelJSON = 'model_7' + '.json'
fileWeights = 'model_7' + '.h5'

save_model(fileModelJSON,fileWeights)

