##Import all libraries used in the model 
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Flatten,Dense,Lambda
from keras.layers import Convolution2D
from keras import backend as K
import csv


#Read given training data sets
K.set_image_dim_ordering('tf')
train_data=[]

with open('driving_log.csv') as train_file:
    read = csv.reader(train_file)
    for line in read:
        train_data.append(line)

#Combine the input training sets. 
#From training sets, use all 3 camera angle views, 
#1. center_image as is with given steering angle
#2. left_image with extra 0.3 steered right
#3. right_image with extra 0.3 steered left
#The optimal steering delta for left/right images was evaluated after trying couple values.

image_set = []
train_data = train_data[1:]
print(len(train_data))
print(train_data[1])
output = []
i = 1
for data in train_data:
    i = i + 1
    
     # create adjusted steering measurements for the side camera images
    correction = 0.3
    steering_center = data[3]
    steering_left = float(steering_center) + correction
    steering_right = float(steering_center) - correction
    
    # read in images from center, left and right cameras
    center_image = data[0]
    left_image = data[1].strip()
    right_image=data[2].strip()
   
    # add images and angles to data set
    image_set.extend([cv2.imread(center_image),cv2.imread(left_image),cv2.imread(right_image)])
    output.extend([steering_center,steering_left,steering_right])   

X_train = np.array(image_set)
y_train = np.array(output)

print(len(y_train))

#NVidea's Convolutional Neural Network Model was chosen, with 5 Conv2D + 3 FullyConnected layers. 
# set up cropping2D layer
model=Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=4)
model.save('model.h5')

