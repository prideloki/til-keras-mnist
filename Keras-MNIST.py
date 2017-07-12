
# coding: utf-8

# # MNIST using Keras

# In[1]:

from keras.datasets import mnist
import matplotlib.pyplot as plt


# ## Download the MNIST dataset

# In[2]:

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[3]:

plt.subplot(221)
plt.imshow(X_train[0], cmap = plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap = plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap = plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap = plt.get_cmap('gray'))


# In[4]:

plt.show()


# ## Explore the dataset

# one digit is represent in 28x28 pixels

# In[5]:

X_train[0].shape


# In[6]:

X_train.shape


# 60,000 data, each of this is a 28x28 pixels

# ## Baseline model with multi-later perceptrons
# 
# simple neural network model with a single hidden layer

# In[7]:

import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


# In[8]:

seed = 7
np.random.seed(seed)


# In[9]:

(X_train, y_train) , (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
num_pixels


# flatten 28*28 images to a long vector (784 slots)

# In[10]:

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')


# reshape function:
# - array
# - newshape (int)
# 
# 

# normalize the data. the values are scale between 0 and 255

# In[11]:

X_train = X_train / 255
X_test = X_test / 255


# hot encoding the categorical values

# In[12]:

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[13]:

num_classes = y_test.shape[1]


# In[14]:

num_classes


# In[15]:

def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim = num_pixels, kernel_initializer='normal', activation='relu'))
#     output layer
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


# `softmax` activation
# 

# In[16]:

model = baseline_model()
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size=200,verbose=2)
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


# ## Simple convolutional neural network for MNIST

# In[17]:

from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# In[18]:

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_test = X_test.reshape(X_test.shape[0],1,28,28).astype('float32')


# In[19]:

X_train[0].shape


# normalize the data, gray color has value range between 0 and 255

# In[20]:

X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# In[21]:

y_test.shape


# In[22]:

def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5,5), input_shape = (1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[23]:

model = baseline_model()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%"%(100-scores[1]*100))


# ## Larger convolutional neural network for MNIST

# In[24]:

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# In[25]:

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[26]:

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')


# In[27]:

X_train = X_train / 255
X_test = X_test / 255


# In[28]:

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# In[29]:

def larger_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape = (1, 28 ,28), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


# In[30]:

model = larger_model()
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, batch_size = 200)

scores = model.evaluate(X_test, y_test, verbose = 0)
print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))


# In[ ]:




# In[ ]:



