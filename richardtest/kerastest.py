# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
#from keras.datasets import mnist
 
# 4. Load pre-shuffled MNIST data into train and test sets
print('loading data')
from os import listdir
from os.path import isfile, join
import os
onlyfiles = [f for f in listdir('dataset')]
onlyfiles = onlyfiles[1:]
data = []
label = []
for f in onlyfiles:
    root = 'dataset/' + f
    endInt = int(f[1:], 2)
    for item in listdir(root):
        path = os.path.join(root, item)
        if not item.startswith('.') and os.path.isfile(path):
            data.append(np.load(path))
            label.append(endInt)
#print(data)
print(label)

X_train = np.array(data)
y_train = np.array(label)
X_test = X_train
y_test = y_train


#(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
# 5. Preprocess input data
print('preprocess input')
X_train = X_train.reshape(X_train.shape[0], 1, 46, 72)
X_test = X_test.reshape(X_test.shape[0], 1, 46, 72)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
# 6. Preprocess class labels
print('preprocess class')
Y_train = np_utils.to_categorical(y_train, 32)
Y_test = np_utils.to_categorical(y_test, 32)
 
# 7. Define model architecture
print('model architecture')
model = Sequential()
 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,46,72)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='softmax'))


print(model.summary())


# 8. Compile model
print('compiling')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
print('fitting model')
print(X_train.shape)
print(Y_train.shape)
model.fit(X_train, Y_train, 
          batch_size=16, nb_epoch=10, verbose=1)
 
# 10. Evaluate model on test data
print('evaluating')
score = model.evaluate(X_test, Y_test, verbose=1)