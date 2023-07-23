#importing libraries
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from plotly.offline import init_notebook_mode
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import loadmat
from tkinter import filedialog
import os
import spectral
import tensorflow as tf

## GLOBAL VARIABLES
windowSize = 25

#loading image datafile and its ground truth file
def loadData():
    global data_name
    data = loadmat(filedialog.askopenfilename(initialdir=os.getcwd(), title='Select Image Data File'))
    data_name=next(reversed(data))
    data=data[data_name]
    labels = loadmat(filedialog.askopenfilename(initialdir=os.getcwd(), title='Select Ground Truth File'))
    label_name=next(reversed(labels))
    labels=labels[label_name]
    #converting into sequential labels for salinas A imagery data
    if label_name == 'salinasA_gt':
        unique_labels = np.unique(labels)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        labels = np.array([[label_map[label] for label in row] for row in labels])
    return data_name,data,labels

#applying PCA for dimentionality reduction
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

#applying padding to maintain same size
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

#making 3-D cubes 
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    #removing unlabelled class data
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

#calling loadData function  
dataset,X, y = loadData()
X_actual,y_actual= X,y

#saving indices of unlabelled classes
y_ravel=y.ravel()
zero_indices = [i for i in range(len(y_ravel)) if y_ravel[i] == 0]

#setting output channel dimentionality for PCA
if dataset == 'indian_pines_corrected':
    K = 30  
else:
    K = 15

#applying PCA
X,pca = applyPCA(X,numComponents=K)

X_total,yk=createImageCubes(X, y, windowSize=windowSize, removeZeroLabels = False)

#creating image cubes
X, y = createImageCubes(X, y, windowSize=windowSize)

#data splitting
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.7, random_state=345,stratify=y)

X_total=X_total.reshape(-1, windowSize, windowSize, K, 1)

# Model and Training :
# data reshaping 
Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
Xtrain.shape

ytrain = np_utils.to_categorical(ytrain)
ytrain.shape

#specifying model parameters
S = windowSize
L = K
if dataset == 'paviaU' :
    output_units = 9 
elif dataset == 'salinasA_corrected':
    output_units = 6
else:
    output_units = 16

## input layer
input_layer = Input((S, S, L, 1))

## convolutional layers
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
conv3d_shape = conv_layer3.shape
conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)
flatten_layer = Flatten()(conv_layer4)
## fully connected layers
dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

# define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

print(model.summary())

# compiling the model
adam = Adam(learning_rate=0.001, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# set the desired file path
filepath = f"C:/Users/mohit/Desktop/MohitP/CODING/Ml_project/best_weights.hdf5"

# checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

iteration = int(input('enter iterations : '))
history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=iteration, callbacks=callbacks_list)

# save the model
model.save(filepath)

# Validation
model.load_weights(f"C:/Users/mohit/Desktop/MohitP/CODING/Ml_project/best_weights.hdf5")
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
ytest = np_utils.to_categorical(ytest)
Y_pred_test = model.predict(Xtest)
y_pred_test = np.argmax(Y_pred_test, axis=1)

#predicting image
Y_pred_total = model.predict(X_total)
y_pred_total = np.argmax(Y_pred_total, axis=1)
y_pred_total+=1
#setting value to zero for zero indices in predicted image
for i in zero_indices:
    y_pred_total[i]=0
y_pred_total=np.array(y_pred_total).reshape(y_actual.shape[0],y_actual.shape[1])

# Visualization of Result
# Normalize
y_norm = (y_actual - y_actual.min()) / (y_actual.max() - y_actual.min())
outputs_norm = (y_pred_total - y_pred_total.min()) / (y_pred_total.max() - y_pred_total.min())

# Create a colormap
cmap = plt.cm.jet

# Apply the colormap and normalization to create ScalarMappable objects
import matplotlib
y_mappable = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=y_norm.min(), vmax=y_norm.max()), cmap=cmap)
outputs_mappable = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=outputs_norm.min(), vmax=outputs_norm.max()), cmap=cmap)

# Create subplots for displaying both images side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Display actual image
axes[0].imshow(y_mappable.to_rgba(y_norm))
axes[0].set_title(f'{data_name} Ground Truth Image')

# Display predicted image
axes[1].imshow(outputs_mappable.to_rgba(outputs_norm))
axes[1].set_title(f'{data_name} Predicted Image')

# Add colorbars
fig.colorbar(y_mappable, ax=axes[0])
fig.colorbar(outputs_mappable, ax=axes[1])
plt.show()

## Plot Training Accuracy
plt.figure(figsize=(7,7))
plt.ylim(0,1.1)
plt.grid()
plt.plot(history.history['accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Training Accuracy')
plt.show()

# Classification Report :
classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
print(classification)