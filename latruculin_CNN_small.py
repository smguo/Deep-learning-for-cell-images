'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import os
from six.moves import cPickle as pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils, plot_model
from keras import backend as K
#import pandas as pd
import numpy as np
import os
import glob

#import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
mpl.rcParams['pdf.fonttype'] = 42 # change the default settings of matplotlib
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams.update({'figure.autolayout': True})
#%%
def load_img(dir_path, subdir_name, imgsub_w, imgsub_h, chan_idx):        
#    subdir_path = glob.glob(os.path.join(dir_path, '*/'))
#    #subdir_name = ['p1','p2','p3', 'p4', 'p6', 'p7', 'p8' , 'p9', 'p10', 'p12']
#    subdir_name = [os.path.split(subdir[:-1])[1] for subdir in subdir_path]   
    nSubdir = len(subdir_name)
    file_path = os.path.join(dir_path, subdir_name[1],"MultiplexImageDataAligned.mat")
    mat = scipy.io.loadmat(file_path)
    imgsProjAligned = mat['imgsProjAligned']
    
    imgSingField = imgsProjAligned [1,0]
    imgSingFieldRound = imgSingField [0,0]
    img_h, img_w = imgSingFieldRound.shape[0], imgSingFieldRound.shape[1]
#    n_channels = imgSingField.shape[0]
    n_channels = len(chan_idx)
    dataset = np.array([]).reshape(0, imgsub_h, imgsub_w,n_channels)
    labels = np.array([],dtype=np.int16)
            
    for j in range(0, nSubdir):
        file_path = os.path.join(dir_path, subdir_name[j],"MultiplexImageDataAligned.mat")        
        mat = scipy.io.loadmat(file_path)
        imgsProjAligned = mat['imgsProjAligned']
        nField = imgsProjAligned.shape[0]               
        n_sample = int((img_h/imgsub_h)*(img_w/imgsub_w)*nField)
        dataset_temp = np.ndarray(shape=(n_sample, imgsub_h, imgsub_w, n_channels),
                             dtype=np.float32)
        labels_temp = np.ndarray(n_sample, dtype=np.int16)
        sample_ind = 0 #index for each sample                     
        for f in range(0, nField):
            imgSingField = np.stack(imgsProjAligned[f,0][:,0],axis=-1).astype(np.float32) # stack images from different channels                                                         
            imgSingField = normalize_img(imgSingField)
            imgSingField =  imgSingField[:,:,chan_idx]
#            if j==0:
#                imgSingField = imgSingField[:,:,1].reshape(img_h, img_w,1)
#            else:
#                imgSingField = imgSingField[:,:,2].reshape(img_h, img_w,1)
                
            for h in range(0,img_h, imgsub_h):
                for w in range(0,img_w, imgsub_w):
                    dataset_temp[sample_ind,:,:,:] = imgSingField[h:h+imgsub_h,w:w+imgsub_w,:]
                    sample_ind += 1
        labels_temp[:] = j
        dataset = np.concatenate((dataset, dataset_temp), axis=0)
        labels = np.concatenate((labels, labels_temp), axis=0)
#        show_images(dataset[-54:,:,:,:])
    return dataset, labels

def normalize_img(img):
    n_channels = img.shape[2]
    for chan in range(0,n_channels):
        img_s = img[:,:,chan]
        img_s= img_s-np.mean(img_s)
        img_s = img_s/np.std(img_s)
        img[:,:,chan]=img_s
    return img

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def show_images(dataset):    
    n_img, img_h, img_w = dataset.shape[0], dataset.shape[1], dataset.shape[2]
    subplot_num = 1
    fig = plt.figure()                    
    for j in range(0,n_img):            
        img = dataset[j,:,:,:].reshape(img_h, img_w)                    
        a=fig.add_subplot(np.ceil(n_img/6),6,subplot_num)            
        imgplot = plt.imshow(img, cmap='gray')                                 
        subplot_num = subplot_num + 1
#%%
dir_path = 'E:\\data\\Neuron\\cortical\\Broad_HCS\\14days\\MF20170215-latruculin-time\\post_processing_multicolor'
data_root = 'E:\Google Drive\Python\deep learning' # Change me to store data elsewhere
os.chdir(data_root)
path = 'E:\\Program Files (x86)\\Graphviz2.38\\bin'
os.environ["PATH"] += os.pathsep + path        
subdir_name = ['DMSO 24hr-3', 'latrunculin 24hr-3']
#subdir_name = ['DMSO 24hr-3', 'DMSO 24hr-2']
imgsub_w, imgsub_h = 72,72
chan_idx = [0]
n_channels = len(chan_idx)
dataset, labels=load_img(dir_path, subdir_name, imgsub_w, imgsub_h, chan_idx)
n_sample = labels.shape[0]
train_size = round(0.8*n_sample)
test_size = n_sample-train_size
shuffled_dataset, shuffled_labels = randomize(dataset, labels)
train = shuffled_dataset[:train_size,:,:,:]
train_labels = shuffled_labels[:train_size]
test = shuffled_dataset[train_size:,:,:,:]
test_labels = shuffled_labels[train_size:] 

print('Training set', train.shape, train_labels.shape)
#print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test.shape, test_labels.shape)
#%%
#train = train_dataset[0:1000,:,:]
#valid = valid_dataset[0:1000,:,:]
#train_sublabl = train_labels[0:1000]
#valid_sublabl = valid_labels[0:1000]
#np.unique(train_sublabl) # check the number of classes

#%%

# dimensions of our images.
img_width, img_height = train.shape[1], train.shape[2]
epochs = 50
batch_size = 16

#train_labels = np_utils.to_categorical(train_labels , 2)
#test_labels = np_utils.to_categorical(test_labels , 2)

if K.image_data_format() == 'channels_first':
    input_shape = (n_channels, img_width, img_height)
else:
    input_shape = (img_width, img_height, n_channels)
#%%
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%
plot_model(model, to_file='model.jpg', rankdir='TB')
#%% this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,    
    horizontal_flip=True,
    vertical_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    train,
    train_labels,
    batch_size=batch_size)

test_generator = test_datagen.flow(
    test,
    test_labels,    
    batch_size=batch_size)


model.fit_generator(
    train_generator,
    steps_per_epoch=train_size // batch_size,
    epochs=epochs,
    workers = 4,
    use_multiprocessing = False,
    validation_data=test_generator,
    validation_steps=test_size // batch_size)

model.save_weights('2nd_try.h5')
#%%
#test_dataset = (test,test_labels)
#model.fit(
#    x=train,
#    y=train_labels,     
#    epochs=epochs,
#    validation_data=test_dataset)    
#
#model.save_weights('first_try.h5')