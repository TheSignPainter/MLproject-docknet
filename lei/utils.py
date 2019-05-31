import constant as c
import pandas as pd
import glob
import os
import numpy as np
def load_train_dataset():
    #load trainning and validation
    dataset = pd.DataFrame()
    files = glob.glob(c.npy_train + '*.npy')
    np.random.shuffle(files)
    dataset['train_img'] = files
    dataset['train_label'] = dataset['train_img'].apply(lambda x: os.path.basename(x).split('_')[0])
    print("Load %d training" % (len(dataset['train_label'])))
    return dataset
def load_valid_dataset():
    dataset = pd.DataFrame()
    files = glob.glob(c.npy_valid + '*.npy')
    np.random.shuffle(files)
    dataset['valid_img'] = files 
    dataset['valid_label'] = dataset['valid_img'].apply(lambda x: os.path.basename(x).split('_')[0])
#    print(dataset['train_label'][0:10])
    print("Load %d validation" % (len(dataset['valid_label'])))
    return dataset

def loadFromList(imgs, labels, start, end):
    x = []
    y = []
    for i in range(start, end):
        img = np.load(imgs[i])
        # shape?
        x.append(img)
        label = labels[i]
        if (label =='docks'):
            y.append(1)
        else:
            y.append(0)
    x = np.asarray(x)
    y = np.asarray(y)
    return x,y

def train_loader(dataset, batch_size):
    imgs = list(dataset['train_img'])
    labels = list(dataset['train_label'])

    L = len(imgs)

    while(True):
        start = 0
        end = batch_size
        
        while end < L:
            x_train, y_train = loadFromList(imgs, labels, start, end)
            start += batch_size
            end += batch_size
            yield (x_train, y_train)
def valid_loader(dataset, batch_size):
     imgs = list(dataset['valid_img'])
     labels = list(dataset['valid_label'])
     
     L = len(imgs)
     while(True):
         start = 0
         end = batch_size
         
         while end < L:
             x_val, y_val = loadFromList(imgs, labels, start, end)
             start += batch_size
             end += batch_size
             yield (x_val, y_val)   


if __name__ == '__main__':
    dataset = load_train_dataset()
    loader = train_loader(dataset, 1)
    x, y = loader.next()
#    print(x)
    print(x.shape)
