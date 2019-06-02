import constant as c
import pandas as pd
from glob import glob
import os
import numpy as np
from keras.utils.np_utils import to_categorical
import re
import logging
def load_dataset(mode):
    #load trainning and validation
    if mode == 'train':
        dataset_name = c.npy_train
    elif mode == 'valid':
        dataset_name = c.npy_valid
    elif mode == 'test':
        dataset_name = c.npy_test
    dataset = pd.DataFrame()
    files = glob(dataset_name + '*.npy')
    #files = glob(c.npy_train + 'docks*.npy')[:1000]
    #files2 = glob(c.npy_train + 'notdocks*.npy')[:1000]
    #files = files + files2
    np.random.shuffle(files)
    dataset['img'] = files
    dataset['label'] = dataset['img'].apply(lambda x: os.path.basename(x).split('_')[0])
    print("Load %d training" % (len(dataset['label'])))
    return dataset

def loadFromList(imgs, labels, start, end):
    x = []
    y = []
    for i in range(start, end):
        img = np.load(imgs[i])
        img = img.astype(np.float)/ 256
        # shape?
        x.append(img)
        label = labels[i]
        if (label =='docks'):
            y.append(1)
        else:
            y.append(0)
    x = np.asarray(x)
    y = np.asarray(y)
    y = np.reshape(y,(end-start, 1))
    #y = to_categorical(y,num_classes = 2)
    return x,y

def loader(dataset, batch_size):
    imgs = list(dataset['img'])
    labels = list(dataset['label'])

    L = len(imgs)

    while(True):
        start = 0
        end = batch_size
        
        while end < L:
            x_train, y_train = loadFromList(imgs, labels, start, end)
            start += batch_size
            end += batch_size
            yield (x_train, y_train)

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
def get_last_checkpoint_if_any(checkpoint_folder):
    if not os.path.exists(checkpoint_folder):
        os.makedir(checkpoint_folder)
#    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob('{}/*.h5'.format(checkpoint_folder))
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]
def create_dir_and_delete_content(directory):
    if not os.path.exists(directory):
        os.makedir(directory)
    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"), 
        map(lambda f: os.path.join(directory, f), os.listdir(directory))),
        key=os.path.getmtime)
    for file in files[:-4]:
        logging.info("removing old model: {}".format(file))
        os.remove(file)
if __name__ == '__main__':
    dataset = load_train_dataset()
    loader = train_loader(dataset, 1)
    x, y = loader.next()
#    print(x)
    print(x.shape)
    print(np.max(x))
    print(np.min(x))
