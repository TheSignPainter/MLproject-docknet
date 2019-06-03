import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

def getdata(path, shuffle=True):
    """
    Generate dataset.
    """

    x = []
    y = []
    docks_path = os.path.join(path, "docks")
    for i in os.listdir(docks_path):
        # print(i)
        img_path = os.path.join(docks_path, i)
        y.append(1)
        img = img_to_array(load_img(img_path, target_size=(256,256)))
        # print("Shape:", img.shape)
        x.append(img)

    nodocks_path = os.path.join(path, "notdocks")
    for i in os.listdir(nodocks_path):
        # print(i)
        img_path = os.path.join(nodocks_path, i)
        y.append(0)
        img = img_to_array(load_img(img_path))
        # print("Shape:", img.shape)
        x.append(img)
    assert len(x) == len(y)
    # new_x = np.expand_dims(x[0], axis=0)
    # print(new_x.shape)
    # for i in range(1, len(x)):
    #     print(x[i].shape)
    #     new_x = np.concatenate((new_x, np.expand_dims(x[i], axis=0)), axis = 0)
    # print(x)
    # print(y)
    x = np.asarray(x)
    y = np.asarray(y)
    print("X shape:", x.shape)
    print("Y shape:", y.shape)
    if shuffle:
        shuff = [i for i in range(len(x))]
        np.random.shuffle(shuff)
        # print(shuff)
        x = x[shuff, :, :, :]
        y = y[shuff]

    return x, y


if __name__ == "__main__":
    getdata("../data/train")
    getdata("../data/valid")