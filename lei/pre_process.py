import os
import scipy.ndimage
import constant as c
import numpy as np
import glob
def jpg_2_npy(filename, target_dir, s):
    target_name = os.path.join(target_dir, s + '_' +  os.path.basename(filename)[:-4] + '.npy')
    img = scipy.ndimage.imread(filename)
    print(target_name)
    np.save(target_name, img)
def main():
    target_dir = '/home/lei/2019/ml/data_npy/valid/'
    #first docks
    #files = glob.glob(c.dataset_train + '/docks/*.jpg')
    files = glob.glob(c.dataset_valid + '/notdocks/*.jpg')
    print("training files docks %s" % (len(files)))
    s = 'notdocks'
    for f in files:
        jpg_2_npy(f, target_dir, s)
if __name__ == '__main__':
    main()
