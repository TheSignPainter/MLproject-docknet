import keras
import glob
import os
from utils import load_dataset, loader
import constant as c
from models import convolutional_model
import numpy as np
def main():
    #load the most recent h5
    directory = c.best_checkpoint_dir
    #best_checkpoints = glob.glob('./checkpoints/model_900_0.49655.h5')
    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"), 
        map(lambda f: os.path.join(directory, f), os.listdir(directory))),
        key=os.path.getmtime)
    best_model = files[-1]
    print("Loaded %s" % (best_model))

    input_shape = (256, 256, 3)
    model = convolutional_model(input_shape, batch_size = c.batch_size)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    model.load_weights(best_model)
    test_dataset = load_dataset('test')
    loader_test = loader(test_dataset, c.batch_size)
    test_steps = len(test_dataset['label']) / c.batch_size
    
    accs = []
    for i in range(test_steps):
        x_test, y_test = loader_test.next()
        _loss, _acc = model.test_on_batch(x_test, y_test)
        accs.append(_acc)
    print(np.mean(np.array(accs)))

main()
