from models import convolutional_model
import constant as c
from utils import load_dataset, loader
from utils import get_last_checkpoint_if_any, create_dir_and_delete_content
import logging
from time import time
import numpy as np
import keras.optimizers as optimizers

def main():
    num_epoches = 200
    #1. first load data
    train_dataset = load_dataset('train')
    valid_dataset = load_dataset('valid')
    loader_train = loader(train_dataset, c.batch_size)
    loader_valid = loader(valid_dataset, c.batch_size)
    test_steps = len(valid_dataset['label'])/c.batch_size
    print(len(train_dataset['label']))
    logging.info("training %d valid %d" % (len(train_dataset['label']), len(valid_dataset['label'])))
    #2. then load model
    input_shape = (256,256,3)
    model = convolutional_model(input_shape, batch_size = c.batch_size)

    #logging.info(model.summary())
    opt = optimizers.Adam(lr = 0.0001)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=['accuracy'])

    grad_steps = 0
    last_checkpoint = get_last_checkpoint_if_any(c.checkpoint_folder)

    steps_per_epoch = len(train_dataset['label']) / c.batch_size

    #last_checkpoint = None
    best_acc = 0
    print(last_checkpoint)
    if last_checkpoint is not None:
        logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
        print('loadding checkpoing %s' %(last_checkpoint))
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split('_')[-2])
        logging.info('[DONE]')
    for i in range(num_epoches):
        print("Epoch %d"%(i))
        for j in range(steps_per_epoch):
            orig_time = time()
            x_train, y_train = loader_train.next()
            [loss, acc] = model.train_on_batch(x_train, y_train)  # return [loss, acc]
            logging.info('Train Steps:{0}, Time:{1:.2f}s, Loss={2}, Accuracy={3}'.format(grad_steps,time() - orig_time, loss, acc))
            if (grad_steps % 100 == 0):
                print("Training epoch   [%d] steps  [%d]    acc [%f]      loss [%f]" % (i, grad_steps, acc, loss))
            with open(c.checkpoint_folder + "/train_loss_acc.txt", "a") as f:
                f.write("{0},{1},{2}\n".format(grad_steps, loss, acc))
                
            if grad_steps % c.test_per_epoches == 0:
                losses = []; accs = []
                for ss in range(test_steps):
                    x_valid, y_valid = loader_valid.next()
                    [loss, acc] = model.test_on_batch(x_valid, y_valid)
                    losses.append(loss); accs.append(acc)
                loss = np.mean(np.array(losses)); acc = np.mean(np.array(accs))
                print("Test at epoch    ", i, "steps    ", grad_steps , "avg loss   ", loss, "avg acc   ", acc)
                logging.info('Test the Data ---------- Steps:{0}, Loss={1}, Accuracy={2}, '.format(grad_steps, loss, acc))
                with open(c.checkpoint_folder + "/test_loss_acc.txt", "a") as f:
                    f.write("{0},{1},{2}\n".format(grad_steps, loss, acc))
                    if grad_steps  % c.save_per_epoches == 0:
                        create_dir_and_delete_content(c.checkpoint_folder)
                        model.save_weights('{0}/model_{1}_{2:.5f}.h5'.format(c.checkpoint_folder, grad_steps, loss))
                # Save the best one
                if acc > best_acc:
                    best_acc = acc
                    create_dir_and_delete_content(c.best_checkpoint_dir)
                    model.save_weights(c.best_checkpoint_dir + '/best_model{0}_{1:.5f}.h5'.format(grad_steps, acc))
            grad_steps += 1
    #3. train, saving

    #4. finalize
    
    
main()
