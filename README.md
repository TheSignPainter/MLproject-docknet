# MLproject-docknet
Combine different methods to separate docks from normal crops.

# Dataset description

The dataset contains roughly 7000 jpeg images. In each split(train/val/test), image are categoried into 2 classes: docks, and notdocks. Image of each class is putted into different folders, so please shuffle them before training.

It is better to have a consistent metric among different models. So I took out \~600 images from /training folder, and made a new "test" dataset. For evaluating your model, please download the newly splitted dataset.

Download: Baidu Yun https://pan.baidu.com/s/1mFHZ3RnW-RauODt5NgMW7A Access code：ihot 

Credit of Docknet dataset goes to https://www.kaggle.com/gavinarmstrong/open-sprayer-images/

# Notice

Exclude `dataset_docknet/data/test/docks/5276_20468_13575.jpg` for the irregular size