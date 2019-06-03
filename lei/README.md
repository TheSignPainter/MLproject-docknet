# Overview

In this work, a simple convolutional neural network is trained to separate the docks from normal crops.

# Methodology

## Preprocessing
We normalized the image data. shuffled before training.

## Model
The model we used is a simple variation of resnet model.

layer name | kernel size | filters | # of residual blocks 
:--:|:--:|:--:|:--:|
conv\_1 | 3x3 | 64 | 3
conv\_2 | 3x3 | 128 | 3
conv\_3 | 3x3 | 256 | 3
conv\_4 | 3x3 | 512 | 3
avgPolling | 2x2 | None | None
Flattern | | |
sigmoid | | |

# Usage
First clone the repo, if you are to train the model youself, you should download the dataset, and specify the `train dataset`, `validation dataset` and `test dataset` path in `constant.py`

To train a model, just run `python train.py`.

To test a model on the test dataset, just run `python test.py`

# Evaluation results
On this simple CNN model, we can reach a 91.9% accuracy on the test dataset.
