# Description

Separate docks from normal crops using a CNN model. The following tricks are used to boost the performance:

 - **A powerful ResNet18 model.** You know it is powerful.
 - **A decent normalization on input features.** Some *magic numbers* are used for normalization (you may refer to PyTorch tutorial, the normalization way is common on ImageNet). Of course it is necessary if you decide to use a ImageNet pretrained model.
 - **The usage of pretrained model on ImageNet for initialization.** Instead of only training the last FC layer (the performance degradation is observed, the cause may be the limited representation capability of a single linear layer), the whole model is jointly trained with a smaller learning rate.
 - **Randomly flipping the original images.** Since the data is quite limited, the data augmentation is necessary. The input image is randomly horizontally/vertically (can be both, p=0.5) flipped to train a more robust model.
 - **Large-Margin Softmax Loss for Convolutional Neural Networks.** Ummm, actually not so effective. A little even not steady boost (but helpful for report writing). L-Softmax proposes a modified softmax classification method to increase the inter-class separability and intra-class compactness. The code can be referred to [Github](https://github.com/amirhfarzaneh/lsoftmax-pytorch). The original paper can be referred to [pdf in arxiv](https://arxiv.org/pdf/1612.02295.pdf).

# Usage

## Train
`python scripts/main.py train --data_dir=path/to/data/ --model_dir=path/to/model/ --log_file=LOG --epoch=10 --batch_size=32 --lr=1e-3`

## Test
`python scripts/main.py test --data_dir=path/to/data/ --model_dir=path/to/model/`

# Detailed configurations

 - SGD (lr=1e-3, momentum=0.9)
 - CrossEntropy Loss
 - Batchsize=32
 - LSoftmax margin=2

# Notice

Exclude `dataset_docknet/data/test/docks/5276_20468_13575.jpg` for its irregular size

# Results

|  Model   | Method                                          | Epoch |  LR   | Accuracy |
| :------: | :---------------------------------------------- | :---: | :---: | :------: |
| ResNet18 | -                                               |  10   | 1e-2  |  92.75%  |
| ResNet18 | Pretrain + Only Train FC                        |  10   | 1e-2  |  90.71%  |
| ResNet18 | Pretrain + Only Train FC                        |  10   | 1e-3  |  90.89%  |
| ResNet18 | Pretrain + Joint Train                          |  10   | 1e-3  |  96.10%  |
| ResNet18 | Pretrain + Joint Train + Random Flip            |  10   | 1e-3  |  97.40%  |
| ResNet18 | Pretrain + Joint Train + Random Flip + LSoftmax |  10   | 1e-3  |  97.77%  |

# Some statistics helpful for the report

## Training loss and CV accuracy

### ResNet18 baseline
```
{'[*] Epoch: [  1/ 10] - Training Loss: 0.01493, CV Acc: 83.44%'}
{'[*] Epoch: [  2/ 10] - Training Loss: 0.01244, CV Acc: 86.88%'}
{'[*] Epoch: [  3/ 10] - Training Loss: 0.00968, CV Acc: 90.00%'}
{'[*] Epoch: [  4/ 10] - Training Loss: 0.00734, CV Acc: 85.47%'}
{'[*] Epoch: [  5/ 10] - Training Loss: 0.00675, CV Acc: 84.53%'}
{'[*] Epoch: [  6/ 10] - Training Loss: 0.00580, CV Acc: 91.09%'}
{'[*] Epoch: [  7/ 10] - Training Loss: 0.00491, CV Acc: 91.09%'}
{'[*] Epoch: [  8/ 10] - Training Loss: 0.00394, CV Acc: 92.19%'}
{'[*] Epoch: [  9/ 10] - Training Loss: 0.00341, CV Acc: 89.84%'}
{'[*] Epoch: [ 10/ 10] - Training Loss: 0.00149, CV Acc: 90.16%'}
```

### ResNet18 (Pretrain + Joint Train)
```
{'[*] Epoch: [  1/ 10] - Training Loss: 0.00860, CV Acc: 94.53%'}
{'[*] Epoch: [  2/ 10] - Training Loss: 0.00370, CV Acc: 95.47%'}
{'[*] Epoch: [  3/ 10] - Training Loss: 0.00189, CV Acc: 95.78%'}
{'[*] Epoch: [  4/ 10] - Training Loss: 0.00118, CV Acc: 95.78%'}
{'[*] Epoch: [  5/ 10] - Training Loss: 0.00065, CV Acc: 95.78%'}
{'[*] Epoch: [  6/ 10] - Training Loss: 0.00037, CV Acc: 95.16%'}
{'[*] Epoch: [  7/ 10] - Training Loss: 0.00025, CV Acc: 95.94%'}
{'[*] Epoch: [  8/ 10] - Training Loss: 0.00019, CV Acc: 95.62%'}
{'[*] Epoch: [  9/ 10] - Training Loss: 0.00016, CV Acc: 95.78%'}
{'[*] Epoch: [ 10/ 10] - Training Loss: 0.00020, CV Acc: 96.09%'}
```

### ResNet18 (Pretrain + Joint Train + Random Flip)
```
{'[*] Epoch: [  1/ 10] - Training Loss: 0.00821, CV Acc: 94.84%'}
{'[*] Epoch: [  2/ 10] - Training Loss: 0.00436, CV Acc: 95.78%'}
{'[*] Epoch: [  3/ 10] - Training Loss: 0.00303, CV Acc: 95.78%'}
{'[*] Epoch: [  4/ 10] - Training Loss: 0.00291, CV Acc: 96.56%'}
{'[*] Epoch: [  5/ 10] - Training Loss: 0.00229, CV Acc: 96.56%'}
{'[*] Epoch: [  6/ 10] - Training Loss: 0.00212, CV Acc: 96.72%'}
{'[*] Epoch: [  7/ 10] - Training Loss: 0.00159, CV Acc: 95.78%'}
{'[*] Epoch: [  8/ 10] - Training Loss: 0.00138, CV Acc: 96.56%'}
{'[*] Epoch: [  9/ 10] - Training Loss: 0.00103, CV Acc: 96.41%'}
{'[*] Epoch: [ 10/ 10] - Training Loss: 0.00101, CV Acc: 96.41%'}
```

### ResNet18 (Pretrain + Joint Train + Random Flip + LSoftmax)
```
{'[*] Epoch: [  1/ 10] - Training Loss: 0.01043, CV Acc: 95.16%'}
{'[*] Epoch: [  2/ 10] - Training Loss: 0.01237, CV Acc: 96.41%'}
{'[*] Epoch: [  3/ 10] - Training Loss: 0.01879, CV Acc: 94.84%'}
{'[*] Epoch: [  4/ 10] - Training Loss: 0.01562, CV Acc: 95.00%'}
{'[*] Epoch: [  5/ 10] - Training Loss: 0.00782, CV Acc: 97.03%'}
{'[*] Epoch: [  6/ 10] - Training Loss: 0.00461, CV Acc: 97.34%'}
{'[*] Epoch: [  7/ 10] - Training Loss: 0.00297, CV Acc: 97.34%'}
{'[*] Epoch: [  8/ 10] - Training Loss: 0.00275, CV Acc: 97.34%'}
{'[*] Epoch: [  9/ 10] - Training Loss: 0.00214, CV Acc: 97.34%'}
{'[*] Epoch: [ 10/ 10] - Training Loss: 0.00182, CV Acc: 97.03%'}
```