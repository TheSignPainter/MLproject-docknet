# Notice

Exclude `dataset_docknet/data/test/docks/5276_20468_13575.jpg` for the irregular size

# Results

|  Model   | Method                                          | Epoch |  LR   | Accuracy |
| :------: | :---------------------------------------------- | :---: | :---: | :------: |
| ResNet18 | -                                               |  10   | 1e-2  |  92.75%  |
| ResNet18 | Pretrain + Only train FC                        |  10   | 1e-2  |  90.71%  |
| ResNet18 | Pretrain + Only train FC                        |  10   | 1e-3  |  90.89%  |
| ResNet18 | Pretrain + Joint train                          |  10   | 1e-3  |  96.10%  |
| ResNet18 | Pretrain + Joint train + Random Flip            |  10   | 1e-3  |  97.40%  |
| ResNet18 | Pretrain + Joint train + Random Flip + LSoftmax |  10   | 1e-3  |  97.77%  |