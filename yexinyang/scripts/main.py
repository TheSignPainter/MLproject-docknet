import os, time
import numpy as np
import logging
import fire

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from model import *
from dataset import *


def train(dataloader, model, optimizer, criterion, device):
    epoch_loss = 0.0
    total_num = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device).squeeze()
        total_num += len(data)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / total_num


def eval(dataloader, model, device):
    c = 0
    total_num = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device).squeeze()
            total_num += len(data)
            out = model(data)
            predicted = torch.max(out, 1)[1]
            c += (predicted == target).sum().item()
    return c * 100.0 / total_num


def main(**kwargs):
    data_dir = kwargs.get('data_dir', '../../dataset_docknet/data')
    model_dir = kwargs.get('model_dir', 'models')
    log_file = kwargs.get('log_file', 'LOG')
    epoch = kwargs.get('epoch', 10)
    batch_size = kwargs.get('batch_size', 32)
    lr = kwargs.get('lr', 1e-3)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    formatter = logging.Formatter(
        "[ %(levelname)s: %(asctime)s ] - %(message)s"
    )
    logging.basicConfig(level=logging.DEBUG,
                        format="[ %(levelname)s: %(asctime)s ] - %(message)s")
    logger = logging.getLogger("Pytorch")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(kwargs)

    train_dataset = DockDataset(featdir=os.path.join(data_dir, 'train'))
    cv_dataset = DockDataset(featdir=os.path.join(data_dir, 'valid'))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )

    cv_loader = DataLoader(
        cv_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=False, progress=True).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(model)

    best_acc = 0.0

    time_list = []
    for e in range(epoch):
        model.train()
        train_loss = train(train_loader, model, optimizer, criterion, device)
        model.eval()
        cv_acc = eval(cv_loader, model, device)

        message = { f"[*] Epoch: [{e+1:3d}/{epoch:3d}] - "
                    f"Training Loss: {train_loss:.5f}, "
                    f"CV Acc: {cv_acc:.2f}%" }

        logger.info(message)

        torch.save(model, os.path.join(model_dir, f"checkpoint_{e+1}.pth"))

        if cv_acc >= best_acc:
            torch.save(model, os.path.join(model_dir, f"model_best.pth"))
            best_acc = cv_acc


    eval_dataset = DockDataset(featdir=os.path.join(data_dir, 'test'), is_train=False, shuffle=False)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model_best = torch.load(os.path.join(model_dir, f"model_best.pth"))
    model_best.eval()
    eval_acc = eval(eval_loader, model_best, device)
    logger.info(f"Test Accuracy is: {eval_acc:.2f}%")


if __name__ == '__main__':
    fire.Fire(main)