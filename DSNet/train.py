import torch, os
from torch._C import OptionalType
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from dataset import get_loader, Rescale, ToTensor, Normalize
import os, argparse
from model import DSNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-lr", "--learning_rate", default=0.001)
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--trainsize', type=int, default=352, help='input size')
parser.add_argument('--trainset', type=str, default='TrainSet_Video', help='training  dataset')
opt = parser.parse_args()
    #args = parser.parse_args()

    # data preparing, set your own data path here
data_path = '/content/drive/MyDrive/dataset/'
image_root = data_path + opt.trainset + '/RGB/'
gt_root = data_path + opt.trainset + '/GT/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batch_size, trainsize=opt.trainsize)
total_step = len(train_loader)



def train(epochs, batch_size, learning_rate):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = F.binary_cross_entropy(output, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print("Loss: {:.4f}\n".format(epoch_loss))

    os.makedirs("models", exist_ok=True)
    torch.save(model, "models/model.pt")


if __name__ == "__main__":
    
    train(epochs=opt.epoch,
          batch_size=opt.batch_size,
          learning_rate=opt.lr)
