import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from torchvision.datasets import ImageFolder
from torchvision import transforms
from dataloader import FoldBagDataset, collate_bag_batches, BagDataset

import sys
from sys import argv
import time

from model import Baseline_model

start_time = time.time()
# Arg 1: L2; Arg 2: Model name to save
l2 = float(argv[1])
saved_model = str(argv[2])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


train_data = BagDataset("/home/jleiby/MSI_pred/data", subset="train", num_tiles=8) #, idx=train_inds[fold-1])
train_data_loader = DataLoader(train_data, batch_size=12, shuffle=True, collate_fn=collate_bag_batches)

epochs = 10

model = torch.load("/home/jleiby/MSI_pred/pretrained_model/resnet18_full.pth")
model = nn.Sequential(model, nn.Sigmoid()).to(device)
opt = optim.SGD(model.parameters(), lr=0.001, weight_decay=l2)
criterion = nn.BCELoss().to(device)
train_loss = []

shuffle_bags = True

for epoch in range(epochs):
  print(epoch+1)
  running_acc = 0
  ct = 0
  model.train()
  for i, (data, label) in enumerate(train_data_loader):

    all_data = torch.tensor([], device=device)
    # basline_model only takes in tiles as input:
    # convert bag = list of length b of NxCxHxW tensors into B*NxCxHxW
    for bag in data:
      all_data = torch.cat((all_data, bag.to(device)))
    # label is tensors Bx1, convert to tensor B*Nx1
    label = torch.repeat_interleave(label, data[0].shape[0])
    label = label.float().to(device)

    opt.zero_grad()
    out = model(all_data)
    # _, out = model(data, device)
    loss = criterion(out, label.unsqueeze(1))
    loss.backward()
    opt.step()
    train_loss.append(loss.item())

    running_acc += sum(torch.round(out).eq(label.unsqueeze(1)))
    ct += out.size(0)

    if i % 200 == 199:
      print("Minibatch: ", i+1)
      print("Training loss: ", loss.item())
      print("Training accuracy: ", running_acc.item()/ct)
      

  # shuffle training bags...
  if epoch % 2 == 1:
    if shuffle_bags:
      train_data = BagDataset("/home/jleiby/MSI_pred/data", subset="train", num_tiles=8)
      train_data_loader = DataLoader(train_data, batch_size=12, shuffle=True, collate_fn=collate_bag_batches)

torch.save(opt_model, f"/home/jleiby/MSI_pred/models/{saved_model}.pth")

print('execution time:', (time.time()-start_time)/60)


