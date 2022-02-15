import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold
from torchvision.datasets import ImageFolder
from torchvision import transforms
from dataloader import FoldBagDataset, collate_bag_batches

import sys
from sys import argv
import time
import copy

from model import Baseline_model

start_time = time.time()
l2 = float(argv[1])
saved_model = str(argv[2])
fold = int(argv[3])
print("Fold:", fold)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Set up the indexes for 5-fold CV
inds = range(0, 46704)
kf = KFold(n_splits=5, shuffle=True, random_state=66)
train_inds = []
val_inds = []
for tr, val in kf.split(inds):
  train_inds.append(tr)
  val_inds.append(val)

# Dataloaders
train_data = FoldBagDataset("/home/jleiby/MSI_pred/data", subset="train", num_tiles=8, idx=train_inds[fold-1])
train_data_loader = DataLoader(train_data, batch_size=12, shuffle=True, collate_fn=collate_bag_batches)

val_data = FoldBagDataset("/home/jleiby/MSI_pred/data", subset="train", num_tiles=8, idx=val_inds[fold-1])
val_data_loader = DataLoader(val_data, batch_size=12, shuffle=True, collate_fn=collate_bag_batches)

epochs = 10

model = Baseline_model().to(device)
opt = optim.SGD(model.parameters(), lr=0.001, weight_decay=l2)
criterion = nn.BCELoss().to(device)
train_loss = []

shuffle_bags = True
top_val_auc = 0
opt_model = None
opt_epoch = 0


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
    loss = criterion(out, label.unsqueeze(1))
    loss.backward()
    opt.step()
    train_loss.append(loss.item())

    running_acc += sum(torch.round(out).eq(label.unsqueeze(1)))
    ct += out.size(0)

    if i % 100 == 99:
      print("Minibatch: ", i+1)
      print("Training loss: ", loss.item())
      print("Training accuracy: ", running_acc.item()/ct)


 # validation check each epoch...
  val_acc = 0
  val_ct = 0
  model.eval()
  all_labels = torch.tensor([], device=device)
  all_outputs = torch.tensor([], device=device)
  with torch.no_grad():
    for data, label in val_data_loader:
      all_data = torch.tensor([], device=device)
    # basline_model only takes in tiles as input:
    # convert bag = list of length b of NxCxHxW tensors into B*NxCxHxW
      for bag in data:
        all_data = torch.cat((all_data, bag.to(device)))
    # label is tensors Bx1, convert to tensor B*Nx1
      label = torch.repeat_interleave(label, data[0].shape[0])
      label = label.float().to(device)
      out = model(all_data)
    
      val_acc += sum(torch.round(out).eq(label.unsqueeze(1))).item()
      val_ct += out.size(0)
     # AUC
      all_labels = torch.cat((all_labels, label), 0)
      all_outputs = torch.cat((all_outputs, out), 0)
  auc = roc_auc_score(all_labels.detach().cpu().numpy(), all_outputs.squeeze(1).detach().cpu().numpy())
  # F1
  all_outputs = torch.round(all_outputs)
  f1 = f1_score(all_labels.detach().cpu().numpy(), all_outputs.squeeze(1).detach().cpu().numpy())
  print("Epoch: ", epoch+1)
  print("Training accuracy: ", running_acc.item()/ct)
  print("Validation accuracy: ", val_acc/val_ct)
  print(f"Validation auc: {auc:.5f}\n")
  print(f"Validation F1: {f1:.5f}\n")
  if auc > top_val_auc:
    top_val_auc = auc
    opt_model = copy.deepcopy(model)
    opt_epoch = epoch
    print("New optimal model at epoch:", opt_epoch+1)

  # shuffle training bags...
  if epoch % 2 == 1:
    if shuffle_bags:
      train_data = FoldBagDataset("/home/jleiby/MSI_pred/data", subset="train", num_tiles=8, idx=train_inds[fold-1])
      train_data_loader = DataLoader(train_data, batch_size=12, shuffle=True, collate_fn=collate_bag_batches)

torch.save(opt_model, f"/home/jleiby/MSI_pred/models/{saved_model}_fold_{fold}.pth")

print('execution time:', (time.time()-start_time)/60)


