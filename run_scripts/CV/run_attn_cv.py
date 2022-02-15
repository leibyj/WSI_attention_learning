import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import KFold
import sys
from sys import argv
import time
import copy
import albumentations as A

from model import ATTN_net, ATTN_net_ShuffleNetV2, ATTN_net_res34_late, ATTN_net_ciga_late
from dataloader import FoldBagDataset, collate_bag_batches #, PatientBagDataset, collate_patient_bags
# from evaluation import wsi_evaluation

start_time = time.time()

# clean this up later...
l2 = float(argv[1])
fold = int(argv[2])
print("Fold:", fold)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

### Set up the indexes for 5-fold CV
inds = range(0, 46704)
kf = KFold(n_splits=5, shuffle=True, random_state=666)
train_inds = []
val_inds = []
for tr, val in kf.split(inds):
  train_inds.append(tr)
  val_inds.append(val)

# Dataloaders
train_data = FoldBagDataset("/home/jleiby/MSI_pred/data", subset="train", num_tiles=8, transform=None, idx=train_inds[fold-1])
train_data_loader = DataLoader(train_data, batch_size=12, shuffle=True, collate_fn=collate_bag_batches)

val_data = FoldBagDataset("/home/jleiby/MSI_pred/data", subset="train", num_tiles=8, idx=val_inds[fold-1])
val_data_loader = DataLoader(val_data, batch_size=12, shuffle=True, collate_fn=collate_bag_batches)

epochs = 10

model = ATTN_net(device).to(device)
mod_name = str(model).split("(")[0]

saved_model = f"class_{mod_name}_{l2}_fold_{fold}"
print(saved_model)

opt = optim.SGD(model.parameters(), lr=0.001, weight_decay=l2)
criterion = nn.BCELoss().to(device)
train_loss = []

shuffle_bags = True
top_val_ap = 0
opt_model = None
opt_epoch = 0

for epoch in range(epochs):
  print(epoch+1)
  running_acc = 0
  ct = 0
  model.train()
  opt.zero_grad()
  for i, (data, label) in enumerate(train_data_loader):
    label = label.to(device)
    # data: list length B of NxCxHxW tensors, N = #tiles/bag
    # ---> 
    # data: BxNxCxHxW tensor, B = batch size
    data = torch.stack(data).to(device)
    opt.zero_grad()
    attns, out = model(data, device)
    
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

  # validation check each epoch...
  val_acc = 0
  val_ct = 0
  model.eval()
  all_labels = torch.tensor([], device=device)
  all_outputs = torch.tensor([], device=device)
  with torch.no_grad():
    for data, label in val_data_loader:
      label = label.to(device)
      # data: list length B of NxCxHxW tensors, N = num tiles in bag
      # ---> 
      # data: BxNxCxHxW tensor, B = batch size
      data = torch.stack(data).to(device)
      _, out = model(data, device)
      val_acc += sum(torch.round(out).eq(label.unsqueeze(1))).item()
      val_ct += out.size(0)
      # AUC
      all_labels = torch.cat((all_labels, label), 0)
      all_outputs = torch.cat((all_outputs, out), 0)
  auc = roc_auc_score(all_labels.detach().cpu().numpy(), all_outputs.squeeze(1).detach().cpu().numpy())
  ap = average_precision_score(all_labels.detach().cpu().numpy(), all_outputs.squeeze(1).detach().cpu().numpy())
  # F1
  all_outputs = torch.round(all_outputs)
  f1 = f1_score(all_labels.detach().cpu().numpy(), all_outputs.squeeze(1).detach().cpu().numpy())

  print("Epoch: ", epoch+1)
  print("Training accuracy: ", running_acc.item()/ct)
  print("Validation accuracy: ", val_acc/val_ct)
  print(f"Validation auc: {auc:.5f}\n")
  print(f"Validation ap: {ap:.5f}\n")
  print(f"Validation F1: {f1:.5f}\n")
  if ap > top_val_ap:
    top_val_ap = ap
    opt_model = copy.deepcopy(model)
    opt_epoch = epoch
    print("New optimal model at epoch:", opt_epoch+1)

  # shuffle bags...
  if epoch % 2 == 1:
    if shuffle_bags:
      train_data = FoldBagDataset("/home/jleiby/MSI_pred/data", subset="train", num_tiles=8, transform=None, idx=train_inds[fold-1])
      train_data_loader = DataLoader(train_data, batch_size=12, shuffle=True, collate_fn=collate_bag_batches)

torch.save(opt_model, f"/home/jleiby/MSI_pred/models/{saved_model}.pth")

print(opt_epoch)
print('execution time:', (time.time()-start_time)/60)
