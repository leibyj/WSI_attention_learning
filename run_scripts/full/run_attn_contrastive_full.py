import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold
import sys
from sys import argv
import time
import copy

from model import ATTN_net, ATTN_net_con, ContrastiveLoss
from dataloader import FoldBagDataset, collate_bag_batches, BagDataset 


start_time = time.time()

# Arg 1: L2; Arg 2: Model name to save
l2 = float(argv[1])
saved_model = str(argv[2])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


train_data = BagDataset("/home/jleiby/MSI_pred/data", subset="train", num_tiles=8)
train_data_loader = DataLoader(train_data, batch_size=12, shuffle=True, collate_fn=collate_bag_batches)

epochs = 10

model = ATTN_net_con().to(device)
opt = optim.SGD(model.parameters(), lr=0.001, weight_decay=l2)
criterion = nn.BCELoss().to(device)
contrast = ContrastiveLoss(0.5).to(device)
train_loss = []

shuffle_bags = True

for epoch in range(epochs):
  print(epoch+1)
  running_acc = 0
  ct = 0
  model.train()
  for i, (data, label) in enumerate(train_data_loader):
    label = label.to(device)
    # data: list length B of NxCxHxW tensors, N = #tiles/bag
    # ---> 
    # data: BxNxCxHxW tensor, B = batch size
    data = torch.stack(data).to(device)
    opt.zero_grad()
    attns, out, in_emb, bag_emb = model(data, device)
    
    loss = criterion(out, label.unsqueeze(1)) + contrast(in_emb, bag_emb, device)
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

torch.save(model, f"/home/jleiby/MSI_pred/models/{saved_model}.pth")

print('execution time:', (time.time()-start_time)/60)
