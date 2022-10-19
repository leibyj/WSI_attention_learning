import torch
from torch.utils.data import DataLoader
from evaluation import get_tile_attns
from model import MIL_net
from dataloader import PatientBagDataset, collate_patient_bags

from sys import argv
import os
import numpy as np

trained_model = str(argv[1])
labels = ["MSI", "MSS"]

model_path = f"/home/jleiby/MSI_pred/models/{trained_model}.pth"
model = torch.load(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/home/jleiby/MSI_pred/data/PAIP/"

for label in labels:
	print(label)
	path = os.path.join(data_path, label)
	pat_ids = os.listdir(path)
	for pat_id in pat_ids:
		print(f"Getting attentions for {pat_id}")
		dataset = PatientBagDataset(data_path, label=label, pat_id=pat_id, num_tiles=4)
		dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_patient_bags)
		attns = get_tile_attns(model, dataloader, device)
		# print(attns)
		print(len(attns))
		# save 
		np.save(f"attention_scores/{pat_id}_attns.npy", attns) 
