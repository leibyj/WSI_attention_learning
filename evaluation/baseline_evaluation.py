import torch
import os
from sys import argv
from torch.utils.data import DataLoader
from model import ATTN_net
from dataloader import PatientBagDataset, collate_patient_bags
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

def majority_mean(pred):
  # pred is binary classification probabilities. Use mean as bag level probability
  # pred = torch.round(pred)
  votes = torch.sum(pred)
  return (votes/pred.size(0)).detach().cpu()
  

def majority_median(pred):
  return (torch.median(pred)).detach().cpu()
  

def wsi_evaluation(model, dataloader_list, device):
  ''' 
  Function for performing evaluation of WSIs.
  Returns: Majority vote for each WSI in the list of dataloaders and attns for each tile
  '''
  model.eval()
  with torch.no_grad():
    all_votes_mean = torch.tensor([])
    all_votes_median = torch.tensor([])
    # all_attns = []
    for dl in dataloader_list:
      slide_out = torch.tensor([], device=device)
      pat_id, _, _, _= next(iter(dl))
      print("Evaluation for", pat_id)
      for pat_id, tile_paths, data, label in dl:
        
        all_data = torch.tensor([], device=device)
        # basline_model only takes in tiles as input:
        # convert bag = BxNxCxHxW into B*NxCxHxW
        for bag in data:
          all_data = torch.cat((all_data, bag.to(device)))
        
        out = model(all_data)
        slide_out = torch.cat((slide_out, out), 0)
        

      vote_mean = majority_mean(slide_out)
      vote_median = majority_median(slide_out)
      
      all_votes_mean = torch.cat((all_votes_mean, vote_mean.unsqueeze(0)), 0)
      all_votes_median = torch.cat((all_votes_median, vote_median.unsqueeze(0)), 0)
      
  
  return all_votes_mean, all_votes_median

def wsi_metrics(mss_pred, msi_pred):
  all_pred = torch.cat((mss_pred, msi_pred))
  # create labels
  mss_labels = torch.zeros_like(mss_pred)
  msi_labels = torch.ones_like(msi_pred)
  all_labels = torch.cat((mss_labels, msi_labels))

  auc = roc_auc_score(all_labels, all_pred)
  ap = average_precision_score(all_labels, all_pred)
  all_pred = torch.round(all_pred)
  f1 = f1_score(all_labels, all_pred)

  return auc, ap, f1

if __name__ == '__main__':
  # clean this later...
  trained_model = str(argv[1])
  dataset = str(argv[2])

  model_path = f"/home/jleiby/MSI_pred/models/{trained_model}.pth"
  model = torch.load(model_path)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  data_path = f"/home/jleiby/MSI_pred/data/{dataset}/"

  data_msi = os.listdir(os.path.join(data_path, "MSI"))
  data_mss = os.listdir(os.path.join(data_path, "MSS"))

  # get bags for eval data
  eval_msi_data = [PatientBagDataset(data_path, label="MSI", pat_id=x, num_tiles=8) for x in data_msi]
  eval_mss_data = [PatientBagDataset(data_path, label="MSS", pat_id=x, num_tiles=8) for x in data_mss]
  # set up list of data loaders....
  eval_msi_loaders = [DataLoader(x, batch_size=12, shuffle=True, collate_fn=collate_patient_bags) for x in eval_msi_data]
  eval_mss_loaders = [DataLoader(x, batch_size=12, shuffle=True, collate_fn=collate_patient_bags) for x in eval_mss_data]

  # predict and get performance metrics
  msi_predictions_mean, msi_predictions_median = wsi_evaluation(model, eval_msi_loaders, device)
  mss_predictions_mean, mss_predictions_median = wsi_evaluation(model, eval_mss_loaders, device)
  print("MSI mean predictions:", msi_predictions_mean)
  print("MSI median predictions:", msi_predictions_median)
  
  print("MSS mean predictions:", mss_predictions_mean)
  print("MSS median predictions:", mss_predictions_median)
  

  auc_mean, ap_mean, f1_mean = wsi_metrics(mss_predictions_mean, msi_predictions_mean)
  print(f"Validation (mean vote) auc: {auc_mean:.5f}\n")
  print(f"Validation (mean vote) average precision: {ap_mean:.5f}\n")
  print(f"Validation (mean vote) F1: {f1_mean:.5f}\n")
  auc_median, ap_median, f1_median = wsi_metrics(mss_predictions_median, msi_predictions_median)
  print(f"Validation (median vote) auc: {auc_median:.5f}\n")
  print(f"Validation (median vote) average precision: {ap_median:.5f}\n")
  print(f"Validation (median vote) F1: {f1_median:.5f}\n")




