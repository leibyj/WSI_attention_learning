import glob
import cv2
import torch
from torch.utils.data import Dataset
import os
import random
from torchvision import transforms

class BagDataset(Dataset):
  def __init__(self, data_path, subset="train", num_tiles=10, transform=None):
    """
    BagDataset for class (label) level bags
    Dir structure: data_path/subset/label/*.png 
    """
    dir_path = os.path.join(data_path, subset)
    class_list = os.listdir(dir_path)
    self.data = []   
    for label in class_list:
      full_path = os.path.join(dir_path, label)
      self.data = self.data + self.partition(glob.glob(full_path + "/*.png"), num_tiles, label)

  def partition(self, list_in, n, label):
    random.shuffle(list_in)
    # For even bag sizes, make sure len(list) is divisible by n. if not, trim
    if len(list_in) % n > 0:
      list_in = list_in[:-(len(list_in) % n)]
    if label == "MSI":
      return [[list_in[i:i+n], 1] for i in range(0, len(list_in), n)]
    else: 
      return [[list_in[i:i+n], 0] for i in range(0, len(list_in), n)]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    tile_path, label = self.data[idx]
    label = torch.tensor(label)
    tile_list = []

    for tile in tile_path:
      img = cv2.imread(tile)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # if self.transform:
      #   img = self.transform(image = img)['image']

      img = torch.from_numpy(img).float()
      # put channel dim first for pytorch
      img = img.permute(2, 0, 1)
      tile_list.append(img)

    return tile_list, label

class FoldBagDataset(Dataset):
  """
  BagDataset for class (label) level bags with indexed tiles (folds for CV)
  Dir structure: data_path/subset/label/*.png 
  """
  def __init__(self, data_path, subset="train", num_tiles=10, transform=None, idx=[]):
    self.transform = transform
    print(subset)
    print(len(idx))
    print(idx)

    dir_path = os.path.join(data_path, subset)
    class_list = os.listdir(dir_path)
    self.data = []   
    for label in class_list:
      full_path = os.path.join(dir_path, label)
      # print(full_path)
      tile_path = glob.glob(full_path + "/*.png")
      tile_path = [tile_path[i] for i in idx]
      self.data = self.data + self.partition(tile_path, num_tiles, label)

  def partition(self, list_in, n, label):
    random.shuffle(list_in)
    # For even bag sizes, make sure len(list) is divisible by n. if not, trim
    if len(list_in) % n > 0:
      list_in = list_in[:-(len(list_in) % n)]
    if label == "MSI":
      return [[list_in[i:i+n], 1] for i in range(0, len(list_in), n)]
    else: 
      return [[list_in[i:i+n], 0] for i in range(0, len(list_in), n)]
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    tile_path, label = self.data[idx]
    label = torch.tensor(label)
    tile_list = []

    for tile in tile_path:
      img = cv2.imread(tile)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      if self.transform is not None:
        img = self.transform(image=img)['image']
      #... transform here?
      # if self.transform:
      #   img = self.transform(image = img)['image']

      img = torch.from_numpy(img).float()
      # put channel dim first for pytorch
      img = img.permute(2, 0, 1)

      tile_list.append(img)

    return tile_list, label
    # return tile_path, label

class PatientBagDataset(Dataset):
  """
  BagDataset for the patient level bags. Takes one pat_id at a time (used in evaluation scripts)
  Dir structure: data_path/label/pat_id/*.png 
  """
  def __init__(self, data_path, label="MSI", pat_id=None, num_tiles=10, transform=None):
    self.transform = transform
    dir_path = os.path.join(data_path, label, pat_id)
    self.data = []   
    self.data = self.data + self.partition(glob.glob(dir_path + "/*.png"), num_tiles, label, pat_id)
    # for label in class_list:
    #   full_path = os.path.join(dir_path, label)
    #   # print(full_path)
    
  def partition(self, list_in, n, label, pat_id):
    random.shuffle(list_in)
    # For even bag sizes, make sure len(list) is divisible by n. if not, trim
    if len(list_in) % n > 0:
      list_in = list_in[:-(len(list_in) % n)]
    if label == "MSI":
      return [[pat_id, list_in[i:i+n], 1] for i in range(0, len(list_in), n)]
    else: 
      return [[pat_id, list_in[i:i+n], 0] for i in range(0, len(list_in), n)]
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    pat_id, tile_paths, label = self.data[idx]
    label = torch.tensor(label)
    tile_list = []

    for tile in tile_paths:
      img = cv2.imread(tile)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # if self.transform:
      #   img = self.transform(image = img)['image']

      img = torch.from_numpy(img).float()
      # put channel dim first for pytorch
      img = img.permute(2, 0, 1)
      tile_list.append(img)
      # perform any transformations...

    return pat_id, tile_paths, tile_list, label
    # return tile_path, label


class PatientFoldBagDataset(Dataset):
  """
  BagDataset for Patient level indexed bags (folds in CV).
  Dir structure: data_path/subset/label/pat_id/*.png 
  """
  def __init__(self, data_path, subset="train", num_tiles=10, transform=None, pat_ids=[]):
    self.transform = transform
    dir_path = os.path.join(data_path, subset)
    class_list = os.listdir(dir_path)
    self.data = []   
    for label in class_list:
      print(label)
      full_path = os.path.join(dir_path, label)
      # get list of patients, filter for patients in this fold...
      class_pats = os.listdir(full_path)
      print(len(class_pats))
      class_pats = [x for x in class_pats if x in pat_ids]
      print(len(class_pats))
      for pat in class_pats:
        tile_path = glob.glob(os.path.join(full_path, pat) + "/*.png")
        if len(tile_path) < num_tiles:
          print(f"Not enough tiles for patient: {pat}")
          continue
        self.data = self.data + self.partition(tile_path, num_tiles, label)

  def partition(self, list_in, n, label):
    random.shuffle(list_in)
    # For even bag sizes, make sure len(list) is divisible by n. if not, trim
    if len(list_in) % n > 0:
      list_in = list_in[:-(len(list_in) % n)]
    if label == "MSI":
      return [[list_in[i:i+n], 1] for i in range(0, len(list_in), n)]
    else: 
      return [[list_in[i:i+n], 0] for i in range(0, len(list_in), n)]
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    tile_path, label = self.data[idx]
    label = torch.tensor(label)
    tile_list = []

    for tile in tile_path:
      img = cv2.imread(tile)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      if self.transform is not None:
        img = self.transform(image=img)['image']

      img = torch.from_numpy(img).float()
      # put channel dim first for pytorch
      img = img.permute(2, 0, 1)

      tile_list.append(img)

    return tile_list, label
    # return tile_path, label

class IndexPatientBagDataset(Dataset):
  """
  BagDataset for the patient level bags data given tile images names
  Dir structure: data_path/label/pat_id/*.png 
  """
  def __init__(self, data_path, label="MSI", pat_id=None, num_tiles=10, transform=None, tile_names=None):
    self.transform = transform
    dir_path = os.path.join(data_path, label, pat_id)
    tile_names = [os.path.join(dir_path, name) for name in tile_names]
    # print(tile_names)
    self.data = []   
    self.data = self.data + self.partition(tile_names, num_tiles, label, pat_id)

  def partition(self, list_in, n, label, pat_id):
    random.shuffle(list_in)
    # For even bag sizes, make sure len(list) is divisible by n. if not, trim
    if len(list_in) % n > 0:
      list_in = list_in[:-(len(list_in) % n)]
    if label == "MSI":
      return [[pat_id, list_in[i:i+n], 1] for i in range(0, len(list_in), n)]
    else: 
      return [[pat_id, list_in[i:i+n], 0] for i in range(0, len(list_in), n)]
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    pat_id, tile_paths, label = self.data[idx]
    label = torch.tensor(label)
    tile_list = []

    for tile in tile_paths:
      img = cv2.imread(tile)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


      img = torch.from_numpy(img).float()
      # put channel dim first for pytorch
      img = img.permute(2, 0, 1)
      tile_list.append(img)
      

    return pat_id, tile_paths, tile_list, label


def collate_bag_batches(batch):
  """
  Collate function to use for Dataloader for BagDataset:
  Returns data batch as a list length #bags/batch of NxCxHxW tensors, N = #tiles/bag
  """
  data = [item[0] for item in batch]
  target = [item[1] for item in batch]
  target = torch.FloatTensor(target)
  for i, item in enumerate(data):
    data[i] = torch.stack(item)

  return data, target

def collate_patient_bags(batch):
  """
  Collate function to use for Dataloader for PAIPBagDataset:
  Returns data batch as a list length #bags/batch of NxCxHxW tensors, N = #tiles/bag
  """
  pat_id = batch[0][0]
  tile_paths = [item[1] for item in batch]
  data = [item[2] for item in batch]
  target = [item[3] for item in batch]
  target = torch.FloatTensor(target)
  for i, item in enumerate(data):
    data[i] = torch.stack(item)

  return pat_id, tile_paths, data, target
