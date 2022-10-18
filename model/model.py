
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Classifier(nn.Module):
  """
  Classifier network: Feature map -> prediction
  """
  def __init__(self):
    super(Classifier, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.net(x) 


class ATTN_module(nn.Module):
  """
  Attention network designed to take in intermediate VGG convolution
  """
  def __init__(self, pool_dim = 7, inter_im = 256):
    super(ATTN_module, self).__init__()
    self.net = nn.Sequential(
        nn.AdaptiveAvgPool2d(pool_dim),
        nn.Flatten(),
        nn.Linear(256*pool_dim*pool_dim, inter_dim),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(inter_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 1, bias=False)
    )

  def forward(self, x):
    return self.net(x)

class Gated_ATTN_module(nn.Module):
  """
  Gated attention network designed to take in intermediate VGG convolution
  """
  def __init__(self, pool_dim = 7, inter_dim = 1024):
    super(Gated_ATTN_module, self).__init__()
    self.attn_base = nn.Sequential(
        nn.AdaptiveAvgPool2d(pool_dim),
        nn.Flatten(),
        nn.Linear(256*pool_dim*pool_dim, inter_dim),
        nn.ReLU(),
        nn.Dropout(0.5),
    )

    self.attn_t = nn.Sequential(
        nn.Linear(inter_dim, 256),
        nn.Tanh()
    )

    self.attn_s = nn.Sequential(
        nn.Linear(inter_dim, 256),
        nn.Sigmoid()
    )

    self.attn_out = nn.Linear(256, 1, bias=False)

  def forward(self, x):
    x = self.attn_base(x)
    b1 = self.attn_t(x)
    b2 = self.attn_s(x)
    x = torch.mul(b1, b2)
    x = self.attn_out(x)
    return x

class Feature_extraction(nn.Module):
  """
  Extracts feature map and optonally intermediate convolution from pretrained VGG19
  """
  def __init__(self, ret_int = True):
    super(Feature_extraction, self).__init__()
    self.ret_int = ret_int
    # this is the "features" sequential portion of VGG19 pretrained on image net
    base_model = torch.load("/home/jleiby/MSI_pred/pretrained_model/vgg19_features.pth")

    # base model up to block 3, for input into attention subnet and rest of
    # feature extraction; output == 256 x 56 x 56
    self.in_to_intermed = base_model[:18]

    # the rest of the feature extraction network
    self.feat_net_2 = base_model[18:]
    self.feat_pool = nn.AdaptiveAvgPool2d(7)
    self.feat_map = nn.Linear(512*7*7, 1024)

  def get_feat_map(self, x):
    '''
    Returns the feature map and the intermediate conv
    '''
    intermed = self.in_to_intermed(x)
    x = self.feat_net_2(intermed)
    x = self.feat_pool(x) 
    # print(x.shape)
    x = x.reshape(x.size(0), -1)
    # print(x.shape)
    x = self.feat_map(x)

    return x, intermed

  def forward(self, x):
    x, intermed = self.get_feat_map(x)

    if self.ret_int:
      return x, intermed
    else:
      return x

class MIL_net(nn.Module):
  """
  VGG19 feature extraction network with attention sub-network
  """
  def __init__(self, gate = True, contrastive = True):
    super(MIL_net, self).__init__()
    self.get_feat_map = Feature_extraction()

    if gate:
      self.attn_sub = Gated_ATTN_module(pool_dim=7)
    else:
      self.attn_sub = ATTN_module(pool_dim=7)

    self.out_net = Classifier()

    self.contrastive = contrastive

  def forward(self, batch, device):
    # create empty tensors to store attns and feat maps 
    feat_maps = torch.zeros(size = (len(batch), len(batch[0]), 1024)).to(device) # bags in batch X tiles in bag X feat_map size
    attns = torch.zeros(size = (len(batch), len(batch[0]), 1)).to(device) # bags in batch X tiles in bag X 1
    
    # iterate over all bags in the batch; get attention and heat map for each tile in each bag
    for i, bag in enumerate(batch):
      # bag = NxCxWxH tensor, N = #tiles/bag
      feat_maps[i,:,:], intermed = self.get_feat_map(bag) 
      attns[i,:,:] = self.attn_sub(intermed)
      
    w_feat_map = torch.mul(attns, feat_maps) # bags in a batch x tiles in bag x 1024
    w_feat_map = w_feat_map.sum(dim=1) / torch.sum(attns, dim=1) # bags in a batch x 1024
    
    # print(w_feat_map.shape)
    out = self.out_net(w_feat_map)

    if self.contrastive:
      return attns, out, feat_maps, w_feat_map
    else:
      return attns, out

class Baseline_model(nn.Module):
  """
  Baseline model for supervised learning. Pretrained VGG19 feature extraction followed by classifer network
  """
  def __init__(self):
    super(Baseline_model, self).__init__()
    self.features = Feature_extraction(ret_int=False)
    
    self.out_net = Classifier()

  def forward(self, x):
    x = self.features(x)
    x = self.out_net(x)
    return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        # self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        # self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(device))
            
    def forward(self, instance_emb, bag_emb, device):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        # reshape instance feature maps from batch X tiles X size --> batch*tiles X size
        emb_i = torch.reshape(instance_emb, (instance_emb.shape[0]*instance_emb.shape[1], instance_emb.shape[2]))
        # expand weight feature maps from batch X size --> batch(repeated)*tiles X size
        emb_j = torch.repeat_interleave(bag_emb, instance_emb.shape[1], dim=0)

        # get batch seize (cant hardcode, may be different for final samples in epoch...)
        bs = emb_i.shape[0]

        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, bs)
        sim_ji = torch.diag(similarity_matrix, -bs)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        negatives_mask = (~torch.eye(bs * 2, bs * 2, dtype=bool)).float().to(device)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * bs)
        return loss


