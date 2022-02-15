
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ATTN_net(nn.Module):
  def __init__(self):
    super(ATTN_net, self).__init__()
    # this is the "features" sequential portion of VGG19 pretrained on image net
    base_model = torch.load("/home/jleiby/MSI_pred/pretrained_model/vgg19_features.pth")

    # base model up to block 3, for input into attention subnet and rest of
    # feature extraction; output should be 256 x 56 x 56
    self.in_to_intermed = base_model[:18]
    # the rest of the feature extraction network
    self.feat_net_2 = base_model[18:]
    self.feat_pool = nn.AdaptiveAvgPool2d(7)
    self.feat_map = nn.Linear(512*7*7, 1024)

    # attention subnet;
    # change the pooling size and dropout rate as needed
    self.attn_sub = nn.Sequential(
        nn.AdaptiveAvgPool2d(7),
        nn.Flatten(),
        nn.Linear(256*7*7, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1, bias=False)
    )

    # final subnet; input will be aggregated feat map (1024)
    self.out_net = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        # output for binary prediction
        nn.Sigmoid()
    )

  def get_feat_map(self, x):
    '''
    Returns the feature map and the intermediate conv for attn subnet
    '''
    intermed = self.in_to_intermed(x)
    x = self.feat_net_2(intermed)
    x = self.feat_pool(x) 
    # print(x.shape)
    x = x.reshape(x.size(0), -1)
    # print(x.shape)
    x = self.feat_map(x)

    return x, intermed
  
  def forward(self, batch, device):
    # create empty tensors to store attns and feat maps 
    feat_maps = torch.zeros(size = (len(batch), len(batch[0]), self.feat_map.out_features)).to(device) # bags in batch X tiles in bag X feat_map size
    attns = torch.zeros(size = (len(batch), len(batch[0]), 1)).to(device) # bags in batch X tiles in bag X 1
    
    # iterate over all bags in the batch; get attention and heat map for each tile in each bag
    for i, bag in enumerate(batch):
      # bag = NxCxWxH tensor, N = #tiles/bag
      feat_maps[i,:,:], intermed = self.get_feat_map(bag) 
      attns[i,:,:] = self.attn_sub(intermed)
      
        # softmax...
    # attns = F.softmax(attns, dim = 1)
    w_feat_map = torch.mul(attns, feat_maps) # bags in a batch x tiles in bag x 1024
    w_feat_map = w_feat_map.sum(dim=1) / torch.sum(attns, dim=1) # bags in a batch x 1024
    
    # print(w_feat_map.shape)
    out = self.out_net(w_feat_map)

    return attns, out

class ATTN_net_con(nn.Module):
  def __init__(self):
    super(ATTN_net_con, self).__init__()
    # this is the "features" sequential portion of VGG19 pretrained on image net
    base_model = torch.load("/home/jleiby/MSI_pred/pretrained_model/vgg19_features.pth")

    # base model up to block 3, for input into attention subnet and rest of
    # feature extraction; output should be 256 x 56 x 56
    self.in_to_intermed = base_model[:18]
    # the rest of the feature extraction network
    self.feat_net_2 = base_model[18:]
    self.feat_pool = nn.AdaptiveAvgPool2d(7)
    self.feat_map = nn.Linear(512*7*7, 1024)

    # attention subnet;
    # change the pooling size and dropout rate as needed
    self.attn_sub = nn.Sequential(
        nn.AdaptiveAvgPool2d(7),
        nn.Flatten(),
        nn.Linear(256*7*7, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1, bias=False)
    )

    # final subnet; input will be aggregated feat map (1024)
    self.out_net = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        # output for binary prediction
        nn.Sigmoid()
    )

  def get_feat_map(self, x):
    '''
    Returns the feature map and the intermediate conv for attn subnet
    '''
    intermed = self.in_to_intermed(x)
    x = self.feat_net_2(intermed)
    x = self.feat_pool(x) 
    # print(x.shape)
    x = x.reshape(x.size(0), -1)
    # print(x.shape)
    x = self.feat_map(x)

    return x, intermed
  
  def forward(self, batch, device):
    # create empty tensors to store attns and feat maps 
    feat_maps = torch.zeros(size = (len(batch), len(batch[0]), self.feat_map.out_features)).to(device) # bags in batch X tiles in bag X feat_map size
    attns = torch.zeros(size = (len(batch), len(batch[0]), 1)).to(device) # bags in batch X tiles in bag X 1
    
    # iterate over all bags in the batch; get attention and heat map for each tile in each bag
    for i, bag in enumerate(batch):
      # bag = NxCxWxH tensor, N = #tiles/bag
      feat_maps[i,:,:], intermed = self.get_feat_map(bag) 
      attns[i,:,:] = self.attn_sub(intermed)
      
        # softmax...
    # attns = F.softmax(attns, dim = 1)
    w_feat_map = torch.mul(attns, feat_maps) # bags in a batch x tiles in bag x 1024
    w_feat_map = w_feat_map.sum(dim=1) / torch.sum(attns, dim=1) # bags in a batch x 1024
    
    # print(w_feat_map.shape)
    out = self.out_net(w_feat_map)

    return attns, out, feat_maps, w_feat_map

class ATTN_net_block2(nn.Module):
  def __init__(self):
    super(ATTN_net_block2, self).__init__()
    # this is the "features" sequential portion of VGG19 pretrained on image net
    base_model = torch.load("/home/jleiby/MSI_pred/pretrained_model/vgg19_features.pth")

    # base model up to block 3, for input into attention subnet and rest of
    # feature extraction; output should be 256 x 56 x 56
    self.in_to_intermed = base_model[:9]
    # the rest of the feature extraction network
    self.feat_net_2 = base_model[9:]
    self.feat_pool = nn.AdaptiveAvgPool2d(7)
    self.feat_map = nn.Linear(512*7*7, 1024)

    # attention subnet;
    # change the pooling size and dropout rate as needed
    self.attn_sub = nn.Sequential(
        nn.AdaptiveAvgPool2d(14),
        nn.Flatten(),
        nn.Linear(128*14*14, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1, bias=False)
    )

    # final subnet; input will be aggregated feat map (1024)
    self.out_net = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        # output for binary prediction
        nn.Sigmoid()
    )

  def get_feat_map(self, x):
    '''
    Returns the feature map and the intermediate conv for attn subnet
    '''
    intermed = self.in_to_intermed(x)
    x = self.feat_net_2(intermed)
    x = self.feat_pool(x) 
    # print(x.shape)
    x = x.reshape(x.size(0), -1)
    # print(x.shape)
    x = self.feat_map(x)

    return x, intermed
  
  def forward(self, batch, device):
    # create empty tensors to store attns and feat maps 
    feat_maps = torch.zeros(size = (len(batch), len(batch[0]), self.feat_map.out_features)).to(device) # bags in batch X tiles in bag X feat_map size
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

    return attns, out

class ATTN_net_res34(nn.Module):
  def __init__(self):
    super(ATTN_net_res34, self).__init__()
    # Pretrained resnet (feat_net_2 goes to -2 to drop the final linear layer and avgpool from the model)
    base_model = torch.load("/home/jleiby/MSI_pred/pretrained_model/resnet34.pth")

    # base model up to block 3, for input into attention subnet and rest of
    # feature extraction; output should be 256 x 56 x 56
    self.in_to_intermed = nn.Sequential(*list(base_model.children())[:-3])
    # the rest of the feature extraction network
    self.feat_net_2 = nn.Sequential(*list(base_model.children())[-3:-2])
    self.feat_pool = nn.AdaptiveAvgPool2d(1)
    # self.feat_map = nn.Linear(512, 1024)

    # attention subnet;
    # change the pooling size and dropout rate as needed
    self.attn_sub = nn.Sequential(
        nn.AdaptiveAvgPool2d(7),
        nn.Flatten(),
        nn.Linear(256*7*7, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1, bias=False)
    )

    # final subnet; input will be aggregated feat map (512)
    self.out_net = nn.Sequential(
        # nn.Linear(512, 512),
        # nn.ReLU(),
        # nn.Dropout(0.5),
        # nn.Linear(512, 256),
        # nn.ReLU(),
        nn.Linear(512, 1),
        # output for binary prediction
        nn.Sigmoid()
    )

  def get_feat_map(self, x):
    '''
    Returns the feature map and the intermediate conv for attn subnet
    '''
    intermed = self.in_to_intermed(x)
    x = self.feat_net_2(intermed)
    x = self.feat_pool(x) 
    # print(x.shape)
    x = x.reshape(x.size(0), -1)
    # print(x.shape)
    # x = self.feat_map(x)

    return x, intermed
  
  def forward(self, batch, device):
    # create empty tensors to store attns and feat maps 
    feat_maps = torch.zeros(size = (len(batch), len(batch[0]), 512)).to(device) # bags in batch X tiles in bag X feat_map size
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

    return attns, out

class ATTN_net_res34_2(nn.Module):
  def __init__(self):
    super(ATTN_net_res34_2, self).__init__()
    # Pretrained resnet (feat_net_2 goes to -2 to drop the final linear layer and avgpool from the model)
    base_model = torch.load("/home/jleiby/MSI_pred/pretrained_model/resnet34.pth")

    # base model up to block 3, for input into attention subnet and rest of
    # feature extraction; output should be 256 x 56 x 56
    self.in_to_intermed = nn.Sequential(*list(base_model.children())[:-4])
    # the rest of the feature extraction network
    self.feat_net_2 = nn.Sequential(*list(base_model.children())[-4:-2])
    self.feat_pool = nn.AdaptiveAvgPool2d(1)
    # self.feat_map = nn.Linear(512, 1024)

    # attention subnet;
    # change the pooling size and dropout rate as needed
    self.attn_sub = nn.Sequential(
        nn.AdaptiveAvgPool2d(7),
        nn.Flatten(),
        nn.Linear(128*7*7, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1, bias=False)
    )

    # final subnet; input will be aggregated feat map (512)
    self.out_net = nn.Sequential(
        # nn.Linear(512, 512),
        # nn.ReLU(),
        # nn.Dropout(0.5),
        # nn.Linear(512, 256),
        # nn.ReLU(),
        nn.Linear(512, 1),
        # output for binary prediction
        nn.Sigmoid()
    )

  def get_feat_map(self, x):
    '''
    Returns the feature map and the intermediate conv for attn subnet
    '''
    intermed = self.in_to_intermed(x)
    x = self.feat_net_2(intermed)
    x = self.feat_pool(x) 
    # print(x.shape)
    x = x.reshape(x.size(0), -1)
    # print(x.shape)
    # x = self.feat_map(x)

    return x, intermed
  
  def forward(self, batch, device):
    # create empty tensors to store attns and feat maps 
    feat_maps = torch.zeros(size = (len(batch), len(batch[0]), 512)).to(device) # bags in batch X tiles in bag X feat_map size
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

    return attns, out

class ATTN_net_ShuffleNetV2(nn.Module):
  def __init__(self):
    super(ATTN_net_ShuffleNetV2, self).__init__()
    # this is the "features" sequential portion of VGG19 pretrained on image net
    base_model = torch.load("/home/jleiby/MSI_pred/pretrained_model/ShuffleNetV2_x1.pth")

    # base model up to block 3, for input into attention subnet and rest of
    # feature extraction; output should be 256 x 56 x 56
    self.in_to_intermed = nn.Sequential(*list(base_model.children())[:3])
    # the rest of the feature extraction network
    self.feat_net_2 = nn.Sequential(*list(base_model.children())[3:-1])
    # self.feat_pool = nn.AdaptiveAvgPool2d(7)
    # self.feat_map = nn.Linear(512*7*7, 1024)

    # attention subnet;
    # change the pooling size and dropout rate as needed
    self.attn_sub = nn.Sequential(
        nn.AdaptiveAvgPool2d(14),
        nn.Flatten(),
        nn.Linear(116*14*14, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1, bias=False)
    )

    # final subnet; input will be aggregated feat map (1024)
    self.out_net = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        # output for binary prediction
        nn.Sigmoid()
    )

  def get_feat_map(self, x):
    '''
    Returns the feature map and the intermediate conv for attn subnet
    '''
    intermed = self.in_to_intermed(x)
    x = self.feat_net_2(intermed)
    x = x.mean([2, 3])  # globalpool, 1 x 1024 after mean

    return x, intermed
  
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

    return attns, out

class Baseline_model(nn.Module):
  # This model has the same general architecture as ATTN_net, however no attenion subnetwork or weighted aggregation
  def __init__(self):
    super(Baseline_model, self).__init__()
    # this is the "features" sequential portion of VGG19 pretrained on image net
    self.convs = torch.load("/home/jleiby/MSI_pred/pretrained_model/vgg19_features.pth")
    # self.convs = models.vgg19(pretrained=True).features
    self.feat_pool = nn.AdaptiveAvgPool2d(7)
    self.feat_map = nn.Linear(512*7*7, 1024)
    self.out_net = nn.Sequential(
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(1024, 256),
      nn.ReLU(),
      nn.Linear(256, 1),
      # output for binary prediction
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.convs(x)
    x = self.feat_pool(x)
    x = x.reshape(x.size(0), -1)
    # print(x.shape)
    x = self.feat_map(x)
    x = self.out_net(x)
    return x


class ATTN_net_res34_late(nn.Module):
  def __init__(self):
    super(ATTN_net_res34_late, self).__init__()
    # Pretrained resnet (feature_extraction goes to -2 to drop the final linear layer and avgpool from the model)
    base_model = torch.load("/home/jleiby/MSI_pred/pretrained_model/resnet34.pth")

    # base model to final feat map layer, followed by pooling (512)
    self.feature_extraction = nn.Sequential(*list(base_model.children())[:-2])
    self.feat_pool = nn.AdaptiveAvgPool2d(1)
    # self.feat_map = nn.Linear(512, 1024)

    # attention subnet;
    # change the pooling size and dropout rate as needed
    self.attn_sub = nn.Sequential(
        nn.Linear(512, 128),
        nn.Tanh(),
        nn.Linear(128, 1, bias=False)
    )

    # final subnet; input will be aggregated feat map (512) 
    self.out_net = nn.Sequential(
        # nn.Linear(512, 512),
        # nn.ReLU(),
        # nn.Dropout(0.5),
        # nn.Linear(512, 256),
        # nn.ReLU(),
        nn.Linear(512, 1),
        # output for binary prediction
        nn.Sigmoid()
    )

  def get_feat_map(self, x):
    '''
    Returns the feature map and the intermediate conv for attn subnet
    '''
    x = self.feature_extraction(x)
    x = self.feat_pool(x) 
    # print(x.shape)
    x = x.reshape(x.size(0), -1)
    # print(x.shape)
    # x = self.feat_map(x)

    return x, x
  
  def forward(self, batch, device):
    # create empty tensors to store attns and feat maps 
    feat_maps = torch.zeros(size = (len(batch), len(batch[0]), 512)).to(device) # bags in batch X tiles in bag X feat_map size
    attns = torch.zeros(size = (len(batch), len(batch[0]), 1)).to(device) # bags in batch X tiles in bag X 1
    
    # iterate over all bags in the batch; get attention and heat map for each tile in each bag
    for i, bag in enumerate(batch):
      # bag = NxCxWxH tensor, N = #tiles/bag
      feat_maps[i,:,:], feats = self.get_feat_map(bag) 
      attns[i,:,:] = self.attn_sub(feats)
      

    w_feat_map = torch.mul(attns, feat_maps) # bags in a batch x tiles in bag x 512
    w_feat_map = w_feat_map.sum(dim=1) / torch.sum(attns, dim=1) # bags in a batch x 512
    
    # print(w_feat_map.shape)
    out = self.out_net(w_feat_map)

    return attns, out


class ATTN_net_res34_con(nn.Module):
  def __init__(self):
    super(ATTN_net_res34_con, self).__init__()
    # Pretrained resnet (feat_net_2 goes to -2 to drop the final linear layer and avgpool from the model)
    base_model = torch.load("/home/jleiby/MSI_pred/pretrained_model/resnet34.pth")

    # base model up to block 3, for input into attention subnet and rest of
    # feature extraction; output should be 256 x 56 x 56
    self.feature_extraction = nn.Sequential(*list(base_model.children())[:-2])
    self.feat_pool = nn.AdaptiveAvgPool2d(1)
    # self.feat_map = nn.Linear(512, 1024)

    # attention subnet;
    # change the pooling size and dropout rate as needed
    self.attn_sub = nn.Sequential(
        nn.Linear(512, 128),
        nn.Tanh(),
        nn.Linear(128, 1, bias=False)
    )

    # final subnet; input will be aggregated feat map (512) 
    self.out_net = nn.Sequential(
        # nn.Linear(512, 512),
        # nn.ReLU(),
        # nn.Dropout(0.5),
        # nn.Linear(512, 256),
        # nn.ReLU(),
        nn.Linear(512, 128),
        nn.Tanh(),
        nn.Linear(128, 1),
        # output for binary prediction
        nn.Sigmoid()
    )

  def get_feat_map(self, x):
    '''
    Returns the feature map and the intermediate conv for attn subnet
    '''
    x = self.feature_extraction(x)
    x = self.feat_pool(x) 
    # print(x.shape)
    x = x.reshape(x.size(0), -1)
    # print(x.shape)
    # x = self.feat_map(x)

    return x, x
  
  def forward(self, batch, device):
    # create empty tensors to store attns and feat maps 
    feat_maps = torch.zeros(size = (len(batch), len(batch[0]), 512)).to(device) # bags in batch X tiles in bag X feat_map size
    attns = torch.zeros(size = (len(batch), len(batch[0]), 1)).to(device) # bags in batch X tiles in bag X 1
    
    # iterate over all bags in the batch; get attention and heat map for each tile in each bag
    for i, bag in enumerate(batch):
      # bag = NxCxWxH tensor, N = #tiles/bag
      feat_maps[i,:,:], feats = self.get_feat_map(bag) 
      attns[i,:,:] = self.attn_sub(feats)
      

    w_feat_map = torch.mul(attns, feat_maps) # bags in a batch x tiles in bag x 512
    w_feat_map = w_feat_map.sum(dim=1) / torch.sum(attns, dim=1) # bags in a batch x 512
    
    # print(w_feat_map.shape)
    out = self.out_net(w_feat_map)

    return attns, out, feat_maps, w_feat_map



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

class ATTN_net_ciga_late(nn.Module):
  def __init__(self, device):
    super(ATTN_net_ciga_late, self).__init__()
    # Pretrained resnet (feature_extraction goes to -2 to drop the final linear layer and avgpool from the model)
    # base_model = torch.load("/home/jleiby/MSI_pred/pretrained_model/resnet34.pth")
    state = torch.load("/home/jleiby/MSI_pred/pretrained_model/tenpercent_resnet18.ckpt", map_location=device)

    self.model = models.__dict__['resnet18'](pretrained=False)

    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    self.model = self.load_model_weights(self.model, state_dict)
    self.model.fc = torch.nn.Sequential()

    # attention subnet;
    # change the pooling size and dropout rate as needed
    self.attn_sub = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 1, bias=False)
    )

    # final subnet; input will be aggregated feat map (512) 
    self.out_net = nn.Sequential(
        # nn.Linear(512, 512),
        # nn.ReLU(),
        # nn.Dropout(0.5),
        # nn.Linear(512, 256),
        # nn.ReLU(),
        nn.Linear(512, 1),
        # output for binary prediction
        nn.Sigmoid()
    )

  def load_model_weights(self, model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model


  def get_feat_map(self, x):
    '''
    Returns the feature map and the intermediate conv for attn subnet
    '''
    x = self.model(x)
    # print(x.shape)
    # x = self.feat_map(x)

    return x, x
  
  def forward(self, batch, device):
    # create empty tensors to store attns and feat maps 
    feat_maps = torch.zeros(size = (len(batch), len(batch[0]), 512)).to(device) # bags in batch X tiles in bag X feat_map size
    attns = torch.zeros(size = (len(batch), len(batch[0]), 1)).to(device) # bags in batch X tiles in bag X 1
    
    # iterate over all bags in the batch; get attention and heat map for each tile in each bag
    for i, bag in enumerate(batch):
      # bag = NxCxWxH tensor, N = #tiles/bag
      feat_maps[i,:,:], feats = self.get_feat_map(bag) 
      attns[i,:,:] = self.attn_sub(feats)
      

    w_feat_map = torch.mul(attns, feat_maps) # bags in a batch x tiles in bag x 512
    w_feat_map = w_feat_map.sum(dim=1) / torch.sum(attns, dim=1) # bags in a batch x 512
    
    # print(w_feat_map.shape)
    out = self.out_net(w_feat_map)

    return attns, out

