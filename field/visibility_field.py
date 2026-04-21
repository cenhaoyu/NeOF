import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter

class Direct(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.d_a=torch.ones(size=[1],dtype=torch.float,requires_grad=False)
    self.d_b=torch.ones(size=[1],dtype=torch.float,requires_grad=False)
  def forward(self,x_world,voxel_point,voxel_normal,score):
    #x_world (m,1,3)
    #voxel_point(1,n,3)
    #voxel_normal(n,3)
    #score(n,1)
    distance=torch.linalg.norm(x_world-voxel_point[:,:,:3],dim=-1)
    with torch.no_grad():
        x_normal=torch.mean(voxel_normal[torch.sort(distance,1)[1][:,:8]],1).unsqueeze(1) #(m,1,3)
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        output=cos(x_normal.repeat(1,len(voxel_normal),1),voxel_normal.unsqueeze(0).repeat(x_normal.shape[1],1,1))
        score_num=torch.sum(output>0.75,dim=1)
        nonzerojudge=(score_num!=0).nonzero().squeeze()
    score_sum=torch.sum(score.repeat(x_world.shape[0],1)*(output>0.75)/(torch.exp(distance)),1)
    x_world_field=score_sum[nonzerojudge]/score_num[nonzerojudge]
    output.cpu().numpy(),score_num.cpu().numpy(),x_normal.cpu().numpy()
    del output,score_num
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return x_world_field,nonzerojudge
class Addparam(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.d_a=Parameter(torch.ones(size=[1],dtype=torch.float,requires_grad=True))
    self.d_b=Parameter(torch.ones(size=[1],dtype=torch.float,requires_grad=True))
  def forward(self,x_world,voxel_point,voxel_normal,score):
    #x_world (m,1,3)
    #voxel_point(1,n,3)
    #voxel_normal(n,3)
    #score(n,1)
    distance=torch.linalg.norm(x_world-voxel_point[:,:,:3],dim=-1)
    with torch.no_grad():
        x_normal=torch.mean(voxel_normal[torch.sort(distance,1)[1][:,:8]],1).unsqueeze(1) #(m,1,3)
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        output=cos(x_normal.repeat(1,len(voxel_normal),1),voxel_normal.unsqueeze(0).repeat(x_normal.shape[1],1,1))
        score_num=torch.sum(output>0.8,dim=1)
        nonzerojudge=(score_num!=0).nonzero().squeeze()
    score_sum=torch.sum(score.repeat(x_world.shape[0],1)*(output>0.8)/(self.d_a*torch.exp(self.d_b*distance)),1)
    x_world_field=score_sum[nonzerojudge]/score_num[nonzerojudge]
    output.cpu().numpy(),score_num.cpu().numpy(),x_normal.cpu().numpy()
    del output,score_num
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return x_world_field,nonzerojudge
class AddAttention(torch.nn.Module):
  def __init__(self, input_channel,output_channel,num_heads,context_voxel_num=64,normal_voxel_num=8):
    super(AddAttention, self).__init__()
    self.num_heads = num_heads
    self.output_channel = output_channel
    self.context_voxel_num = context_voxel_num
    self.normal_voxel_num = normal_voxel_num

    assert output_channel % num_heads == 0

    self.depth = output_channel // num_heads

    self.feature_mlp = nn.Sequential(
        nn.Linear(input_channel, output_channel),
        nn.ReLU(inplace=True),
    )
    self.Wq = nn.Linear(output_channel, output_channel)
    self.Wk = nn.Linear(output_channel, output_channel)

  def calculate_x_voxelselect(self,x_world,voxel_point,voxel_normal,v,exclude_closest_context=False):
    # x_world(m,1,3) voxel_point(n,3) voxel_normal(n,3) v(n,k)
    with torch.no_grad():
      distance=torch.linalg.norm(x_world-voxel_point.unsqueeze(0),dim=-1)
      sorted_index = torch.argsort(distance, dim=1)
      start_index = 1 if exclude_closest_context and voxel_point.shape[0] > 1 else 0
      available_context = max(voxel_point.shape[0] - start_index, 1)
      context_num = min(self.context_voxel_num, available_context)
      normal_num = min(self.normal_voxel_num, voxel_normal.shape[0])
      context_index = sorted_index[:,start_index:start_index + context_num]
      if context_index.shape[1] == 0:
        context_index = sorted_index[:,:1]
      normal_index = sorted_index[:,:normal_num]
      voxel_point_select = voxel_point[context_index]
      voxel_normal_select = voxel_normal[context_index]
      v_select = v[context_index]
      x_normal=torch.mean(voxel_normal[normal_index],1,keepdim=True)
      normal_relative=x_normal-voxel_normal_select
    position_relative=x_world-voxel_point_select
    x=torch.cat((position_relative,normal_relative),-1)
    return x,v_select
    
  def forward(self,x_world,voxel_point,voxel_normal,v,density=None,mask=None,exclude_closest_context=False):
    #x_world (m,1,3)
    #voxel_point (n,3)
    #voxel_normal (n,3)
    #v (n,k)
    x,v=self.calculate_x_voxelselect(
        x_world,
        voxel_point,
        voxel_normal,
        v,
        exclude_closest_context=exclude_closest_context,
    )
    x=self.feature_mlp(x)
    batch_size = x.size(0)
    context_size = x.size(1)

    query_feature = torch.mean(x, dim=1, keepdim=True)
    Q = self.Wq(query_feature).view(batch_size, 1, self.num_heads, self.depth).transpose(1,2)
    K = self.Wk(x).view(batch_size, context_size, self.num_heads, self.depth).transpose(1,2)
    scores = torch.matmul(Q,K.transpose(-1,-2)).squeeze(-2) / np.sqrt(self.depth)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = torch.softmax(scores,dim=-1)
    value = v.unsqueeze(1).expand(-1,self.num_heads,-1,-1)
    out = torch.matmul(attention.unsqueeze(-2),value).squeeze(-2)
    out = torch.mean(out,dim=1)
    return out
