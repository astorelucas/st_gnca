import torch
from torch import nn
from datetime import datetime
import numpy as np
import pandas as pd
from tensordict import TensorDict

def datetime_to_str(dt):
  return dt.strftime("%Y%m%d%H%M%S")

def str_to_datetime(dt):
  return datetime.strptime(dt, '%Y%m%d%H%M%S')

def to_pandas_datetime(values):
  return pd.to_datetime(values, format='%m/%d/%Y %H:%M')

def from_np_to_datetime(dt):
  dt = to_pandas_datetime(dt)
  return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)

def from_pd_to_datetime(dt):
  return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
  
def from_datetime_to_pd(date : datetime):

  return to_pandas_datetime(np.datetime64(date.astimezone(None)))


class TemporalEmbedding(nn.Module):
  def __init__(self, dates, **kwargs):
    super().__init__()
    self.device = kwargs.get('device','cpu')
    self.dtype = kwargs.get('dtype',torch.float32)
    self.pi2 = torch.tensor([2 * torch.pi], dtype=self.dtype, device=self.device)
    self.day_minutes = torch.tensor([(1440)], dtype=self.dtype, device=self.device )
    self.week_minutes = torch.tensor([(7 * 1440)], dtype=self.dtype, device=self.device)
    tmp_dict = {}
    self.length = 0
    for date in dates:
      tmp_dict[datetime_to_str(date)] = self.forward(date)
      self.length += 1 
    self.embeddings : TensorDict = TensorDict(tmp_dict) 

  def week_embedding(self, date):
    day_of_week = date.isocalendar()[2]  # Day of week (1: Monday, 7: Sunday)
    num = torch.tensor([day_of_week * 1440 + date.hour * 60 + date.minute], dtype=self.dtype, device=self.device)
    angle = num * self.pi2 / self.week_minutes
    return torch.sin(angle), torch.cos(angle)

  def month_embedding(self, date):
    month = date.month  # 1-12
    angle = torch.tensor([month], dtype=self.dtype, device=self.device) * self.pi2 / 12
    return torch.sin(angle), torch.cos(angle)

  def minute_embedding(self, date):
    minute_of_day = torch.tensor([(date.hour * 60 + date.minute)/ 1440], dtype=self.dtype, device=self.device)
    angle = minute_of_day * self.pi2
    return torch.sin(angle), torch.cos(angle)
  
  def forward(self, dt):
     date = from_pd_to_datetime(dt)
     moe_sin, moe_cos = self.month_embedding(date)
     we_sin, we_cos = self.week_embedding(date)
     me_sin, me_cos = self.minute_embedding(date)
     return torch.stack([moe_sin, moe_cos, we_sin, we_cos, me_sin, me_cos]).squeeze()
  
  def __getitem__(self, date):
    if isinstance(date, np.datetime64):
      date =  from_np_to_datetime(date)
      return self.embeddings[datetime_to_str(date)]
    elif isinstance(date, datetime):
      return self.embeddings[datetime_to_str(date)]
    elif isinstance(date, int):
      return self.embeddings[date]
    else:
      raise Exception("Unknown index type")
    
  
  def all(self):
    ret = torch.empty(self.length, 4, dtype=self.dtype, device=self.device)
    for it,emb in enumerate(self.embeddings.values(sort=True)):
      ret[it, :] = emb
    return ret
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.pi2 = self.pi2.to(*args, **kwargs)
    self.week_minutes = self.week_minutes.to(*args, **kwargs)
    return self


class SinusoidalTemporalEncoding(nn.Module):
    def __init__(self, dates, **kwargs):
        super().__init__()
        self.device = kwargs.get('device','cpu')
        self.dtype = kwargs.get('dtype',torch.float32)
        self.pi2 = torch.tensor([2 * torch.pi], dtype=self.dtype, device=self.device)
        self.day_minutes = torch.tensor([(1440)], dtype=self.dtype, device=self.device )
        self.week_minutes = torch.tensor([(7 * 1440)], dtype=self.dtype, device=self.device)
        tmp_dict = {}
        self.length = 0
        for date in dates:
            tmp_dict[datetime_to_str(date)] = self.forward(date)
            self.length += 1 
            # print(tmp_dict[datetime_to_str(date)].shape)
        self.embeddings : TensorDict = TensorDict(tmp_dict) 

    def forward(self, dt, d_model=4):
        '''
        forward method to compute the sinusoidal encoding based on position for a given datetime.

        Args:
            dt: A datetime object or a pandas Timestamp.
            d_model: Dimensionality of the output embedding (must be even).
        Returns:
            A tensor of shape (d_model,) containing the sinusoidal encoding.
        '''

        assert d_model % 2 == 0
        base_time = pd.Timestamp("2010-01-01 00:00:00")
        seconds = (dt - base_time).total_seconds()
        pos = torch.tensor([[seconds]])
        i = torch.arange(d_model // 2).unsqueeze(0)
        div_term = torch.pow(10000, (2 * i) / d_model)

        angle_rads = pos / div_term
        sin_part = torch.sin(angle_rads)
        cos_part = torch.cos(angle_rads)
        
        return torch.cat([sin_part, cos_part], dim=-1).view(-1)

    def __getitem__(self, date):
        if isinstance(date, np.datetime64):
            date =  from_np_to_datetime(date)
            return self.embeddings[datetime_to_str(date)]
        elif isinstance(date, datetime):
            return self.embeddings[datetime_to_str(date)]
        elif isinstance(date, int):
            return self.embeddings[date]
        else:
            raise Exception("Unknown index type")
    
    def all(self):
        ret = torch.empty(self.length, 4, dtype=self.dtype, device=self.device)
        for it,emb in enumerate(self.embeddings.values(sort=True)):
            ret[it, :] = emb
        return ret
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if isinstance(args[0], str):
            self.device = args[0]
        else:
            self.dtype = args[0]
        self.pi2 = self.pi2.to(*args, **kwargs)
        return self

