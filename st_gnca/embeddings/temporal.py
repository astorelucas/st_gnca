import torch
from torch import nn
from datetime import datetime
import numpy as np
import pandas as pd
from tensordict import TensorDict
from datetime import datetime
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str_to_datetime(dt):
  return datetime.strptime(dt, '%Y%m%d%H%M%S')

def to_pandas_datetime(values):
  return pd.to_datetime(values, format='%m/%d/%Y %H:%M')

def from_np_to_datetime(dt):
  dt = to_pandas_datetime(dt)
  return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)

def from_pd_to_datetime(dt):
  return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
  
def from_datetime_to_pd(date: datetime):
    if date.tzinfo is not None:
        date = date.replace(tzinfo=None)
    return to_pandas_datetime(date)

def datetime_to_str(dt):
    return pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")

def from_np_to_datetime(np_dt):
    return pd.Timestamp(np_dt).to_pydatetime()

def datetime_to_str(dt):
    """Convert datetime to string format"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def from_np_to_datetime(np_dt):
    """Convert numpy datetime64 to python datetime"""
    return pd.Timestamp(np_dt).to_pydatetime()

class SinusoidalTemporalEncoding_old(nn.Module):
    def __init__(self, dates, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', DEVICE)
        self.dtype = kwargs.get('dtype', torch.float32)
        self.pi2 = torch.tensor([2 * torch.pi], dtype=self.dtype, device=self.device)
        self.day_minutes = torch.tensor([(1440)], dtype=self.dtype, device=self.device)
        self.week_minutes = torch.tensor([(7 * 1440)], dtype=self.dtype, device=self.device)
        tmp_dict = {}
        self.length = 0
        for date in dates:
            tmp_dict[datetime_to_str(date)] = self.forward(date)
            self.length += 1
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
    
class SinusoidalTemporalEncoding(nn.Module):
    def __init__(self, dates, emb_dim, device=None, dtype=torch.float32):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.emb_dim = emb_dim
        
        # Store embeddings for all dates
        tmp_dict = {}
        for date in dates:
            tmp_dict[datetime_to_str(date)] = self._compute_encoding(date, emb_dim)
        
        self.embeddings = tmp_dict
        self.length = len(dates)

    def _compute_encoding(self, dt, emb_dim):
        """
        Compute sinusoidal encoding for a specific datetime
        """
        base_time = pd.Timestamp("2010-01-01 00:00:00")
        seconds = (dt - base_time).total_seconds()
        
        pos = torch.tensor([seconds], dtype=self.dtype, device=self.device)
        
        i = torch.arange(emb_dim // 2, dtype=self.dtype, device=self.device)
        div_term = torch.exp(i * (-math.log(10000.0) / (emb_dim // 2)))
        
        angle_rads = pos * div_term
        sin_part = torch.sin(angle_rads)
        cos_part = torch.cos(angle_rads)
        
        encoding = torch.zeros(emb_dim, dtype=self.dtype, device=self.device)
        encoding[0::2] = sin_part
        encoding[1::2] = cos_part
        return encoding
    
    def forward(self, date):
        """
        Get temporal embedding for a datetime
        
        Args:
            date: datetime object, numpy datetime64, or datetime string
            
        Returns:
            Temporal embedding tensor [d_model]
        """
        if isinstance(date, np.datetime64):
            date = from_np_to_datetime(date)
            date_str = datetime_to_str(date)
        elif isinstance(date, datetime):
            date_str = datetime_to_str(date)
        elif isinstance(date, str):
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d %H:%M:%S")
        else:
            raise ValueError(f"Unsupported date type: {type(date)}")
        
        return self.embeddings[date_str]
    
    def __getitem__(self, date):
        return self.forward(date)
    
    def all(self):
        """Get all temporal embeddings as a tensor"""
        embeddings_list = []
        for date_str in sorted(self.embeddings.keys()):
            embeddings_list.append(self.embeddings[date_str])
        
        return torch.stack(embeddings_list) 

class MultiScaleTemporalEncoding(nn.Module):
    def __init__(self, dates, emb_dim, device=None, dtype=torch.float32, normalize=True):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.emb_dim = emb_dim
        self.normalize = normalize

        self.embeddings = {
            datetime_to_str(date): self._compute_encoding(date)
            for date in dates
        }
        self.length = len(dates)

    def _sinusoidal_component(self, value, period, emb_dim_part):
        """Computes a sinusoidal encoding for a single temporal component."""
        i = torch.arange(emb_dim_part // 2, dtype=self.dtype, device=self.device)
        div_term = torch.exp(i * (-math.log(10000.0) / (emb_dim_part // 2)))
        angle_rads = (2 * math.pi * value / period) * div_term
        sin_part = torch.sin(angle_rads)
        cos_part = torch.cos(angle_rads)
        encoding = torch.cat([sin_part, cos_part], dim=0)
        return encoding

    def _compute_encoding(self, dt):
        """
        Compute multi-scale temporal encoding for a specific datetime.
        - hour of day
        - day of week
        - day of year
        """
        hour = dt.hour + dt.minute / 60.0
        day_of_week = dt.weekday()
        day_of_year = dt.timetuple().tm_yday

        emb_dim_part = self.emb_dim // 3

        enc_hour = self._sinusoidal_component(hour, 24, emb_dim_part)
        enc_week = self._sinusoidal_component(day_of_week, 7, emb_dim_part)
        enc_year = self._sinusoidal_component(day_of_year, 365, emb_dim_part)

        encoding = torch.cat([enc_hour, enc_week, enc_year])

        if self.normalize:
            encoding = encoding / torch.norm(encoding, p=2)

        return encoding.to(self.device)

    def forward(self, date):
        if isinstance(date, np.datetime64):
            date = from_np_to_datetime(date)
            date_str = datetime_to_str(date)
        elif isinstance(date, datetime):
            date_str = datetime_to_str(date)
        elif isinstance(date, str):
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d %H:%M:%S")
        else:
            raise ValueError(f"Unsupported date type: {type(date)}")

        return self.embeddings[date_str]

    def __getitem__(self, date):
        return self.forward(date)

    def all(self):
        """Return all temporal embeddings as a tensor"""
        embeddings_list = [
            self.embeddings[date_str]
            for date_str in sorted(self.embeddings.keys())
        ]

        return torch.stack(embeddings_list)  
