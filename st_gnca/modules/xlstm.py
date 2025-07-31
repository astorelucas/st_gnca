# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class mLSTMblock(nn.Module):
#     def __init__(self, x_example, factor, depth, dropout=0.2):
#         super().__init__()
#         self.input_size = x_example.shape[2]
#         self.hidden_size = int(self.input_size*factor)
        
#         self.ln = nn.LayerNorm(self.input_size)
        
#         self.left = nn.Linear(self.input_size, self.hidden_size)
#         self.right = nn.Linear(self.input_size, self.hidden_size)
        
#         self.init_states(x_example)
    
#     def init_states(self, x_example):
#         self.ct_1 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)
#         self.nt_1 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)

#     def forward(self, x):
#           assert x.ndim == 3
          
#           x = self.ln(x) # layer norm on x
          
#           left = self.left(x) # part left 
#           right = F.silu(self.right(x)) # part right with just swish (silu) function

# class BlockDiagonal(nn.Module):
#     def __init__(self, in_features, out_features, num_blocks, bias=True):
#         super(BlockDiagonal, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_blocks = num_blocks

#         assert out_features % num_blocks == 0
        
#         block_out_features = out_features // num_blocks
        
#         self.blocks = nn.ModuleList([
#             nn.Linear(in_features, block_out_features, bias=bias)
#             for _ in range(num_blocks)
#         ])
        
#     def forward(self, x):
#         x = [block(x) for block in self.blocks]
#         x = torch.cat(x, dim=-1)
#         return x
    
# class CausalConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
#         super(CausalConv1D, self).__init__()
#         self.padding = (kernel_size - 1) * dilation
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

#     def forward(self, x):
#         x = self.conv(x)
#         return x[:, :, :-self.padding]
    
# class mLSTMblock(nn.Module):
#     def __init__(self, x_example, factor, depth, dropout=0.2):
#         super().__init__()
#         self.input_size = x_example.shape[2]
#         self.hidden_size = int(self.input_size*factor)
        
#         self.ln = nn.LayerNorm(self.input_size)
        
#         self.left = nn.Linear(self.input_size, self.hidden_size)
#         self.right = nn.Linear(self.input_size, self.hidden_size)
        
#         self.conv = CausalConv1D(self.hidden_size, self.hidden_size, int(self.input_size/10)) 
#         self.drop = nn.Dropout(dropout+0.1)
        
#         self.lskip = nn.Linear(self.hidden_size, self.hidden_size)
        
#         self.wq = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
#         self.wk = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
#         self.wv = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
#         self.dropq = nn.Dropout(dropout/2)
#         self.dropk = nn.Dropout(dropout/2)
#         self.dropv = nn.Dropout(dropout/2)
        
#         self.i_gate = nn.Linear(self.hidden_size, self.hidden_size)
#         self.f_gate = nn.Linear(self.hidden_size, self.hidden_size)
#         self.o_gate = nn.Linear(self.hidden_size, self.hidden_size)

#         self.ln_c = nn.LayerNorm(self.hidden_size)
#         self.ln_n = nn.LayerNorm(self.hidden_size)
        
#         self.lnf = nn.LayerNorm(self.hidden_size)
#         self.lno = nn.LayerNorm(self.hidden_size)
#         self.lni = nn.LayerNorm(self.hidden_size)

#         self.drop2 = nn.Dropout(dropout)

#     def init_states(self, x_example):
#           self.ct_1 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)
#           self.nt_1 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)
      
#     def forward(self, x):
#         assert x.ndim == 3
        
#         x = self.ln(x) # layer norm on x
        
#         left = self.left(x) # part left 
#         right = F.silu(self.right(x)) # part right with just swish (silu) function

#         left_left = left.transpose(1, 2)
#         left_left = F.silu( self.drop( self.conv( left_left ).transpose(1, 2) ) )
#         l_skip = self.lskip(left_left)

#         # start mLSTM
#         q = self.dropq(self.wq(left_left))
#         k = self.dropk(self.wk(left_left))
#         v = self.dropv(self.wv(left))
        
#         i = torch.exp(self.lni(self.i_gate(left_left)))
#         f = torch.exp(self.lnf(self.f_gate(left_left)))
#         o = torch.sigmoid(self.lno(self.o_gate(left_left)))

#         ct_1 = self.ct_1
#         ct = f*ct_1 + i*v*k
#         ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
#         self.ct_1 = ct.detach()
        
#         nt_1 = self.nt_1
#         nt = f*nt_1 + i*k
#         nt =torch.mean( self.ln_n(nt), [0, 1], keepdim=True)
#         self.nt_1 = nt.detach()
        
#         ht = o * ((ct*q) / torch.max(nt*q)) # [batchs_size, ?, hiddden_size]
#         # end mLSTM
#         ht = ht
        
#         left = self.drop2(self.GN(ht + l_skip))

# class mLSTMblock(nn.Module):
#     def __init__(self, x_example, factor, depth, dropout=0.2):
#         super().__init__()
#         self.input_size = x_example.shape[2]
#         self.hidden_size = int(self.input_size*factor)
        
#         self.ln = nn.LayerNorm(self.input_size)
        
#         self.left = nn.Linear(self.input_size, self.hidden_size)
#         self.right = nn.Linear(self.input_size, self.hidden_size)
        
#         self.conv = CausalConv1D(self.hidden_size, self.hidden_size, int(self.input_size/10)) 
#         self.drop = nn.Dropout(dropout+0.1)
        
#         self.lskip = nn.Linear(self.hidden_size, self.hidden_size)
        
#         self.wq = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
#         self.wk = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
#         self.wv = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
#         self.dropq = nn.Dropout(dropout/2)
#         self.dropk = nn.Dropout(dropout/2)
#         self.dropv = nn.Dropout(dropout/2)
        
#         self.i_gate = nn.Linear(self.hidden_size, self.hidden_size)
#         self.f_gate = nn.Linear(self.hidden_size, self.hidden_size)
#         self.o_gate = nn.Linear(self.hidden_size, self.hidden_size)

#         self.ln_c = nn.LayerNorm(self.hidden_size)
#         self.ln_n = nn.LayerNorm(self.hidden_size)
        
#         self.lnf = nn.LayerNorm(self.hidden_size)
#         self.lno = nn.LayerNorm(self.hidden_size)
#         self.lni = nn.LayerNorm(self.hidden_size)
        
#         self.GN = nn.LayerNorm(self.hidden_size)
#         self.ln_out = nn.LayerNorm(self.hidden_size)

#         self.drop2 = nn.Dropout(dropout)
        
#         self.proj = nn.Linear(self.hidden_size, self.input_size)
#         self.ln_proj = nn.LayerNorm(self.input_size)
        
#         self.init_states(x_example)
    
#     def init_states(self, x_example):
#         self.ct_1 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)
#         self.nt_1 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)
    
#     def forward(self, x):
#         assert x.ndim == 3
        
#         x = self.ln(x) # layer norm on x
        
#         left = self.left(x) # part left 
#         right = F.silu(self.right(x)) # part right with just swish (silu) function

#         left_left = left.transpose(1, 2)
#         left_left = F.silu( self.drop( self.conv( left_left ).transpose(1, 2) ) )
#         l_skip = self.lskip(left_left)

#         # start mLSTM
#         q = self.dropq(self.wq(left_left))
#         k = self.dropk(self.wk(left_left))
#         v = self.dropv(self.wv(left))
        
#         i = torch.exp(self.lni(self.i_gate(left_left)))
#         f = torch.exp(self.lnf(self.f_gate(left_left)))
#         o = torch.sigmoid(self.lno(self.o_gate(left_left)))

#         ct_1 = self.ct_1
#         ct = f*ct_1 + i*v*k
#         ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
#         self.ct_1 = ct.detach()
        
#         nt_1 = self.nt_1
#         nt = f*nt_1 + i*k
#         nt =torch.mean( self.ln_n(nt), [0, 1], keepdim=True)
#         self.nt_1 = nt.detach()
        
#         ht = o * ((ct*q) / torch.max(nt*q)) # [batchs_size, ?, hiddden_size]
#         # end mLSTM
#         ht = ht
        
#         left = self.drop2(self.GN(ht + l_skip))
        
#         out = self.ln_out(left * right)
#         out = self.ln_proj(self.proj(out))
        
#         return out
    

# class sLSTMblock(nn.Module):
#     def __init__(self, x_example, depth, dropout=0.2):
#         super().__init__()
#         self.input_size = x_example.shape[2]
#         conv_channels = x_example.shape[1]
        
#         self.ln = nn.LayerNorm(self.input_size)
        
#         self.conv = CausalConv1D(self.input_size, self.input_size, int(self.input_size/8))
#         self.drop = nn.Dropout(dropout)
        
#         self.i_gate = BlockDiagonal(self.input_size, self.input_size, depth)
#         self.f_gate = BlockDiagonal(self.input_size, self.input_size, depth)
#         self.o_gate = BlockDiagonal(self.input_size, self.input_size, depth)
#         self.z_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        
#         self.ri_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
#         self.rf_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
#         self.ro_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
#         self.rz_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)

#         self.ln_i = nn.LayerNorm(self.input_size)
#         self.ln_f = nn.LayerNorm(self.input_size)
#         self.ln_o = nn.LayerNorm(self.input_size)
#         self.ln_z = nn.LayerNorm(self.input_size)
        
#         self.GN = nn.LayerNorm(self.input_size)
#         self.ln_c = nn.LayerNorm(self.input_size)
#         self.ln_n = nn.LayerNorm(self.input_size)
#         self.ln_h = nn.LayerNorm(self.input_size)

#     def init_states(self, x):
#         self.nt_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
#         self.ct_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
#         self.ht_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
#         self.mt_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
        
#     def forward(self, x):
#         x = self.ln(x)
        
#         x_conv = F.silu( self.drop(self.conv( x.transpose(1, 2) ).transpose(1, 2) ) )
        
#         # start sLSTM
#         ht_1 = self.ht_1
        
#         i = torch.exp(self.ln_i( self.i_gate(x_conv) + self.ri_gate(ht_1) ) )
#         f = torch.exp( self.ln_f(self.f_gate(x_conv) + self.rf_gate(ht_1) ) )

#         m = torch.max(torch.log(f)+self.mt_1[:, 0, :].unsqueeze(1), torch.log(i))
#         i = torch.exp(torch.log(i) - m)
#         f = torch.exp(torch.log(f) + self.mt_1[:, 0, :].unsqueeze(1)-m)
#         self.mt_1 = m.detach()
        
#         o = torch.sigmoid( self.ln_o(self.o_gate(x) + self.ro_gate(ht_1) ) )
#         z = torch.tanh( self.ln_z(self.z_gate(x) + self.rz_gate(ht_1) ) )
        
#         ct_1 = self.ct_1
#         ct = f*ct_1 + i*z
#         ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
#         self.ct_1 = ct.detach()
        
#         nt_1 = self.nt_1
#         nt = f*nt_1 + i
#         nt = torch.mean(self.ln_n(nt), [0, 1], keepdim=True)
#         self.nt_1 = nt.detach()
        
#         ht = o*(ct/nt) # torch.Size([4, 8, 16])
#         ht = torch.mean(self.ln_h(ht), [0, 1], keepdim=True)
#         self.ht_1 = ht.detach()
#         # end sLSTM
        
#         slstm_out = self.GN(ht)

# class xLSTM(nn.Module):
#     def __init__(self, layers, x_example, depth=4, factor=2):
#         super(xLSTM, self).__init__()

#         self.layers = nn.ModuleList()
#         for layer_type in layers:
#             if layer_type == 's':
#                 layer = sLSTMblock(x_example, depth)
#             elif layer_type == 'm':
#                 layer = mLSTMblock(x_example, factor, depth)
#             else:
#                 raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
#             self.layers.append(layer)
    
#     def init_states(self, x):
#         [l.init_states(x) for l in self.layers]
        
#     def forward(self, x):
#         x_original = x.clone()
#         for l in self.layers:
#              x = l(x) + x_original

#         return x