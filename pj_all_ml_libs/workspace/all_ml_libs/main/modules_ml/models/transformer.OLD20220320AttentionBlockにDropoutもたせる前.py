'''
- Refs
    - https://github.com/LiamMaclean216/Pytorch-Transfomer
'''

import math
import torch
from torch import nn
import torch.nn.functional as torchF

def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m, -1)

def attention(Q, K, V):
    a = a_norm(Q, K)
    return torch.matmul(a, V)

class Key(nn.Module):
    def __init__(self,
        dim_input,
        dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class Value(nn.Module):
    def __init__(self,
        dim_input,
        dim_attn):
        super(Value, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class Query(nn.Module):
    def __init__(self,
        dim_input,
        dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self,
        d_model,
        dropout = None,
        max_len = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, training = True):
        if not training:
            # Dropoutは基本的に予測時で使わない
            self.dropout = None # self.dropout = nn.Dropout(0.0)
        x = x + self.pe[:x.size(1), :].squeeze(1)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self,
        dim_val,
        dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
    
    def forward(self, x, kv = None):
        if kv is None:
            return attention(self.query(x), self.key(x), self.value(x))
        return attention(self.query(x), self.key(kv), self.value(kv))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,
        dim_val,
        dim_attn,
        n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for _ in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))
        self.heads = nn.ModuleList(self.heads)
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = False)
    
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))
        a = torch.stack(a, dim = -1)
        a = a.flatten(start_dim = 2)
        x = self.fc(a)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,
        dim_val,
        dim_attn,
        n_heads = 1,
        dropout = None):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, training = True):
        if not training:
            # Dropoutは基本的に予測時で使わない
            self.dropout = None # self.dropout = nn.Dropout(0.0)
        a = self.attn(x)
        x = self.norm1(x + a)
        x1 = torchF.elu(self.fc1(x))
        if self.dropout is not None:
            x1 = self.dropout(x1)
        a = self.fc2(x1)
        x = self.norm2(x + a) # TODO: x1ではなくx？
        return x

class DecoderLayer(nn.Module):
    def __init__(self,
        dim_val,
        dim_attn,
        n_heads = 1,
        dropout = None):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc, training = True):
        if not training:
            # Dropoutは基本的に予測時で使わない
            self.dropout = None # self.dropout = nn.Dropout(0.0)
        a = self.attn1(x)
        x = self.norm1(a + x)
        a = self.attn2(x, kv = enc)
        x = self.norm2(a + x)
        x1 = torchF.elu(self.fc1(x))
        if self.dropout is not None:
            x1 = self.dropout(x1)
        a = self.fc2(x1)
        x = self.norm3(a + x) # TODO: x1ではなくx？
        return x

class TransformerModel(nn.Module):
    def __init__(self,
        # 特徴量(=説明変数(+目的変数))のカラム数
        input_size,
        # (学習時の？)直前のローリングにおける出力の個数
        ## 1つの目的変数の時系列予測ならばout_seq_lenに合わせる？
        ## 一方enc_seq_lenはローリングにおける入力の個数
        ### 時系列予測なら主に日数？
        dec_seq_len,
        # 出力の個数
        ## 時系列予測なら予測する未来の個数
        ### 主に日数？
        out_seq_len,
        # 以下，隠れ層のパラメータだが詳しくわからん
        dim_val,
        dim_attn,
        ## Encoderの個数
        n_encoder_layers = 1,
        ## Decoderの個数
        n_decoder_layers = 1,
        n_heads = 1,
        # 以下，Dropout
        ## 過学習を抑えるのに有効らしい？
        dropout_enc = None,
        dropout_dec = None,
        dropout_pe = None):
        super(TransformerModel, self).__init__()
        self.dec_seq_len = dec_seq_len

        self.encs = nn.ModuleList()
        for _ in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads, dropout = dropout_enc))
        self.decs = nn.ModuleList()
        for _ in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads, dropout = dropout_dec))
        
        self.pos = PositionalEncoding(dim_val, dropout = dropout_pe)
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)
    
    def forward(self, x, training = True):
        ec = self.encs[0](self.pos(self.enc_input_fc(x), training = training))
        for enc in self.encs[1:]:
            ec = enc(ec, training = training)
        dc = self.decs[0](self.dec_input_fc(x[:, -self.dec_seq_len:]), ec)
        for dec in self.decs[1:]:
            dc = dec(dc, ec, training = training)
        x = self.out_fc(dc.flatten(start_dim = 1))
        return x

