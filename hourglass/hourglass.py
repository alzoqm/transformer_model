# -*- coding: utf-8 -*-

!pip install einops

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import einops

class MultiHeadAttention(torch.nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.dropout = torch.nn.Dropout(dropout)
    self.softmax = nn.Softmax(dim=-1)

    self.q_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)

    self.proj = nn.Linear(d_model, d_model)

    self.depth = d_model // num_heads
    self.scale = self.depth ** -0.5

  def make_relative_positional_encoding(self, seq_len):
    results = []
    for i in range(seq_len):
      line = list(range(0-i, seq_len - i))
      results.append(line)
    
    return torch.Tensor(results)

  def forward(self, q, k, v, mask=None):
    batch_size = q.shape[0]

    q = self.q_linear(q)
    k = self.k_linear(k)
    v = self.v_linear(v)

    q = torch.reshape(q, shape=(batch_size, -1, self.num_heads, self.depth)).permute(0, 2, 1, 3)
    k = torch.reshape(k, shape=(batch_size, -1, self.num_heads, self.depth)).permute(0, 2, 1, 3)
    v = torch.reshape(v, shape=(batch_size, -1, self.num_heads, self.depth)).permute(0, 2, 1, 3)

    attn_mat = q @ k.transpose(-2, -1)
    attn_mat *= self.scale
    for i in range(attn_mat.shape[0]):
      seq_len = attn_mat[i].shape[-1]
      RPE = self.make_relative_positional_encoding(seq_len)
      attn_mat[i]  = attn_mat[i] + RPE

    if mask is not None:
      mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
      attn_mat.masked_fill_(mask, -1e9)
    
    attn_mat = self.softmax(attn_mat)

    output = torch.reshape((attn_mat @ v).permute(0, 2, 1, 3), shape=(batch_size, -1, self.d_model))
    output = self.proj(output)

    return output, attn_mat

class FFNN(torch.nn.Module):
  def __init__(self, dff, d_model, dropout):
    super().__init__()
    self.linear1 = nn.Linear(d_model, dff)
    self.linear2 = nn.Linear(dff, d_model)
    self.dropout = nn.Dropout(dropout)
    self.gelu = nn.GELU()

  def forward(self, x):
    x = self.linear1(x)
    x = self.gelu(x)
    x = self.dropout(x)
    x = self.linear2(x)

    return x

class DecoderLayer(torch.nn.Module):
  def __init__(self, dff, d_model, num_heads, dropout):
    super().__init__()
    self.attn_layer = MultiHeadAttention(d_model, num_heads, dropout)
    self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-7)

    self.ffnn = FFNN(dff, d_model, dropout)
    self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-7)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    attn_output, attn_mat = self.attn_layer(x, x, x, mask)
    attn_output = self.dropout(attn_output)
    attn_output = self.layer_norm1(x + attn_output)

    output = self.ffnn(attn_output)
    output = self.dropout(output)
    output = self.layer_norm2(attn_output + output)
    return output, attn_mat

class NaivePooling(nn.Module):
  def __init__(self, k):
    super().__init__()
    self.k = k
  
  def forward(self, x):
    return einops.reduce(x, 'b (m k) d -> b m d', 'mean', k = self.k)
    

class LinearPooling(nn.Module):
  def __init__(self, d_model, k):
    super().__init__()
    self.proj = nn.Linear(d_model*k, d_model)
    self.k = k
    self.d_model = d_model

  def forward(self, x):
    assert x.shape[-2] % self.k == 0
    x = einops.rearrange(x, 'b (m k) d -> b m (k d)', k = self.k)
    x = self.proj(x)
    return x

class NaiveUpsampling(nn.Module):
  def __init__(self, k):
    super().__init__()
    self.k = k

  def forward(self, x):
    return einops.repeat(x, 'b m d -> b (m k) d', k=self.k)

class LinearUpsampling(nn.Module):
  def __init__(self, d_model, k):
    super().__init__()
    self.proj = nn.Linear(d_model, d_model*k)
    self.k = k
    self.d_model = d_model

  def forward(self, x):
    x = self.proj(x)
    x = einops.rearrange(x, 'b m (k d) -> b (m k) d', k = self.k)
    return x

class Hourglass(torch.nn.Module):
  def __init__(self, max_len, vocab_size, num_layers, dff, d_model, num_heads, dropout, k, updown_mode='Linear'):
    super().__init__()
    assert len(num_layers) == len(k) * 2 + 1, f"List Length Error! must be len(num_layers) == len(k) * 2 + 1. now len(k): {len(k)} len(num_layers): {len(num_layers)}."
    self.k_len = len(k)
    self.word_emb = nn.Embedding(vocab_size, d_model)
    
    self.pre_vanilla_layers = nn.ModuleList([
                                             DecoderLayer(dff, d_model, num_heads, dropout) for i in range(num_layers[0])
    ])

    if updown_mode.lower() == 'linear':
      self.shortening_layers = nn.ModuleList([
                                              LinearPooling(d_model, k[i]) for i in range(len(k))
      ])
    elif updown_mode.lower() == 'naive':
      self.shortening_layers = nn.ModuleList([
                                              NaivePooling(k[i]) for i in range(len(k))
      ])
    else:
      raise Exception("unknown updown mode")

    k.reverse()

    if updown_mode.lower() == 'linear':
      self.upsampling_layers = nn.ModuleList([
                                              LinearUpsampling(d_model, k[i]) for i in range(len(k))
      ])
    elif updown_mode.lower() == 'naive':
      self.upsampling_layers = nn.ModuleList([
                                              NaiveUpsampling(k[i]) for i in range(len(k))
      ])
    else:
      raise Exception("unknown updown mode")


    if len(k) > 1:
      self.Shortened_layers = nn.ModuleList([
                                            nn.ModuleList([
                                                            DecoderLayer(dff, d_model, num_heads, dropout) for j in range(num_layers[i+1])
                                            ]) for i in range(len(k) - 1)
      ])

    self.middle_layers = nn.ModuleList([
                                    DecoderLayer(dff, d_model, num_heads, dropout) for i in range(num_layers[len(k)])
    ])


    if len(k) > 1:
      self.up_layers = nn.ModuleList([
                                      nn.ModuleList([
                                                     DecoderLayer(dff, d_model, num_heads, dropout) for j in range(num_layers[i])
                                      ]) for i in range(len(num_layers) // 2 + 1, len(num_layers) - 1)
      ])

    self.post_vanilla_layers = nn.ModuleList([
                                              DecoderLayer(dff, d_model, num_heads, dropout) for i in range(num_layers[-1])
    ])

  def forward(self, x):
    output = self.word_emb(x)

    residual_list = []
    for i in range(self.k_len):
      if i == 0:
        for layer in self.pre_vanilla_layers:
          output, attn_mat = layer(output)
      else:
        for layer in self.Shortened_layers[i-1]:
          output, attn_mat = layer(output)
      residual_list.append(output)
      output = self.shortening_layers[i](output)

    residual_list.reverse()
    for layer in self.middle_layers:
      output, attn_mat = layer(output)

    for i in range(self.k_len):
      output = self.upsampling_layers[i](output)
      output = output + residual_list[i]
      
      if i == self.k_len - 1:
        for index, layer in enumerate(self.post_vanilla_layers):
          output, attn_mat = layer(output)
          if index == 0:
            output = output + residual_list[i]         
      else:
        for index, layer in enumerate(self.up_layers[i]):
          output, attn_mat = layer(output)
          if index == 0:
            output = output + residual_list[i]

    return output

hourglass = Hourglass(128, 3000, [2, 2, 2, 2, 2], 256*4, 256, 8, 0.1, [2, 2], 'linear')

x = torch.arange(3*128).reshape(3, 128)
y = hourglass(x)

print(y.shape)

print(hourglass.parameters)

