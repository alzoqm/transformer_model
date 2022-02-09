import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import os
import tqdm
import einops

from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchinfo import summary
from tqdm import tqdm
from tqdm.notebook import trange

class PatchEmbedding(nn.Module):
  def __init__(self, in_channels, patch_size, emb_size, img_size):
    super().__init__()
    self.patch_size = patch_size
    self.in_channels = in_channels
    self.reshape = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
    self.projection = nn.Linear(self.patch_size * self.patch_size * self.in_channels, emb_size)

    self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size)) #-> 분류문제일 경우 활용
    self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size)) #분류 문제일 경우 **2 + 1

  def forward(self, x):
    batch_size = x.shape[0]
    x = self.reshape(x)
    x = self.projection(x)

    cls_token = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
    x = torch.cat([cls_token, x], dim=1)
    #분류


    x += self.positions
    return x

class MultiHeadAttention(nn.Module):
  def __init__(self, emb_size, num_heads, dropout):
    super().__init__()
    
    assert emb_size % num_heads == 0

    self.emb_size = emb_size
    self.num_heads = num_heads
    self.head_dim = emb_size // num_heads

    self.q_linear = nn.Linear(emb_size, emb_size)
    self.k_linear = nn.Linear(emb_size, emb_size)
    self.v_linear = nn.Linear(emb_size, emb_size)

    self.linear = nn.Linear(emb_size, emb_size)

    self.dropout = nn.Dropout(dropout)

    self.scale = 1 / (self.head_dim ** 0.5)

  def forward(self, q, k, v):
    batch_size = q.shape[0]

    q = self.q_linear(q)
    k = self.k_linear(k)
    v = self.v_linear(v)

    q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    logits = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
    attention = torch.softmax(logits, dim = -1)
    attention = self.dropout(attention)

    outputs = torch.matmul(attention, v)
    outputs = outputs.permute(0, 2, 1, 3).contiguous()
    outputs = outputs.view(batch_size, -1, self.emb_size)
    outputs = self.linear(outputs)

    return outputs

class EncoderLayer(nn.Module):
  def __init__(self, dff, emb_size, num_heads, dropout):
    super().__init__()
    self.attn_layer = MultiHeadAttention(emb_size, num_heads, dropout)
    self.linear1 = nn.Linear(emb_size, dff)
    self.linear2 = nn.Linear(dff, emb_size)

    self.layer_norm1 = nn.LayerNorm(emb_size, eps=1e-9)
    self.layer_norm2 = nn.LayerNorm(emb_size, eps=1e-9)

    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

    self.gelu = nn.GELU()

  def forward(self, inputs):
    attn_outputs = self.layer_norm1(inputs)
    attn_outputs = self.attn_layer(attn_outputs, attn_outputs, attn_outputs)
    attn_outputs = self.dropout1(attn_outputs)
    attn_outputs = attn_outputs + inputs

    outputs = self.layer_norm2(attn_outputs)
    outputs = self.linear1(outputs)
    outputs = self.gelu(outputs)
    outputs = self.dropout2(outputs)
    outputs = self.linear2(outputs)
    outputs = self.dropout3(outputs)
    outputs = attn_outputs + outputs

    return outputs

class VisionTransformer(nn.Module):
  def __init__(self, num_layers, in_channels, patch_size, emb_size, img_size, dff, num_heads, dropout):
    super().__init__()
    
    assert img_size % patch_size == 0

    self.embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
    self.dropout = nn.Dropout(dropout)
    self.layers = nn.ModuleList([EncoderLayer(dff, emb_size, num_heads, dropout) for _ in range(num_layers)])

    
  def forward(self, inputs):
    outputs = self.embedding(inputs)
    outputs = self.dropout(outputs)
    for layer in self.layers:
      outputs = layer(outputs)
    return outputs

class Classifier(nn.Module):
  def __init__(self, emb_size, num_classes):
    super().__init__()
    self.reduce = Reduce('b n e -> b e', reduction='mean')
    self.layer_norm = nn.LayerNorm(emb_size, eps=1e-9)
    self.linear = nn.Linear(emb_size, num_classes)

  def forward(self, inputs):
    outputs = self.reduce(inputs)
    outputs = self.layer_norm(outputs)
    outputs = self.linear(outputs)

    return outputs

class ViT(nn.Module):
  def __init__(self, num_layers, in_channels, patch_size, emb_size, img_size, dff, num_heads, dropout, num_classes):
    super().__init__()
    self.transformer = VisionTransformer(num_layers, in_channels, patch_size, emb_size, img_size, dff, num_heads, dropout)
    self.classifier = Classifier(emb_size, num_classes)

  def forward(self, inputs):
    outputs = self.transformer(inputs)
    outputs = self.classifier(outputs)

    return outputs

if __name__ == "__main__":
  NUM_LAYERS = 5
  IN_CHANNELS = 3
  PATCH_SIZE = 2
  EMB_SIZE = 512
  IMG_SIZE = 32
  DFF = 2048
  NUM_HEADS = 8
  DROPOUT = 0.1
  NUM_CLASSES = 10
  
  model = ViT(NUM_LAYERS, IN_CHANNELS, PATCH_SIZE, EMB_SIZE, IMG_SIZE, DFF, NUM_HEADS, DROPOUT, NUM_CLASSES)
