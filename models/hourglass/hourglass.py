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

  def make_decoder_mask(self, attn_mat):
    return torch.tril(torch.ones_like(attn_mat)) == 0

  def forward(self, q, k, v, mask=None, rpe=True):
    batch_size = q.shape[0]

    q = self.q_linear(q)
    k = self.k_linear(k)
    v = self.v_linear(v)

    q = torch.reshape(q, shape=(batch_size, -1, self.num_heads, self.depth)).permute(0, 2, 1, 3)
    k = torch.reshape(k, shape=(batch_size, -1, self.num_heads, self.depth)).permute(0, 2, 1, 3)
    v = torch.reshape(v, shape=(batch_size, -1, self.num_heads, self.depth)).permute(0, 2, 1, 3)

    attn_mat = q @ k.transpose(-2, -1) # -> shape(b, m, m)
    attn_mat *= self.scale
    decoder_mask = self.make_decoder_mask(attn_mat)
    if rpe:
      for i in range(attn_mat.shape[0]):
        seq_len = attn_mat[i].shape[-1]
        RPE = self.make_relative_positional_encoding(seq_len)
        attn_mat[i]  = attn_mat[i] + RPE

    if mask is not None:
      mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
      attn_mat.masked_fill_(mask, -1e9)
      
    attn_mat.masked_fill_(decoder_mask, -1e9)
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
  def __init__(self, d_model, num_heads, k):
    super().__init__()
    self.k = k
    self.d_model = d_model
    self.attn = MultiHeadAttention(d_model, num_heads, 0.0)
  
  def forward(self, x):
    x_reshape = einops.reduce(x, 'b (m k) d -> b m d', 'mean', k = self.k)
    x = x_reshape + self.attn(x_reshape, x, x, None, False)[0]
    return x
    


class LinearPooling(nn.Module):
  def __init__(self, d_model, num_heads, k):
    super().__init__()
    self.proj = nn.Linear(d_model*k, d_model)
    self.k = k
    self.d_model = d_model
    self.attn = MultiHeadAttention(d_model, num_heads, 0.0)

  def forward(self, x):
    assert x.shape[-2] % self.k == 0, f"Error! shorten factor can't divide length. shorten factor: {self.k}, length: {x.shape[-2]}."
    x_reshape = einops.rearrange(x, 'b (m k) d -> b m (k d)', k = self.k)
    x_reshape = self.proj(x_reshape)
    x = x_reshape + self.attn(x_reshape, x, x, None, False)[0]
    return x


class NaiveUpsampling(nn.Module):
  def __init__(self, d_model, num_heads, k):
    super().__init__()
    self.k = k
    self.d_model = d_model
    self.attn = MultiHeadAttention(d_model, num_heads, 0.0)

  def forward(self, x, resi_x):
    x_reshape = einops.repeat(x, 'b m d -> b (m k) d', k=self.k)
    x = x_reshape + self.attn(x_reshape, x, x, None, False)[0]
    return x


class LinearUpsampling(nn.Module):
  def __init__(self, d_model, num_heads, k):
    super().__init__()
    self.proj = nn.Linear(d_model, d_model*k)
    self.k = k
    self.d_model = d_model
    self.attn = MultiHeadAttention(d_model, num_heads, 0.0)

  def forward(self, x, resi_x):
    x_reshape = self.proj(x)
    x_reshape = einops.rearrange(x_reshape, 'b m (k d) -> b (m k) d', k = self.k)
    x = x_reshape + self.attn(x_reshape, x, x, None, False)[0]
    return x


class Hourglass(torch.nn.Module):
  r"""" Hourglass
  args 설명:
    max_len: 입력 토큰의 최대 갯수를 말합니다.

    num_layers: 트랜스포머 디코더 레이어의 갯수를 리스트 형태로 입력 받습니다. index값 0과 -1의 값은 바닐라 트랜스포머의 갯수이며
    그 사이의 값은 shorten과 upsample 이후의 트랜스포머 레이어의 수를 입력합니다.
    ex) num_layers = [2, 2, 2, 2, 2] -> num_layers[0], num_layers[-1]: pre vanilla layer 갯수, post vanilla layer 갯수

    shorten_factors: pooling과 upsample할 횟수를 리스트 형태로 입력 받습니다. 리스트 내의 값들은 각각 pooling과 upsample 시행할때 줄이는 길이를 말합니다.
    ex) shorten_factors = [2, 2] -> upsample과 pooling 모두 len(shorten_factors) = 2번 반복하며, 각각의 시행할때 마다 length // 2로 시행됩니다.
    ****shorten_factors의 값은 max_len으로 나누어 떨어지는 값을 가져야 합니다. ex) shorten_factors=[3], max_len =128 -> ERROR 발생!****
    ****shorten_factors의 리스트의 길이와 num_layers의 리스트의 길이는 아래의 공식을 따라야 합니다.****
           len(num_layers) == len(shorten_factors) * 2 + 1

    updown_mode: pooling과 upsample의 방식을 설정할 수 있습니다. "linear"를 입력시 linear pooling 과 linear upsample을 시행하며,
    "naive"를 입력시 naive pooling, naive upsample를 시행합니다. (대문자도 허용)

  """
  def __init__(self, max_len, vocab_size, num_layers, dff, d_model, num_heads, dropout, shorten_factors, updown_mode='linear'):
    super().__init__()
    assert len(num_layers) == len(shorten_factors) * 2 + 1, f"List Length Error! must be len(num_layers) == len(shorten_factors) * 2 + 1. now len(shorten_factors): {len(shorten_factors)} len(num_layers): {len(num_layers)}."
    self.shorten_factors_len = len(shorten_factors)
    self.word_emb = nn.Embedding(vocab_size, d_model)
    

    ####pre vanilla layers####
    self.pre_vanilla_layers = nn.ModuleList([
                                             DecoderLayer(dff, d_model, num_heads, dropout) for i in range(num_layers[0])
    ])

    ####shortening layers####
    if updown_mode.lower() == 'linear':
      self.shortening_layers = nn.ModuleList([
                                              LinearPooling(d_model, num_heads, shorten_factors[i]) for i in range(len(shorten_factors))
      ])
    elif updown_mode.lower() == 'naive':
      self.shortening_layers = nn.ModuleList([
                                              NaivePooling(d_model, num_heads, shorten_factors[i]) for i in range(len(shorten_factors))
      ])
    else:
      raise Exception("unknown updown mode")

    shorten_factors.reverse() #reverse 이후 upsampling

    ####upsampling layers####
    if updown_mode.lower() == 'linear':
      self.upsampling_layers = nn.ModuleList([
                                              LinearUpsampling(d_model, num_heads, shorten_factors[i]) for i in range(len(shorten_factors))
      ])
    elif updown_mode.lower() == 'naive':
      self.upsampling_layers = nn.ModuleList([
                                              NaiveUpsampling(d_model, num_heads, shorten_factors[i]) for i in range(len(shorten_factors))
      ])
    else:
      raise Exception("unknown updown mode")

    ####shortening transformer layers####
    if len(shorten_factors) > 1:
      self.shortened_layers = nn.ModuleList([
                                            nn.ModuleList([
                                                            DecoderLayer(dff, d_model, num_heads, dropout) for j in range(num_layers[i+1])
                                            ]) for i in range(len(shorten_factors) - 1)
      ])

    self.middle_layers = nn.ModuleList([
                                    DecoderLayer(dff, d_model, num_heads, dropout) for i in range(num_layers[len(shorten_factors)])
    ])

    ####upsampling transformer layers####
    if len(shorten_factors) > 1:
      self.up_layers = nn.ModuleList([
                                      nn.ModuleList([
                                                     DecoderLayer(dff, d_model, num_heads, dropout) for j in range(num_layers[i])
                                      ]) for i in range(len(num_layers) // 2 + 1, len(num_layers) - 1)
      ])

    ####post vanilla layers####
    self.post_vanilla_layers = nn.ModuleList([
                                              DecoderLayer(dff, d_model, num_heads, dropout) for i in range(num_layers[-1])
    ])

  def make_pad_mask(self, x):
    B, L = x.shape
    mask = x == 0 #padding index = 0
    return mask.unsqueeze(1).expand(B, L, L)

  def forward(self, x):
    mask = self.make_pad_mask(x) #pre, post vanila layer에만 적용할 예정
    output = self.word_emb(x)

    residual_list = []

    for i in range(self.shorten_factors_len):
      if i == 0:
        for layer in self.pre_vanilla_layers: 
          output, attn_mat = layer(output, mask) #pre layer padding mask 적용
      else:
        for layer in self.shortened_layers[i-1]:
          output, attn_mat = layer(output)
      residual_list.append(output)
      output = self.shortening_layers[i](output)

    residual_list.reverse()
    for layer in self.middle_layers:
      output, attn_mat = layer(output)

    for i in range(self.shorten_factors_len):
      output = self.upsampling_layers[i](output, residual_list[i])
      #output = output + residual_list[i] #residual connect
      
      if i == self.shorten_factors_len - 1:
        for index, layer in enumerate(self.post_vanilla_layers):
          output, attn_mat = layer(output, mask) #post layer padding mask 적용
          if index == 0:
            output = output + residual_list[i] #첫 번째 레이어일 경우 residual connect      
      else:
        for index, layer in enumerate(self.up_layers[i]):
          output, attn_mat = layer(output)
          if index == 0:
            output = output + residual_list[i]

    return output

  
if __name__ == "__main__":
  print(Hourglass.__doc__)
  hourglass = Hourglass(128, 3000, [2, 2, 2, 2, 2, 2, 2, 2, 2], 256*4, 256, 8, 0.1, [2, 2, 2, 2], 'linear')

  x = torch.arange(3*256).reshape(3, 256)
  y = hourglass(x)


  #print(hourglass.parameters)
  print(y.shape)

