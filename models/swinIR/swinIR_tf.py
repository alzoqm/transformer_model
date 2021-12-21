import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import cv2
import tqdm
import torch
import math

from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

"""
Parameter

HEIGHT = 128
WIDTH = 128
PATCH_SIZE = 1
IN_CHANS = 3
EMB_SIZE = 240
DEPTHS = [6, 6, 6, 6, 6]
NUM_HEADS = [8, 8, 8, 8, 8]
WINDOW_SIZE = 8
DFF = EMB_SIZE * 4
DROPOUT = 0.1
APE = True
UPSCALE = 2
IMG_RANGE = 1.
"""

def window_partition(x, window_size):
  _, height, width, channels = x.shape

  x = tf.reshape(x, shape=(-1, height // window_size, window_size, width // window_size, window_size, channels))
  x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
  window = tf.reshape(x, shape=(-1, window_size, window_size, channels))
  return window

def window_reverse(window, window_size, height, width, channels):
  patch_num_y = height // window_size
  patch_num_x = width // window_size

  x = tf.reshape(window, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels))
  x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
  x = tf.reshape(x, shape=(-1, height, width, channels))
  return x

class WindowAttention(tf.keras.layers.Layer):
  def __init__(self, window_size, emb_size, num_heads, dropout):
    super().__init__()
    self.emb_size = emb_size
    self.window_size = window_size
    self.num_heads = num_heads
    self.scale = (emb_size//num_heads) ** -0.5

    self.qkv_linear = tf.keras.layers.Dense(emb_size*3)

    self.linear = tf.keras.layers.Dense(emb_size)

    self.dropout = tf.keras.layers.Dropout(dropout)

    num_window_elements = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
    self.relative_position_bias_table = self.add_weight(shape=(num_window_elements, self.num_heads), initializer=tf.initializers.Zeros(), trainable=True)

    coords_h = np.arange(self.window_size[0])
    coords_w = np.arange(self.window_size[1])
    coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = np.stack(coords_matrix)
    coords_flatten = coords.reshape(2, -1)

    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.transpose([1, 2, 0])
    relative_coords[:, :, 0] += self.window_size[0] - 1
    relative_coords[:, :, 1] += self.window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)

    self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(relative_position_index), trainable=False)

  def call(self, input, mask=None):
    _, size, channels = input.shape
    depth = channels // self.num_heads

    qkv = self.qkv_linear(input)
    qkv = tf.reshape(qkv, shape=(-1, size, 3, self.num_heads, depth))
    qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale
    k = tf.transpose(k, perm=[0, 1, 3, 2])
    attn = q @ k

    num_window_elements = self.window_size[0] * self.window_size[1]
    relative_position_index_flat = tf.reshape(self.relative_position_index, shape=(-1, ))

    relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat)
    relative_position_bias = tf.reshape(relative_position_bias, shape=(num_window_elements, num_window_elements, -1))
    relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
    attn = attn + tf.expand_dims(relative_position_bias, axis=0)

    if mask is not None:
      nW = mask.get_shape()[0]
      mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
      attn = (tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size)) + mask_float)
      attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
      attn = tf.keras.activations.softmax(attn, axis=-1)
    else:
      attn = tf.keras.activations.softmax(attn, axis=-1)
    attn = self.dropout(attn)

    output = attn @ v
    output = tf.transpose(output, perm=(0, 2, 1, 3))
    output = tf.reshape(output, shape=(-1, size, channels))
    output = self.linear(output)
    output = self.dropout(output)
    return output

class MLP(tf.keras.layers.Layer):
  def __init__(self, dff, emb_size, dropout):
    super().__init__()
    self.layer1 = tf.keras.layers.Dense(dff)
    self.layer2 = tf.keras.layers.Dense(emb_size)
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, input):
    output = self.layer1(input)
    output = tf.nn.gelu(output)
    output = self.dropout(output)
    output = self.layer2(output)
    output = self.dropout(output)
    return output

class SwinTransformerBlock(tf.keras.layers.Layer):
  def __init__(self, window_size, input_resolution, dff, emb_size, shift_size, num_heads, dropout):
    super().__init__()
    self.window_size = window_size
    self.dff = dff
    self.emb_size = emb_size
    self.shift_size = shift_size
    self.num_heads = num_heads
    self.input_resolution = input_resolution

    if min(input_resolution) <= self.window_size:
      self.shift_size = 0
      self.window_size = min(input_resolution)
    assert 0 <= self.shift_size < self.window_size

    self.norm_layer1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.attn_layer = WindowAttention((window_size, window_size), emb_size, num_heads, dropout)

    self.norm_layer2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.mlp = MLP(dff, emb_size, dropout)

    self.dropout = tf.keras.layers.Dropout(dropout)

    if self.shift_size > 0:
      self.input_attn_mask = self.calculate_mask(self.input_resolution)
    else:
      self.input_attn_mask = None


  def calculate_mask(self, input_size):
    height, width = input_size

    h_slice = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None), )
    w_slice = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None), )
    mask_array = np.zeros((1, height, width, 1))

    count = 0
    for h in h_slice:
      for w in w_slice:
        mask_array[:, h, w, :] = count
        count += 1
    mask_array = tf.convert_to_tensor(mask_array)

    mask_windows = window_partition(mask_array, self.window_size)
    mask_windows = tf.reshape(mask_windows, shape=(-1, self.window_size * self.window_size))
    attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
    attn_mask = tf.where(attn_mask != 0, -100., attn_mask)
    attn_mask = tf.where(attn_mask == 0, 0., attn_mask)

    return tf.Variable(attn_mask, trainable=False)

  def call(self, input, input_size):
    height, width = input_size
    _, num_patch_before, channels = input.shape

    attn_output = self.norm_layer1(input)
    attn_output = tf.reshape(attn_output, shape=(-1, height, width, channels))

    if self.shift_size > 0:
      attn_output = tf.roll(attn_output, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])

    attn_output = window_partition(attn_output, self.window_size)
    attn_output = tf.reshape(attn_output, shape=(-1, self.window_size * self.window_size, channels))

    if self.input_resolution == input_size:
      attn_output = self.attn_layer(attn_output, self.input_attn_mask)
    else:
      attn_output = self.attn_layer(attn_output, self.calculate_mask(input_size))

    attn_output = tf.reshape(attn_output, shape=(-1, self.window_size, self.window_size, channels))
    attn_output = window_reverse(attn_output, self.window_size, height, width, channels)

    if self.shift_size > 0:
      attn_output = tf.roll(attn_output, shift=[self.shift_size, self.shift_size], axis=[1, 2])

    attn_output = tf.reshape(attn_output, shape=(-1, height * width, channels))
    attn_output = self.dropout(attn_output)
    attn_output = input + attn_output

    output = self.norm_layer2(attn_output)
    output = self.mlp(output)
    output = self.dropout(output)
    output = attn_output + output

    return output

class PatchMerging(tf.keras.layers.Layer):
  def __init__(self, input_resolution, emb_size):
    super().__init__()
    self.input_resolution = input_resolution
    self.emb_size = emb_size
    self.reduction = tf.keras.layers.Dense(emb_size*2, use_bias=False)
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, input):
    height, width = self.input_resolution
    _, num_patch_before, channels = input.shape
    assert num_patch_before == height * width, "input feature has wrong size"
    assert height % 2 == 0 and width % 2 == 0

    input = tf.reshape(input, shape=(-1, height, width, channels))

    input0 = input[:, 0::2, 0::2, :]
    input1 = input[:, 1::2, 0::2, :]
    input2 = input[:, 0::2, 1::2, :]
    input3 = input[:, 1::2, 1::2, :]
    input = tf.concat((input0, input1, input2, input3), axis=-1)
    input = tf.reshape(input, shape=(-1, (height //2 ) * (width // 2), 4*channels))
    
    output = self.layer_norm(input)
    output = self.reduction(output)

    return output

class BasicLayer(tf.keras.Model):
  def __init__(self, emb_size, input_resolution, depth, num_heads, window_size, dff, dropout, downsample=None):
    super().__init__()
    self.emb_size = emb_size
    self.input_resolution = input_resolution
    self.depth = depth

    self.block = [SwinTransformerBlock(window_size, input_resolution, dff, emb_size, 0 if(i%2==0) else window_size // 2, num_heads, dropout) for i in range(depth)]
    
    if downsample is not None:
      pass
    else:
      self.downsample = None

  def call(self, input, input_size):
    for blk in self.block:
      input = blk(input, input_size)

    return input

class PatchEmbed(tf.keras.layers.Layer):
  def __init__(self, img_size, patch_size, in_chans, emb_size, norm_layer=None):
    super().__init__()
    #img_size = (img_size, img_size)
    patch_size = (patch_size, patch_size)

    #수정 부분
    height_patches_resolution = img_size[0] // patch_size[0]
    width_patches_resolution = img_size[1] // patch_size[1]
    patches_resolution = [height_patches_resolution, width_patches_resolution]

    #patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
    self.img_size = img_size
    self.patch_size = patch_size
    self.patches_resolution = patches_resolution
    self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

    self.in_chans = in_chans
    self.emb_size = emb_size

    if norm_layer is not None:
      self.norm = norm_layer
    else:
      self.norm = None

  def call(self, input):
    batch_size, _, _, channels = input.shape
    input = tf.reshape(input, shape=(batch_size, -1, channels))
    if self.norm is not None:
      input = self.norm(input)
    return input


class PatchUnEmbed(tf.keras.layers.Layer):
  def __init__(self, img_size, patch_size, in_chans, emb_size):
    super().__init__()
    #img_size = (img_size, img_size)
    patch_size = (patch_size, patch_size)

    #수정 부분
    height_patches_resolution = img_size[0] // patch_size[0]
    width_patches_resolution = img_size[1] // patch_size[1]
    patches_resolution = [height_patches_resolution, width_patches_resolution]

    #patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
    self.img_size = img_size
    self.patch_size = patch_size
    self.patches_resolution = patches_resolution
    self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

    self.in_chans = in_chans
    self.emb_size = emb_size

  def call(self, input, input_size):
    _, HW, channels = input.shape
    input = tf.reshape(input, shape=(-1, input_size[0], input_size[1], channels))
    return input

class RSTB(tf.keras.Model):
  def __init__(self, emb_size, input_resolution, depth, num_heads, window_size, dff, dropout, downsample=None, img_size=224, patch_size=4):
    super().__init__()
    self.emb_size = emb_size
    self.input_resolution = input_resolution
    self.residual_group = BasicLayer(emb_size, input_resolution, depth, num_heads, window_size, dff, dropout, downsample)

    self.conv = tf.keras.Sequential([
                                     tf.keras.layers.Conv2D(emb_size//4, 3, 1, padding='same', activation=tf.nn.leaky_relu),
                                     tf.keras.layers.Conv2D(emb_size//4, 1, 1, padding='same', activation=tf.nn.leaky_relu),
                                     tf.keras.layers.Conv2D(emb_size, 3, 1, padding='same')                             
    ])
    self.patch_emb = PatchEmbed(img_size, patch_size, 0, emb_size, norm_layer=None)
    self.patch_unemb = PatchUnEmbed(img_size, patch_size, 0, emb_size)

  def call(self, input, input_size):
    output = self.residual_group(input, input_size)
    output = self.patch_unemb(output, input_size)
    output = self.conv(output)
    output = self.patch_emb(output) + input
    return output

class UpSample(tf.keras.layers.Layer):
  def __init__(self, scale, num_feat):
    super().__init__()
    self.m = []
    self.scale = scale
    self.num_feat = num_feat
    self.scale_check = False
    if (scale & (scale-1) == 0):
      self.scale_check = True
      for _ in range(int(math.log(scale, 2))):
        self.m.append(tf.keras.layers.Conv2D(4 * num_feat, 3, 1, padding='same'))
    elif scale == 3:
      self.scale_check = False
      self.m.append(tf.keras.layers.Conv2D(9 * num_feat, 3, 1, padding='same'))
    else:
      raise ValueError(f"scale {scale} is not supported. scale: 2^n and 3.")
  
  def call(self, input):
    for con in self.m:
      input = con(input)
      if self.scale_check == True:
        input = tf.nn.depth_to_space(input, 2)
      elif self.scale_check == False:
        input = tf.nn.depth_to_space(input, 3)
      
    return input

class SwinIR(tf.keras.Model):
  def __init__(self, img_size, patch_size, in_chans, emb_size, depths, num_heads, window_size, dff, dropout, ape, upscale, img_range):
    super().__init__()
    num_in_ch = in_chans
    num_out_ch = in_chans
    num_feat = 64
    self.img_range = img_range
    if in_chans==3:
      rgb_mean = (0.4488, 0.4371, 0.4040)
      rgb_mean = tf.convert_to_tensor(rgb_mean)
      self.mean = tf.reshape(rgb_mean, shape=(1, 1, 1, 3))
    else:
      self.mean = tf.zeros(shape=(1, 1, 1, 1))
    self.upscale = upscale
    self.window_size = window_size


    self.conv_first = tf.keras.layers.Conv2D(emb_size, 3, 1, padding='same')


    self.num_layers = len(depths)
    self.emb_size = emb_size
    self.ape = ape
    self.num_features = emb_size
    self.dff = dff
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, emb_size, norm_layer=self.layer_norm)
    num_patches = self.patch_embed.num_patches
    patches_resolution = self.patch_embed.patches_resolution
    self.patches_resolution = patches_resolution

    self.patch_unembed = PatchUnEmbed(img_size, patch_size, in_chans, emb_size)
    
    if self.ape == True:
      self.absolute_pos_embed = tf.Variable(tf.zeros(shape=(1, num_patches, emb_size)), trainable=True)

    self.dropout = tf.keras.layers.Dropout(dropout)
    
    self.layer = [RSTB(emb_size, (patches_resolution[0], patches_resolution[1]), depths[i], num_heads[i], window_size, dff, dropout, downsample=None,
                       img_size=img_size, patch_size=patch_size) for i in range(self.num_layers)]
    
    self.conv_after_body = tf.keras.Sequential([
                                                tf.keras.layers.Conv2D(emb_size // 4, 3, 1, padding='same', activation=tf.nn.leaky_relu),
                                                tf.keras.layers.Conv2D(emb_size // 4, 1, 1, padding='same', activation=tf.nn.leaky_relu),
                                                tf.keras.layers.Conv2D(emb_size, 3, 1, padding='same')
    ])

    self.conv_before_upsample = tf.keras.layers.Conv2D(num_feat, 3, 1, padding='same')
    self.upsample = UpSample(upscale, num_feat)
    self.conv_last = tf.keras.layers.Conv2D(in_chans, 3, 1, padding='same')

  def call_features(self, x):
    x_size = (x.shape[1], x.shape[2])
    x = self.patch_embed(x)
    if self.ape == True:
      x = x + self.absolute_pos_embed

    x = self.dropout(x)

    for layer in self.layer:
      x = layer(x, x_size)

    x = self.layer_norm(x)
    x = self.patch_unembed(x, x_size)

    return x

  def call(self, x):
    height, width = x.shape[1], x.shape[2]
    
    self.mean = tf.cast(self.mean, dtype=x.dtype)
    x = (x - self.mean) * self.img_range

    x = self.conv_first(x)
    x = self.conv_after_body(self.call_features(x)) + x
    x = self.conv_before_upsample(x)
    x = self.conv_last(self.upsample(x))

    x = x / self.img_range + self.mean
    
    return x


"""
model = SwinIR(img_size=(HEIGHT, WIDTH), patch_size=PATCH_SIZE, in_chans=IN_CHANS, emb_size=EMB_SIZE, depths=DEPTHS, num_heads=NUM_HEADS, window_size=WINDOW_SIZE, dff=DFF, dropout=DROPOUT, ape=APE, upscale=UPSCALE, img_range=IMG_RANGE)
x = tf.random.normal(shape=(1, 128, 128, 3))
output = model(x)
output.shape
"""

