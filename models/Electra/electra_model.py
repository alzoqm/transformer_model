import tensorflow as tf
import numpy as np
import tqdm
import os
import math
import json
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

from random import random, shuffle, choice, randrange
from tqdm import tqdm_notebook, tqdm, trange

class MultiHeadAttention(tf.keras.Model):
  def __init__(self, d_model, num_heads):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads

    assert d_model % num_heads == 0
    self.depth = d_model // num_heads

    self.q_linear = tf.keras.layers.Dense(d_model)
    self.k_linear = tf.keras.layers.Dense(d_model)
    self.v_linear = tf.keras.layers.Dense(d_model)

    self.linear = tf.keras.layers.Dense(d_model)

    self.scale = 1 / (self.depth ** 0.5)

  def call(self, inputs, mask=None):
    B, L, D = inputs.shape

    q = self.q_linear(inputs)
    k = self.k_linear(inputs)
    v = self.v_linear(inputs)

    q = tf.reshape(q, shape=(-1, L, self.num_heads, self.depth))
    q = tf.transpose(q, perm=[0, 2, 1, 3])

    k = tf.reshape(k, shape=(-1, L, self.num_heads, self.depth))
    k = tf.transpose(k, perm=[0, 2, 1, 3])

    v = tf.reshape(v, shape=(-1, L, self.num_heads, self.depth))
    v = tf.transpose(v, perm=[0, 2, 1, 3])

    logits = tf.matmul(q, k, transpose_b=True)

    if mask is not None:
      logits += (mask*-1e9)
    attn_mat = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attn_mat, v)
    output = tf.transpose(output, perm=[0, 2, 1, 3])

    output = tf.reshape(output, shape=(-1, L, self.d_model))

    output = self.linear(output)

    return output, attn_mat

class MLP(tf.keras.Model):
  def __init__(self, dff, d_model, dropout):
    super().__init__()
    self.linear1 = tf.keras.layers.Dense(dff)
    self.linear2 = tf.keras.layers.Dense(d_model)

    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, inputs):
    outputs = self.linear1(inputs)
    outputs = tf.nn.gelu(outputs)
    outputs = self.dropout(outputs)
    outputs = self.linear2(outputs)

    return outputs

class EncoderLayer(tf.keras.Model):
  def __init__(self, dff, d_model, num_heads, dropout):
    super().__init__()
    self.attn_layer = MultiHeadAttention(d_model, num_heads)
    self.mlp_layer = MLP(dff, d_model, dropout)
    
    self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-9)
    self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-9)

    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, inputs, mask):
    attn_outputs, attn_mat = self.attn_layer(inputs, mask)
    attn_outputs = self.dropout(attn_outputs)
    attn_outputs = self.layer_norm1(inputs + attn_outputs)

    mlp_outputs = self.mlp_layer(attn_outputs)
    mlp_outputs = self.dropout(mlp_outputs)
    outputs = self.layer_norm2(attn_outputs + mlp_outputs)

    return outputs, attn_mat

class GeneratorEncoder(tf.keras.Model):
  def __init__(self, max_len, seg_type, vocab_size, num_layers, dff, d_model, emb_size, num_heads, dropout):
    super().__init__()
    # self.word_emb = tf.keras.layers.Embedding(vocab_size, emb_size)
    # self.seg_emb = tf.keras.layers.Embedding(seg_type, emb_size)
    # self.pos_emb = tf.keras.layers.Embedding(max_len + 1, emb_size) #Discriminator??? ????????? ???????????? ?????? -> ??? ?????? ????????? ?????????

    self.encoder_layers = [EncoderLayer(dff, d_model, num_heads, dropout) for i in range(num_layers)]
    self.emb_proj = tf.keras.layers.Dense(d_model)
  
  def create_attn_mask(self, inputs, value):
    inputs = tf.cast(tf.math.equal(inputs, value), tf.float32)
    return inputs[:, tf.newaxis, tf.newaxis, :]

  def call(self, inputs, segments, word_emb, seg_emb, pos_emb):
    batch_size = tf.shape(inputs)[0]
    pos = np.arange(inputs.shape[1]) + 1
    pos = tf.broadcast_to(pos, shape=(batch_size, inputs.shape[1]))
    #pos = np.broadcast_to(pos, shape=(batch_size, inputs.shape[1]))
    pos_mask = tf.math.equal(inputs, 0)
    pos = tf.where(~pos_mask, x=pos, y=0) 

    outputs = word_emb(inputs) + seg_emb(segments) + pos_emb(pos)
    outputs = self.emb_proj(outputs)

    attn_mask = self.create_attn_mask(inputs, 0)

    attn_mat_list = []
    for layer in self.encoder_layers:
      outputs, attn_mat = layer(outputs, attn_mask)
      attn_mat_list.append(attn_mat)

    return outputs, attn_mat_list

class DiscriminatorEncoder(tf.keras.Model):
  def __init__(self, max_len, seg_type, vocab_size, num_layers, dff, d_model, emb_size, num_heads, dropout):
    super().__init__()
    self.word_emb = tf.keras.layers.Embedding(vocab_size, emb_size)
    self.seg_emb = tf.keras.layers.Embedding(seg_type, emb_size)
    self.pos_emb = tf.keras.layers.Embedding(max_len + 1, emb_size)

    self.encoder_layers = [EncoderLayer(dff, d_model, num_heads, dropout) for i in range(num_layers)]
    self.emb_proj = tf.keras.layers.Dense(d_model)
  
  def create_attn_mask(self, inputs, value):
    inputs = tf.cast(tf.math.equal(inputs, value), tf.float32)
    return inputs[:, tf.newaxis, tf.newaxis, :]

  def call(self, inputs, segments):
    batch_size = tf.shape(inputs)[0]
    pos = np.arange(inputs.shape[1]) + 1
    pos = tf.broadcast_to(pos, shape=(batch_size, inputs.shape[1]))
    #pos = np.broadcast_to(pos, shape=(batch_size, inputs.shape[1]))
    pos_mask = tf.math.equal(inputs, 0)
    pos = tf.where(~pos_mask, x=pos, y=0) 

    outputs = self.word_emb(inputs) + self.seg_emb(segments) + self.pos_emb(pos)
    outputs = self.emb_proj(outputs)

    attn_mask = self.create_attn_mask(inputs, 0)

    attn_mat_list = []
    for layer in self.encoder_layers:
      outputs, attn_mat = layer(outputs, attn_mask)
      attn_mat_list.append(attn_mat)

    return outputs, attn_mat_list

class Discriminator(tf.keras.Model):
  def __init__(self, max_len, seg_type, vocab_size, num_layers, dff, d_model, emb_size, num_heads, dropout):
    super().__init__()
    self.encoder = DiscriminatorEncoder(max_len, seg_type, vocab_size, num_layers, dff, d_model, emb_size, num_heads, dropout)
    self.linear = tf.keras.layers.Dense(d_model)

  def call(self, inputs, segments):
    outputs, attn_mat_list = self.encoder(inputs, segments)
    outputs_cls = outputs[:, 0]
    outputs = self.linear(outputs)
    outputs = tf.nn.gelu(outputs)
    
    return outputs, outputs_cls, attn_mat_list

  def save(self, path):
    self.save_weights(path, overwrite=True)
  

  def load(self, path):
    self.load_weights(path)

class Generator(tf.keras.Model):
  def __init__(self, max_len, seg_type, vocab_size, num_layers, dff, d_model, emb_size, num_heads, dropout):
    super().__init__()
    self.encoder = GeneratorEncoder(max_len, seg_type, vocab_size, num_layers, dff, d_model, emb_size, num_heads, dropout)
    self.linear = tf.keras.layers.Dense(d_model)

  def call(self, inputs, segments,  word_emb, seg_emb, pos_emb):
    outputs, attn_mat_list = self.encoder(inputs, segments,  word_emb, seg_emb, pos_emb)
    outputs_cls = outputs[:, 0]
    outputs_cls = self.linear(outputs_cls)
    outputs_cls = tf.nn.gelu(outputs_cls)
    
    return outputs, outputs_cls, attn_mat_list

  def save(self, path):
    self.save_weights(path, overwrite=True)
  

  def load(self, path):
    self.load_weights(path)

class ElectraPretrain(tf.keras.Model):
  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    self.generator = Generator(config['max_len'], config['seg_type'], config['vocab_size'], config['gen_num_layers'], config['gen_dff'], config['gen_d_model'], config['gen_emb_size'],
                               config['gen_num_heads'], config['gen_dropout'])
    
    self.discriminator = Discriminator(config['max_len'], config['seg_type'], config['vocab_size'], config['dis_num_layers'], config['dis_dff'], config['dis_d_model'], config['dis_emb_size'],
                                       config['dis_num_heads'], config['dis_dropout'])
    
    self.word_emb = self.discriminator.encoder.word_emb
    self.seg_emb = self.discriminator.encoder.seg_emb
    self.pos_emb = self.discriminator.encoder.pos_emb
    
    self.gen_linear_cls = tf.keras.layers.Dense(config['num_classes'], use_bias=False)
    self.gen_linear_lm = tf.keras.layers.Dense(config['vocab_size'], use_bias=False)

    self.dis_linear = tf.keras.layers.Dense(2, activation='sigmoid') #??????????????? ????????? ????????? ????????? ????????? ??????
    self.dis_lambda = config['dis_lambda']

  def gen_loss_function(self, labels_lm, labels_cls, logits_lm, logits_cls, inputs):
    mask = tf.math.not_equal(labels_lm, -1)  # masking??? ?????? True
    mask_float = tf.cast(mask, dtype=tf.float32)
    labels_lm_custom = tf.where(mask, x=labels_lm, y=inputs)  #mask??? True??? ?????? lm??????, ?????? ?????? inputs -> -1??? ?????? ??????

    loss_cls = tf.losses.sparse_categorical_crossentropy(labels_cls, logits_cls)
    loss_cls = tf.reshape(loss_cls, shape=(-1, 1))
    loss_lm = tf.losses.sparse_categorical_crossentropy(labels_lm_custom, logits_lm)

    loss_lm = tf.multiply(loss_lm, mask_float)  #masking??? ????????? mask_float ?????? 1, ?????? ?????? 0??????.
    loss = loss_lm + loss_cls
    return loss, mask, labels_lm_custom  #mask??? labels_lm_custom??? discriminator??? input??? ????????? sampling ????????? ???????????? ??????

  def call(self, inputs):
    input, segments, labels_cls, labels_lm = inputs
    gen_outputs, gen_outputs_cls, gen_attn_mat_list = self.generator(input, segments, self.word_emb, self.seg_emb, self.pos_emb)
    gen_outputs_cls = self.gen_linear_cls(gen_outputs_cls)  #??? ?????? ????????? ????????? ????????? ?????? ???????????? ??????
    gen_outputs = self.gen_linear_lm(gen_outputs)

    gen_loss, mask, labels_lm_custom = self.gen_loss_function(labels_lm, labels_cls, gen_outputs, gen_outputs_cls, input)
    ##sampling
    #labels_lm_custom -> ?????? ????????? ??? ???????????????
    gen_outputs_softmax = tf.nn.softmax(gen_outputs, axis=2)
    sampling = tf.constant([], dtype=tf.int32)
    #for index, one_batch in enumerate(gen_outputs_softmax):
    for index in range(tf.shape(gen_outputs_softmax)[0]):
      """
      softmax ????????? ?????? sampling??? ??????, 2?????? ???????????? ????????? ?????? ?????????,
       for?????? ?????? batch size?????? ??????(??? ???????????? ?????? ???????????? ??? ???)
      """
      one_batch_sampling = tf.random.categorical(gen_outputs_softmax[index], 1, dtype=tf.int32)
      one_batch_sampling = tf.expand_dims(one_batch_sampling, axis=0)
      if index == 0: #-> predict??? ??? ?????? ??????(???????????????)
        sampling = one_batch_sampling
      else:
        sampling = tf.concat([sampling, one_batch_sampling], axis=0)
      #sampling.append(one_batch_sampling)
    #sampling = np.array(sampling)
    #sampling = tf.convert_to_tensor(sampling)
    sampling = tf.squeeze(sampling, axis=2)

    sampling = tf.where(mask, x=sampling, y = labels_lm_custom) # mask??? ????????? token??? sampling ????????? ??????
    """
    sampling ???????????? ????????? ????????? token?????? sampling??? ????????? mask??? ?????? ?????? True???(Real),
    ?????? ????????? sampling??? ????????? False(Fake)
    """
    same_token_mask = sampling == labels_lm_custom #same_token_mask??? discriminator??? label?????????.
    same_token_mask = tf.cast(same_token_mask, dtype=tf.float32)

    #input_shape -> (1, 512)
    dis_outputs, dis_outputs_cls, dis_attn_mat_list = self.discriminator(sampling, segments)
    dis_outputs = self.dis_linear(dis_outputs)

    dis_loss = tf.losses.sparse_categorical_crossentropy(same_token_mask, dis_outputs)
    total_loss = gen_loss + dis_loss*self.dis_lambda

    return total_loss, gen_loss, sampling, dis_loss

##?????? ?????? 2?????? ?????? ??????

if __name__ == '__main__':
  #model hyper parameter??? electra small++??? ???????????? ??????
  config = {}
  config['num_classes'] = 2
  config['max_len'] = 512
  config['seg_type'] = 2
  config['vocab_size'] = 12007
  config['gen_num_layers'] = 12
  config['gen_dff'] = 1024
  config['gen_d_model'] = 256
  config['gen_emb_size'] = 128
  config['gen_num_heads'] = 4
  config['gen_dropout'] = 0.1
  config['dis_num_layers'] = 12
  config['dis_dff'] = 1024
  config['dis_d_model'] = 256
  config['dis_emb_size'] = 128
  config['dis_num_heads'] = 4
  config['dis_dropout'] = 0.1
  config['dis_lambda'] = 50

  electra_pretrain = ElectraPretrain(config)

  x = tf.range(512*10)
  x = tf.reshape(x, shape=(-1, 512))
  test_batch_size = x.shape[0]
  seg = tf.ones_like(x)
  label_cls = tf.ones(shape=(test_batch_size, ))
  total_loss, gen_loss, sampling, dis_loss = electra_pretrain([x, seg, label_cls, x])
  print(gen_loss.shape)
  print(dis_loss.shape)


