from google.colab import drive
drive.mount('/content/drive')

!pip install sentencepiece
!pip install wget

import tensorflow as tf
import numpy as np
import sentencepiece as spm
import tqdm
import os
import math
import json
import matplotlib.pyplot as plt
import wget

from random import random, shuffle, choice, randrange
from tqdm import tqdm_notebook, tqdm, trange

# vocab loading
vocab_file = "/content/drive/MyDrive/ColabNotebooks/kowikidata/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

vocab.vocab_size()

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
    batch_size = inputs.shape[0]

    q = self.q_linear(inputs)
    k = self.k_linear(inputs)
    v = self.v_linear(inputs)

    q = tf.reshape(q, shape=(batch_size, -1, self.num_heads, self.depth))
    q = tf.transpose(q, perm=[0, 2, 1, 3])

    k = tf.reshape(k, shape=(batch_size, -1, self.num_heads, self.depth))
    k = tf.transpose(k, perm=[0, 2, 1, 3])

    v = tf.reshape(v, shape=(batch_size, -1, self.num_heads, self.depth))
    v = tf.transpose(v, perm=[0, 2, 1, 3])

    logits = tf.matmul(q, k, transpose_b=True)

    if mask is not None:
      logits += (mask*-1e9)
    attn_mat = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attn_mat, v)
    output = tf.transpose(output, perm=[0, 2, 1, 3])

    output = tf.reshape(output, shape=(batch_size, -1, self.d_model))

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

class Encoder(tf.keras.Model):
  def __init__(self, max_len, seg_type, vocab_size, num_layers, dff, d_model, num_heads, dropout):
    super().__init__()
    self.word_emb = tf.keras.layers.Embedding(vocab_size, d_model)
    self.seg_emb = tf.keras.layers.Embedding(seg_type, d_model)
    self.pos_emb = tf.keras.layers.Embedding(max_len + 1, d_model)

    self.encoder_layers = [EncoderLayer(dff, d_model, num_heads, dropout) for i in range(num_layers)]
  
  def create_attn_mask(self, inputs, value):
    inputs = tf.cast(tf.math.equal(inputs, value), tf.float32)
    return inputs[:, tf.newaxis, tf.newaxis, :]

  def call(self, inputs, segments):
    batch_size = inputs.shape[0]
    pos = np.arange(inputs.shape[1]) + 1
    pos = np.broadcast_to(pos, shape=(batch_size, inputs.shape[1]))
    pos_mask = tf.math.equal(inputs, 0)
    pos = tf.where(~pos_mask, x=pos, y=0) # '~': ??? ?????????

    outputs = self.word_emb(inputs) + self.seg_emb(segments) + self.pos_emb(pos)

    attn_mask = self.create_attn_mask(inputs, 0)

    attn_mat_list = []
    for layer in self.encoder_layers:
      outputs, attn_mat = layer(outputs, attn_mask)
      attn_mat_list.append(attn_mat)

    return outputs, attn_mat_list

class BERT(tf.keras.Model):
  def __init__(self, max_len, seg_type, vocab_size, num_layers, dff, d_model, num_heads, dropout):
    super().__init__()
    self.encoder = Encoder(max_len, seg_type, vocab_size, num_layers, dff, d_model, num_heads, dropout)
    self.linear = tf.keras.layers.Dense(d_model)

  def call(self, inputs, segments):
    outputs, attn_mat_list = self.encoder(inputs, segments)
    outputs_cls = outputs[:, 0]
    outputs_cls = self.linear(outputs_cls)
    outputs_cls = tf.nn.gelu(outputs_cls)
    
    return outputs, outputs_cls, attn_mat_list

  def save(self, path):
    self.save_weights(path, overwrite=True)
  

  def load(self, path):
    self.load_weights(path)

class BERTPretrain(tf.keras.Model):
  def __init__(self, num_classes, max_len, seg_type, vocab_size, num_layers, dff, d_model, num_heads, dropout):
    super().__init__()
    self.bert = BERT(max_len, seg_type, vocab_size, num_layers, dff, d_model, num_heads, dropout)
    self.linear_cls = tf.keras.layers.Dense(num_classes, use_bias=False)
    self.linear_lm = tf.keras.layers.Dense(vocab_size, use_bias=False)

  def call(self, inputs, segments):
    outputs, outputs_cls, attn_mat_list = self.bert(inputs, segments)
    logits_cls = self.linear_cls(outputs_cls)
    logits_lm = self.linear_lm(outputs)

    return logits_lm, logits_cls, attn_mat_list

def create_pretrain_mask(tokens, mask_cnt, vocab_list):
    cand_idx = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
            cand_idx[-1].append(i)
        else:
            cand_idx.append([i])
    shuffle(cand_idx)

    mask_lms = []
    for index_set in cand_idx:
        if len(mask_lms) >= mask_cnt:
            break
        if len(mask_lms) + len(index_set) > mask_cnt:
            continue
        for index in index_set:
            masked_token = None
            if random() < 0.8: # 20% replace with [MASK]
                masked_token = "[MASK]"
            else:
                if random() < 0.5: # 10% keep original
                    masked_token = tokens[index]
                else: # 10% random word
                    masked_token = choice(vocab_list)
            mask_lms.append({"index": index, "label": tokens[index]})
            tokens[index] = masked_token
    mask_lms = sorted(mask_lms, key=lambda x: x["index"])
    mask_idx = [p["index"] for p in mask_lms]
    mask_label = [p["label"] for p in mask_lms]

    return tokens, mask_idx, mask_label

def trim_tokens(tokens_a, tokens_b, max_len):
  while(True):
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_len:
      break
    if len(tokens_a) > len(tokens_b):
      del tokens_a[0] #a??? ?????? ???????????? ??????
    else:
      tokens_b.pop() #b??? ?????? ???????????? ??????

def create_pretrain_instances(docs, doc_idx, doc, n_seq, mask_prob, vocab_list):
    # for CLS], [SEP], [SEP]
    max_seq = n_seq - 3
    tgt_seq = max_seq
    
    instances = []
    current_chunk = []
    current_length = 0
    for i in range(len(doc)):
        current_chunk.append(doc[i]) # line
        current_length += len(doc[i])
        if i == len(doc) - 1 or current_length >= tgt_seq:
            if 0 < len(current_chunk):
                a_end = 1
                if 1 < len(current_chunk):
                    a_end = randrange(1, len(current_chunk))
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                
                tokens_b = []
                if len(current_chunk) == 1 or random() < 0.5:
                    is_next = 0
                    tokens_b_len = tgt_seq - len(tokens_a)
                    random_doc_idx = doc_idx
                    while doc_idx == random_doc_idx:
                        random_doc_idx = randrange(0, len(docs))
                    random_doc = docs[random_doc_idx]

                    random_start = randrange(0, len(random_doc))
                    for j in range(random_start, len(random_doc)):
                        tokens_b.extend(random_doc[j])
                else:
                    is_next = 1
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                trim_tokens(tokens_a, tokens_b, max_seq)
                assert 0 < len(tokens_a)
                assert 0 < len(tokens_b)

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                segment = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                tokens, mask_idx, mask_label = create_pretrain_mask(tokens, int((len(tokens) - 3) * mask_prob), vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment": segment,
                    "is_next": is_next,
                    "mask_idx": mask_idx,
                    "mask_label": mask_label
                }
                instances.append(instance)

            current_chunk = []
            current_length = 0
    return instances

def make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob):
    vocab_list = []
    for id in range(vocab.get_piece_size()):
        if not vocab.is_unknown(id):
            vocab_list.append(vocab.id_to_piece(id))

    line_cnt = 0
    with open(in_file, "r") as in_f:
        for line in in_f:
            line_cnt += 1
    
    docs = []
    with open(in_file, "r") as f:
        doc = []
        with tqdm_notebook(total=line_cnt, desc=f"Loading") as pbar:
            for i, line in enumerate(f):
                line = line.strip()
                if line == "":
                    if 0 < len(doc):
                        docs.append(doc)
                        doc = []
                else:
                    pieces = vocab.encode_as_pieces(line)
                    if 0 < len(pieces):
                        doc.append(pieces)
                pbar.update(1)
        if doc:
            docs.append(doc)

    for index in range(count):
        output = out_file.format(index)
        if os.path.isfile(output): continue

        with open(output, "w") as out_f:
            with tqdm_notebook(total=len(docs), desc=f"Making") as pbar:
                for i, doc in enumerate(docs):
                    instances = create_pretrain_instances(docs, i, doc, n_seq, mask_prob, vocab_list)
                    for instance in instances:
                        out_f.write(json.dumps(instance))
                        out_f.write("\n")
                    pbar.update(1)

in_file = "/content/drive/MyDrive/ColabNotebooks/project/translation/vocab/kowiki.txt"
out_file = "/content/drive/MyDrive/ColabNotebooks/models/BERT/kowiki_bert" + "_{}.json"
count = 1
n_seq = 384
mask_prob = 0.15

make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob)

class PretrainDataSet(dict):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels_cls = []
        self.labels_lm = []
        self.sentences = []
        self.segments = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit=" lines")):
                instance = json.loads(line)
                self.labels_cls.append(instance["is_next"])
                sentences = [vocab.piece_to_id(p) for p in instance["tokens"]]
                self.sentences.append(sentences)
                self.segments.append(instance["segment"])
                mask_idx = np.array(instance["mask_idx"], dtype=np.int)
                mask_label = np.array([vocab.piece_to_id(p) for p in instance["mask_label"]], dtype=np.int)
                label_lm = np.full(len(sentences), dtype=np.int, fill_value=-1)
                label_lm[mask_idx] = mask_label
                self.labels_lm.append(label_lm)
    
    def __len__(self):
        assert len(self.labels_cls) == len(self.labels_lm)
        assert len(self.labels_cls) == len(self.sentences)
        assert len(self.labels_cls) == len(self.segments)
        return len(self.labels_cls)
    
    def __getitem__(self, item):
        return (np.array(self.labels_cls[item]),
                np.array(self.labels_lm[item]),
                np.array(self.sentences[item]),
                np.array(self.segments[item]))

#Parameter
NUM_CLASSES = 2
MAX_LEN = 384
SEG_TYPE = 2
VOCAB_SIZE = vocab.vocab_size()
NUM_LAYERS = 8 #12
D_MODEL = 512 # 768
DFF = D_MODEL * 4
NUM_HEADS = 8 #12
DROPOUT = 0.1

dataset_raw = PretrainDataSet(vocab, "/content/drive/MyDrive/ColabNotebooks/models/BERT/kowiki_bert_0.json")

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])

tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)

LR = 5e-5
BATCH_SIZE_PER_REPLICA = 8
global_batch_size = (BATCH_SIZE_PER_REPLICA *
                     strategy.num_replicas_in_sync)
#global_batch_size = 1 * 8

def dataload_fn(index, global_batch_size):
  labels_cls_list = []
  labels_lm_list = []
  sentences_list = []
  segments_list = []

  for i in range(global_batch_size):
    labels_cls, labels_lm, sentences, segments = dataset_raw[index * global_batch_size + i]
    sentences = tf.reshape(sentences, shape=(-1, len(sentences)))
    labels_lm = tf.reshape(labels_lm, shape=(-1, len(labels_lm)))
    segments = tf.reshape(segments, shape=(-1, len(segments)))
    labels_cls = tf.reshape(labels_cls, shape=(-1, 1))
    labels_lm = tf.keras.preprocessing.sequence.pad_sequences(labels_lm, maxlen=MAX_LEN, padding='post', value=-1)
    sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=MAX_LEN, padding='post', value=0)
    segments = tf.keras.preprocessing.sequence.pad_sequences(segments, maxlen=MAX_LEN, padding='post', value=0)
    labels_cls_list.append(labels_cls)
    labels_lm_list.append(labels_lm)
    sentences_list.append(sentences)
    segments_list.append(segments)

    del labels_cls, labels_lm, sentences, segments

  labels_cls_list = np.array(labels_cls_list).reshape(global_batch_size, -1)
  labels_lm_list = np.array(labels_lm_list).reshape(global_batch_size, -1)
  sentences_list = np.array(sentences_list).reshape(global_batch_size, -1)
  segments_list = np.array(segments_list).reshape(global_batch_size, -1)

  return labels_lm_list, labels_cls_list, sentences_list, segments_list

def loss_function(labels_lm, labels_cls, logits_lm, logits_cls, sentences):
  mask = tf.math.not_equal(labels_lm, -1)
  mask_float = tf.cast(mask, dtype=tf.float32)
  labels_lm_custom = tf.where(mask, x=labels_lm, y=sentences)

  loss_fn_cls = tf.losses.SparseCategoricalCrossentropy(reduction='none')
  loss_fn_lm = tf.losses.SparseCategoricalCrossentropy(reduction='none')

  loss_cls = loss_fn_cls(labels_cls, logits_cls)
  loss_cls = tf.reshape(loss_cls, shape=(-1, 1))
  loss_lm = loss_fn_lm(labels_lm_custom, logits_lm)

  loss_lm = tf.multiply(loss_lm, mask_float)
  loss = loss_lm + loss_cls
  return loss

def train_step(inputs):
  labels_lm, labels_cls, sentences, segments = inputs
  with tf.GradientTape() as tape:
    logits_lm, logits_cls, attn_mat_list = model(sentences, segments, training=True)
    loss = loss_function(labels_lm, labels_cls, logits_lm, logits_cls, sentences)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss 

@tf.function
def distributed_train_step(inputs):
  per_replica_losses = strategy.run(train_step, args=(inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)



path = '/content/drive/MyDrive/ColabNotebooks/models/BERT/bert_TF_pretrained_model.h5'
length = len(dataset_raw) // global_batch_size


with strategy.scope():
  model = BERTPretrain(NUM_CLASSES, MAX_LEN, SEG_TYPE, VOCAB_SIZE, NUM_LAYERS, DFF, D_MODEL, NUM_HEADS, DROPOUT)
  optimizer = tf.keras.optimizers.Adam(learning_rate=LR, epsilon=1e-9)
  training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)

  sen = tf.convert_to_tensor(np.arange(8*384).reshape(8, 384))
  seg = tf.convert_to_tensor(np.zeros_like(sen))
  lm, cls, attn_mat_list = model(sen, seg)
  model.load_weights('/content/drive/MyDrive/ColabNotebooks/models/BERT/bert_TF_pretrain_model.h5')

  for epoch in range(1):
    total_loss = 0.0
    step_loss = 0.0
    num_batch = 0

    with tqdm(total=length, desc=f"Train_file_number({epoch})") as pbar:
      for input_len in range(length):
        labels_lm_list, labels_cls_list, sentences_list, segments_list = dataload_fn(input_len, global_batch_size)
        dataset = tf.data.Dataset.from_tensor_slices((
            labels_lm_list, labels_cls_list, sentences_list, segments_list
        ))
        dataset = dataset.cache()
        dataset = dataset.batch(global_batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = strategy.experimental_distribute_dataset(dataset)

        for x in dataset:
          step_loss = distributed_train_step(x)
          total_loss += tf.reduce_mean(step_loss)
          num_batch += 1

        pbar.update(1)

        if num_batch % 400 == 99:
          print(f"epoch: {epoch}, step: {num_batch}, loss:{total_loss / num_batch}")
          model.bert.save(path)
          model.save_weights('/content/drive/MyDrive/ColabNotebooks/models/BERT/bert_TF_pretrain_model.h5', overwrite=True)

        del labels_lm_list, labels_cls_list, sentences_list, segments_list

    print(f"epoch: {epoch}, loss: {total_loss / num_batch}")
    model.bert.save(path)
    model.save_weights('/content/drive/MyDrive/ColabNotebooks/models/BERT/bert_TF_pretrain_model.h5', overwrite=True) #????????? ??????????????? ????????? ??????

