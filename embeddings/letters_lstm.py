#%% import packages
import datetime
import re
import time
from collections import Counter

import altair as alt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


#%% prepare some raw text with only lower case letters, digits and underscore
def make_text(filename):
  with open(filename, 'r') as file:
    text = file.read()
  text = re.sub('[^a-z0-9 ]', '', text).lower()
  return text.replace(' ', '_')

def make_vocab(text):
  return sorted(list(set(text)))

def debug_vocab():
  text = make_text('rod_steiger.txt')
  print(f'len of text: {len(text)}, text[:20]: {text[:200]}')
  vocab = make_vocab(text)
  print(f'len of vocab: {len(vocab)}\n\nvocab: {vocab}\n\ncount: {Counter(text)}')

#%% create the dataset with numericalisation - version 3
def numericalise(vocab, text):
  nums = []
  for i in range(len(text)):
    index = vocab.index(text[i])
    nums.append(index)
  return nums

def nums_to_dataset(nums, seq_len):
  ds = { 'input': [], 'output': []}
  for i in range(len(nums) - seq_len - 1):
    ds['input'].append(nums[i:i+seq_len])
    ds['output'].append(nums[i+seq_len])
  return ds

def make_dataset(vocab, text, sequence_length):
  nums = numericalise(vocab, text)
  ds = nums_to_dataset(nums, sequence_length)
  print(f'len of input: {len(ds["input"])}, len of output: {len(ds["output"])}')
  print(f'first few: {ds["input"][:7], ds["output"][:7]}')
  return ds

#%% neural network using pytorch
class TrainEmbeddings(nn.Module):
  def __init__(self, vocab_size, embed_size, bs, hidden_nodes):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_nodes, batch_first=True)
    self.output = nn.Linear(hidden_nodes, vocab_size)
    self.hidden_state = torch.Tensor(1, bs, hidden_nodes)
    self.cell_state = torch.Tensor(1, bs, hidden_nodes)
    self.debug = False

  def forward(self, x):
    res = self.embeddings(x)
    if self.debug: print(f'res after embeddings layer: {res}')
    res, (h, c) = self.lstm(res, (self.hidden_state, self.cell_state))
    if self.debug: print(f'res after lstm layer: {res}')
    self.hidden_state = h.detach()
    self.cell_state = c.detach()
    res = self.output(res)
    if self.debug: print(f'res after output layer: {res}')
    self.debug = False
    return res.view(len(x), -1)

  def reset(self):
    self.hidden_state.zero_()
    self.cell_state.zero_()

#%% trainer
def train(dl, model, loss_fn, optimizer):
  # device = ("cuda" if torch.cuda.is_available() else "cpu")
  device = 'cpu'
  for batch, (X, y) in enumerate(dl):
    # TODO: why not move entire dataloader to 'device' at once before calling `train`
    X, y = X.to(device), y.to(device)

    preds = model(X)
    # print(f'preds.shape: {preds.shape}, len of preds: {len(preds)}')
    # print(f'y.shape: {y.shape}, len of y: {len(y)}')
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
      print(f'batch: {batch}/{len(dl)}, loss: {loss.item()}')
  return model, optimizer

#%% re-arrange dataset for sequential processing by LSTM
def rearrange_dataset(ds, bs):
  # print('len(ds), bs:', len(ds['input']), bs)
  # print('ds before:', ds['input'][:3])
  num_batches = int(len(ds['input'])/bs)
  print('number of batches,', num_batches)
  res = {'input': [], 'output': []}
  for i in range(num_batches):
    # print('first loop')
    for j in range(bs):
      # print('i, j', i, j)
      res['input'].append(ds['input'][i+(j*num_batches)])
      res['output'].append(ds['output'][i+(j*num_batches)])
  # print('ds after:', res['input'][:3])
  return res

#%% training loop
def train_loop(vocab, text, batch_size, embed_size, seq_len, hidden_nodes, lr, epochs):
  # device = ("cuda" if torch.cuda.is_available() else "cpu")
  device = 'cpu'

  # 1. Create a dataloader in a shape that works with pytorch trainer
  ds = make_dataset(vocab, text, seq_len)
  ds = rearrange_dataset(ds, batch_size)
  tensor_ds = TensorDataset(torch.tensor(ds['input']), torch.tensor(ds['output']))
  dl = DataLoader(tensor_ds, batch_size=batch_size)

  # 2. Create a learner/model object that can be trained on the above dataloader
  model = TrainEmbeddings(len(vocab),
                          embed_size=embed_size,
                          bs=batch_size,
                          hidden_nodes=hidden_nodes)
  model = model.to(device)

  # 3. Create a suitable optimizer & a loss function
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)

  # 4. training loop
  start = time.time()
  for i in range(epochs):
    print(f'Epoch {i}=======================')
    model, optimizer = train(dl, model, loss_fn, optimizer)
  print(f'Training took {str(datetime.timedelta(seconds=time.time() - start))}.')
  return plot_embeddings(model, vocab)
True
def plot_embeddings(model, vocab):
  embeddings = list(model.parameters())[0]
  print(f'len of embeddings: {len(embeddings)}, few embeddings: {embeddings[:6]}')

  df = pd.DataFrame(embeddings.cpu().detach().numpy(), columns=['x', 'y'])
  df['label'] = vocab

  scatter = alt.Chart(df).mark_circle().encode(
    x='x',
    y='y'
  ).interactive()

  return scatter.mark_text(align='left', baseline='middle', dx=7).encode(
    text='label'
  )

#%% driver
sequence_length=6
text = make_text('rod_steiger.txt')
vocab = make_vocab(text)
train_loop(vocab=vocab,
            text=text,
            batch_size=2056,
            embed_size=2,
            seq_len=sequence_length,
            hidden_nodes=5,
            lr=5,
            epochs=8)

# %%
