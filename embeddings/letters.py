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
  text = re.sub('[^a-zA-Z0-9 ]', '', text).lower()
  return text.replace(' ', '_')

text = make_text('rod_steiger.txt')
print(f'len of text: {len(text)}, text[:20]: {text[:20]}')

vocab = sorted(list(set(text)))
print(f'len of vocab: {len(vocab)}\n\nvocab: {vocab}\n\ncount: {Counter(text)}')

#%% create the dataset with numericalisation - version 3
def numericalise(vocab, text):
  nums = []
  for i in range(len(text)):
    index = vocab.index(text[i])
    nums.append(index)
  return nums

def make_ds(nums, seq_len):
  ds = { 'input': [], 'output': []}
  for i in range(len(nums) - seq_len - 1):
    ds['input'].append(nums[i:i+seq_len])
    ds['output'].append(nums[i+seq_len])
  return ds

sequence_length = 5
nums = numericalise(vocab, text)
ds = make_ds(nums, sequence_length)
print(f'len of input: {len(ds["input"])}, len of output: {len(ds["output"])}')
print(f'first few: {ds["input"][:7], ds["output"][:7]}')

#%% train using pytorch
class TrainEmbeddings(nn.Module):
  def __init__(self, vocab_size, embed_size, seq_len, hidden_nodes):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size, embed_size)
    self.layer1 = nn.Linear(embed_size * seq_len, hidden_nodes)
    self.layer2 = nn.Linear(hidden_nodes, hidden_nodes)
    self.output = nn.Linear(hidden_nodes, vocab_size)
    self.debug = True

  def forward(self, x):
    # if self.debug: print(f'x: {x}')
    res = self.embeddings(x)
    # if self.debug: print(f'res 0: {res}')
    res = res.view(len(x), -1)
    # if self.debug: print(f'res 1: {res}')
    res = F.relu(self.layer1(res))
    res = F.relu(self.layer2(res))
    res = self.output(res)
    # if self.debug: print(f'res 3: {res}')
    self.debug = False
    return res.view(len(x), -1)

# device = ("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
def train(dl, model, loss_fn, optimizer):
  for batch, (X, y) in enumerate(dl):
    X, y = X.to(device), y.to(device)

    optimizer.zero_grad()
    preds = model(X)
    # print(f'preds.shape: {preds.shape}, len of preds: {len(preds)}')
    # print(f'y.shape: {y.shape}, len of y: {len(y)}')
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      print(f'batch: {batch}/{len(dl)}, loss: {loss.item()}')

#%%
# Create a dataloader in a shape that works with pytorch trainer
tds = TensorDataset(torch.tensor(ds['input']), torch.tensor(ds['output']))
dl = DataLoader(tds, batch_size=256)
# 3. Create a learner/model object that can be trained on the above dataloader
model = TrainEmbeddings(len(vocab),
                        embed_size=2,
                        seq_len=sequence_length,
                        hidden_nodes=30)
model = model.to(device)
# 4. Create a suitable optimizer & a loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)

epochs = 10
start = time.time()
for i in range(epochs):
  print(f'Epoch {i}=======================')
  train(dl, model, loss_fn, optimizer)
print(f'Training took {str(datetime.timedelta(seconds=time.time() - start))}.')

embeddings = list(model.parameters())[0]
print(f'len of embeddings: {len(embeddings)}, embeddings: {embeddings[:5]}')

df = pd.DataFrame(embeddings.cpu().detach().numpy(), columns=['x', 'y'])
df['label'] = vocab

scatter = alt.Chart(df).mark_circle().encode(
  x='x',
  y='y'
).interactive()

scatter.mark_text(align='left', baseline='middle', dx=7).encode(
  text='label'
)

# %%
