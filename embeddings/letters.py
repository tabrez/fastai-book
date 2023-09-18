#%% import packages
# import spacy
import string
from collections import Counter

#%% prepare some raw text with only lower case letters, digits and underscore
with open('rod_steiger.txt', 'r') as file:
  text = file.read()

text = text.lower()
# print(text)
table = str.maketrans('\n\t -ˈ—éəɪɡ–', '___________', string.punctuation)
text = text.translate(table)
# print(text)

print(f'len of text: {len(text)}, text[:20]: {text[:20]}')

#%% create the dataset - version 1
vocab = sorted(list(set(text)))
# vocab = sorted(set(text))
print(f'len of vocab: {len(vocab)}\n\nvocab: {vocab}\n\ncount: {Counter(text)}')

ds1 = []
# 0th element is not present in the input and last element is not present in the output
# alternatively add a dummy element at the start and the end of the text
for i in range(len(text)-1):
  pair = (text[i], text[i+1])
  ds1.append(pair)

print(f'len of ds1: {len(ds1)}, firt few: {ds1[:5]}')
#%% create the dataset - version 2
ds2 = { 'input': [], 'output': []}
for i in range(len(text)-1):
  ds2['input'].append(text[i])
  ds2['output'].append(text[i+1])

print(f'len of input: {len(ds2["input"])}, len of output: {len(ds2["output"])}')

#%% create the dataset with numericalisation - version 3
vocab = sorted(list(set(text)))
print(f'len of vocab: {len(vocab)}\n\nvocab: {vocab}\n\ncount: {Counter(text)}')

nums = []
for i in range(len(text)):
  index = vocab.index(text[i])
  nums.append(index)

ds = { 'input': [], 'output': []}
for i in range(len(nums)-6):
  ds['input'].append(nums[i:i+5])
  ds['output'].append(nums[i+5])

print(f'len of input: {len(ds["input"])}, len of output: {len(ds["output"])}')
print(f'first few: {ds["input"][:5], ds["output"][:5]}')

#%% train using pytorch
from torch import nn
import torch.nn.functional as F

class TrainEmbeddings(nn.Module):
  def __init__(self, vocab_size, embed_size, input_size, hidden_nodes):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size, embed_size)
    self.layer1 = nn.Linear(embed_size * input_size, hidden_nodes)
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
    # if self.debug: print(f'res 2: {res}')
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
      current = (batch + 1) * len(X)
      print(f'loss: {loss.item()}, {current}/{len(dl.dataset)}')

#%%
# 1. Create dataset in a shape that works with pytorch trainer
import torch
from torch.utils.data import TensorDataset, DataLoader
# .unsqueeze(1)?
tds = TensorDataset(torch.tensor(ds['input']), torch.tensor(ds['output']))
# 2. Create a dataloader from the above dataset
dl = DataLoader(tds, batch_size=256)
# 3. Create a learner/model object that can be trained on the above dataloader
model = TrainEmbeddings(len(vocab), 2, 5, 30)
# 4. Create a suitable optimizer & a loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5)

epochs = 250
for i in range(epochs):
  print(f'Epoch {i}=======================')
  train(dl, model, loss_fn, optimizer)
print('Training done.')

#%%
import pandas as pd
import altair as alt
embeddings = list(model.parameters())[0]
print(f'len of embeddings: {len(embeddings)}, embeddings: {embeddings[:5]}')

df = pd.DataFrame(embeddings.detach().numpy(), columns=['x', 'y'])
df['label'] = vocab

scatter = alt.Chart(df).mark_circle(size=60).encode(
  x='x',
  y='y'
).interactive()

scatter.mark_text(align='left', baseline='middle', dx=7).encode(
  text='label'
)

#%% How nn.CrossEntropyLoss() works
## https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
import torch
loss_fn = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
loss = loss_fn(input, target)
print(f'input: {input}, \ntarget: {target}, \nloss: {loss}')

#%% How TensorDataset works
print(f'len of ds.input: {len(ds["input"])}, len of ds.output: {len(ds["output"])}')
tds = TensorDataset(torch.tensor(ds['input']), torch.tensor(ds['output']))
print(f'len tds: {len(tds)}, tds[0:3]: {tds[0:3]}')
print(f'{torch.tensor([1,2,3])}')

#%% flatten a tensor list

t = torch.rand([1, 5, 2])
print(f't.shape: {t.shape}, t: {t}')
print(f't.unstack(): {t.view(-1)}')
