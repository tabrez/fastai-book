# %% Imports
import pydoc
import warnings
from fastai.text.all import *
from fastbook import *
import fastai
import fastbook

fastbook.setup_book()
print(f"fastai version: {fastai.__version__}")

warnings.filterwarnings(
  "ignore",
  message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.",
)
warnings.filterwarnings(
  "ignore",
  message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet34_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet34_Weights.DEFAULT` to get the most up-to-date weights.",
)

# BASE_DIR='/content/gdrive/MyDrive/Colab Notebooks'
BASE_DIR = "/home/tabrez/MEGA/MEGA/code/fastai-book"
#!ln -sf '/content/gdrive/MyDrive/Colab Notebooks' ~/.fastai

def source(fn):
    pydoc.pager(pydoc.render_doc(fn))

#%% Download data
## URLs.HUMAN_NUMBERS contains numbers spelled out as words:
## One, Two, Three...One hundred and thirteen, etc.
path = untar_data(URLs.HUMAN_NUMBERS)
print(f'path: {path}, len(path.ls()): {len(path.ls())}, files: {path.ls()}')
## Read all the lines from the train.txt and valid.txt files at `path`
lines = L()
with open(path/'train.txt') as f:
  lines += L(*f.readlines())
with open(path/'valid.txt') as f:
  lines += L(*f.readlines())
print(f'len(lines): {len(lines)}, first few lines: {lines[:5]}')
## insert ' . ' between each word
text = ' . '.join([l.strip() for l in lines])
print(f'len(text): {len(text)}, first few words: {text[:55]}')
## tokenize
tokens = text.split(' ')
print(f'len(tokens): {len(tokens)}, first few tokens: {tokens[:15]}')
## create a vocab
vocab = L(*tokens).unique()
print(f'len(vocab): {len(vocab)}, first few vocab tokens: {vocab[:15]}')
## create a dict that can be used to map a token/word to its index in the `vocab`
word2idx = {w:i for i,w in enumerate(vocab)}
print(f'type(word2idx): {type(word2idx)}, len(word2idx): {len(word2idx.keys())}, one item: {word2idx["three"]}')
## numericalise: convert the list of tokens to list of their corresponding index from `vocab`
nums = L(word2idx[token] for token in tokens)
print(f'type(nums): {type(nums)}, len(nums): {len(nums)}, first few nums: {nums[:13]}')

#%% create data loader `dls`
## use the sliding method to create a dataset:
## pick 3 indexes as input, pick 4th index as output/label, slide right by 4 numbers and repeat
# L((tokens[i:i+3], tokens[i+3]) for i in range(0, len(tokens)-4, 1))
# L((tokens[i:i+3], tokens[i+3]) for i in range(0, len(tokens)-4, 3))
seqs = L((tensor(nums[i:i+3]), nums[i+3]) for i in range(0, len(nums)-4, 3))
print(f'type(seqs): {type(seqs)}, len(seqs): {len(seqs)}, first few seqs: {seqs[:13]}')
## split the first 80% of `seqs` as training dataset and remaining 20% as validation dataset
train_sz = int(len(seqs) * 0.8) # training set size
dls = DataLoaders.from_dsets(seqs[:train_sz], seqs[train_sz:], bs=64, shuffle=False)

#%% inspect dls
def inspect_dls(dls):
  print(f'type(items): {type(dls.items)}, len(items): {len(dls.items)}')
  x, y = dls.one_batch()
  print(f'first batch -> x.shape: {x.shape}')
  print(f'first batch -> y.shape: {y.shape}')
  print(f'''\nlen(train): {len(dls.train)},\
        \nlen(valid): {len(dls.valid)},\
        \nlen(train_ds): {len(dls.train_ds)},\
        \nlen(valid_ds): {len(dls.valid_ds)}''')

  x, y = dls.train.one_batch()
  print(f'''\ntrain -> first batch -> x.shape: {x.shape},\
        \ntrain -> first batch -> y.shape: {y.shape}''')

inspect_dls(dls)

#%% RNN Language Model in PyTorch
class LMModel1(Module):
  def __init__(self, vocab_sz, n_hidden):
    self.i_h = nn.Embedding(vocab_sz, n_hidden)
    self.h_h = nn.Linear(n_hidden, n_hidden)
    self.h_o = nn.Linear(n_hidden, vocab_sz)
    self.debug = True

  def forward(self, batch):
    h = 0
    mapped = self.i_h(batch[:,0])
    if self.debug:
      print(f'self.h_h: {self.h_h}')
      print(f'shape of batch: {batch.shape}')
      print(f'shape of mapped: {mapped.shape}')
      print(f'first item in mapped: {mapped[0]}')
    ## map from indexes of tokens in `batch` to their corresponding embeddings in self.i.h
    ## `self.i_h(batch)` fetches one embedding item of length `n_hidden` per item in `batch`
    h = F.relu(self.h_h(h + self.i_h(batch[:,0])))
    if self.debug:
      self.debug = False
      print(f'self.h: {h.shape}')
    h = F.relu(self.h_h(h + self.i_h(batch[:,1])))
    h = F.relu(self.h_h(h + self.i_h(batch[:,2])))
    return self.h_o(h)

#%% train the model
learn = Learner(dls, model=LMModel1(len(vocab), 64), loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(4, 1e-3)

#%% a basic model that always returns the most common token as the prediction
len(nums), nums[:3], type(nums), type(list(nums))
a = [1, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 3, 995, 1]
from collections import Counter
count = Counter(nums)
print(count.most_common(1))
print(len(vocab))

#%% Converting above model to RNN using for loop
class LMModel2(Module):
  def __init__(self, vocab_sz, n_hidden):
    self.i_h = nn.Embedding(vocab_sz, n_hidden)
    self.h_h = nn.Linear(n_hidden, n_hidden)
    self.h_o = nn.Linear(n_hidden, vocab_sz)

  def forward(self, batch):
    h = 0
    for i in range(3):
      h = F.relu(self.h_h(h + self.i_h(batch[:,i])))
    return self.h_o(h)

#%% fine tune
learn = Learner(dls, model=LMModel2(len(vocab), 64), loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(4, 1e-3)

#%% Remember the hidden state for longer
class LMModel3(Module):
  def __init__(self, vocab_sz, n_hidden):
    self.i_h = nn.Embedding(vocab_sz, n_hidden)
    self.h_h = nn.Linear(n_hidden, n_hidden)
    self.h_o = nn.Linear(n_hidden, vocab_sz)
    self.h = 0

  def forward(self, batch):
    for i in range(3):
      self.h = F.relu(self.h_h(self.h + self.i_h(batch[:,i])))
    out = self.h_o(self.h)
    self.h = self.h.detach()
    return out

  def reset(self): self.h = 0

#%% fine tune
# learn = Learner(dls, model=LMModel3(len(vocab), 64), loss_func=F.cross_entropy, metrics=accuracy)
# learn.fit_one_cycle(4, 1e-3)

#%% re-arrange dataset for nlp tasks
## First few samples of the first batch of data could be [("This", " . ", "movie"),  " . "],
## [(" . ", "movie", " . "), "is"], [("movie", " . ", "is"), "not"], etc.
## We want [("This", " . ", "movie"),  " . "] to be the first sample of first batch and
## [(" . ", "movie", " . "), "is"] to be the first sample of second batch.

## example using synthetic data
# seqqs = range(43)
# bs = 5
# num_batches = int(len(seqqs)/bs) # 8

# seqs2 = []
# for i in range(num_batches):
#   for j in range(bs):
#     seqs2.append(seqqs[i + j*num_batches])
    # print(seqqs[i + j*num_batches])
  # print()
# limit = num_batches*bs
# for i in range(limit, len(seqqs)):
#   seqs2.append(seqqs[i])
# seqs2
## remaining data from last, incomplete batch is discarded

##% implement above functionality on seqs

def make_dataset(seqs, bs):
  num_batches = int(len(seqs)/bs)
  res = []
  for i in range(num_batches):
    for j in range(bs):
      res.append(seqs[i + j*num_batches])
  return res

bs = 64
train_len = int(len(seqs) * 0.8)
seqs_train = make_dataset(seqs[:train_len], bs)
seqs_valid = make_dataset(seqs[train_len:], bs)

# print(f'type(seqs): {type(seqs)}, len(seqs): {len(seqs)}, first few: {seqs[:3,:4]}')
# print(f'type(seqs_train): {type(seqs_train)}, len(seqs_train): {len(seqs_train)},\
#       first: {seqs_train[0]}, 64th: {seqs_train[64]}, 128th: {seqs_train[128]}')

# def loss_func(preds, targets):
#   print(f'preds.shape: {preds.shape}, targets.shape: {targets.shape}')
#   return F.cross_entropy(preds, targets)

dls3 = DataLoaders.from_dsets(seqs_train, seqs_valid, bs=bs, drop_last=True, shuffle=False)
learn = Learner(dls3, LMModel3(len(vocab), 64), loss_func=F.cross_entropy, metrics=accuracy,
                cbs=ModelResetter)
learn.fit_one_cycle(10, 3e-3)

#%% Creating more signal
class LMModel4(Module):
  def __init__(self, vocab_sz, n_hidden):
    self.i_h = nn.Embedding(vocab_sz, n_hidden)
    self.h_h = nn.Linear(n_hidden, n_hidden)
    self.h_o = nn.Linear(n_hidden, vocab_sz)
    self.h = 0

  def forward(self, batch):
    out = []
    for i in range(16):
      self.h = F.relu(self.h_h(self.h + self.i_h(batch[:,i])))
      out.append(self.h_o(self.h))
    self.h = self.h.detach()
    ## 1. `out` will be a list of length `seq_len`, each item with shape `n_hidden` x `vocab_sz`
    ##    `y` will be `n_hidden` x `seq_len`
    ## 2. reshape `out` to be `n_hidden` x `seq_len` x `vocab_sz` shaped tensor before returning it
    ## 3. `seq_len` = 16, `n_hidden` = 64, `vocab_sz` = 30 in the current case
    return torch.stack(out, dim=1)

  def reset(self): self.h = 0

seq_len = 16
seqs_16 = L((tensor(nums[i:i+seq_len]), tensor(nums[i+1:i+seq_len+1])) for i in range(0, len(nums)-seq_len-1, seq_len))
bs = 64
train_len = int(len(seqs_16) * 0.8)
seqs_train = make_dataset(seqs_16[:train_len], bs)
seqs_valid = make_dataset(seqs_16[train_len:], bs)
dls4 = DataLoaders.from_dsets(seqs_train, seqs_valid, bs=bs, drop_last=True, shuffle=False)

#%% check one prediction
def loss_func(preds, targets):
  ## `preds` will be of shape `n_hidden` x `seq_len` x `vocab_sz`, a 3D tensor
  ## `targets` will be of shape `bs` x `seq_len`, a 2D tensor (`bs` is the same as `n_hidden`)
  # print(f'preds.shape after reshaping: {preds.view(-1, len(vocab)).shape}')
  # print(f'len(target): {len(targets.view(-1))}')
  # print(f'target.shape after reshaping: {targets.view(-1).shape}')
  ## `preds` will be shaped to (`n_hidden` * `seq_len`) x `vocab_sz`, a 2D tensor
  ## `targets` will be shaped to (`bs` * `seq_len`) x 1, a 1D tensor
  return F.cross_entropy(preds.view(-1, len(vocab)), targets.view(-1))

learn = Learner(dls4, LMModel4(len(vocab), 64), loss_func=loss_func, metrics=accuracy,
                cbs=ModelResetter)
b1 = dls4.one_batch()
print(f'first batch => x.shape: {b1[0].shape}, y.shape: {b1[1].shape}')
pred = learn.model(dls4.one_batch()[0])
print(f'pred.shape: {pred.shape}')
loss_func(pred, b1[1])

#%% fine tune
learn.fit_one_cycle(15, 3e-3)

#%% 2 layers for the RNN
class LMModel5(Module):
  def __init__(self, vocab_sz, n_hidden, num_layers=2):
    self.i_h = nn.Embedding(vocab_sz, n_hidden)
    self.rnn = nn.RNN(n_hidden, n_hidden, num_layers, batch_first=True)
    # self.h_h = nn.Linear(n_hidden, n_hidden)
    self.h_o = nn.Linear(n_hidden, vocab_sz)
    self.h = torch.zeros(num_layers, bs, n_hidden)

  def forward(self, batch):
    out, h = self.rnn(self.i_h(batch), self.h)
      # self.h = F.relu(self.h_h(self.h + self.i_h(batch[:,i])))
    out = self.h_o(out)
    self.h = h.detach()
    return out

  def reset(self): self.h.zero_()

learn = Learner(dls4, LMModel5(len(vocab), 64, 2), loss_func=CrossEntropyLossFlat(),
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(15, 3e-3)

#%% Building an LSTM
class LSTMCell(Module):
  def __init__(self, ni, nh, num_layers=1):
    self.forget_gate = nn.Linear(ni+nh, nh)
    self.input_gate = nn.Linear(ni+nh, nh)
    self.cell_gate = nn.Linear(ni+nh, nh)
    self.output_gate = nn.Linear(ni+nh, nh)

  def forward(self, input, state):
    hidden_state, cell_state = state
    hidden_state = torch.stack([hidden_state, input], dim=1)
    #1
    forget_nn = torch.sigmoid(self.forget_gate(hidden_state))
    cell_state = cell_state * forget_nn
    #2
    input_nn = torch.sigmoid(self.input_gate(hidden_state))
    cell_nn = torch.tanh(self.cell_gate(hidden_state))
    cell_state = cell_state + (input_nn * cell_nn)
    #3
    output_nn = torch.sigmoid(self.output_gate(hidden_state))
    hidden_state = output_nn * torch.tanh(cell_state)
    return hidden_state, (hidden_state, cell_state)

class LMModel6(nn.Module):
  def __init__(self, vocab_sz, n_input, n_hidden, bs, num_layers):
    super().__init__()
    self.i_h = nn.Embedding(vocab_sz, n_input)
    self.lstm = nn.LSTM(n_input, n_hidden, num_layers, batch_first=True)
    self.h_o = nn.Linear(n_hidden, vocab_sz)
    self.hidden_state = torch.Tensor(num_layers, bs, n_hidden)
    self.cell_state = torch.zeros(num_layers, bs, n_hidden)

  def forward(self, batch):
    h = self.i_h(batch)
    res, (h, c) = self.lstm(h, (self.hidden_state, self.cell_state))
    self.hidden_state = h.detach()
    self.cell_state = c.detach()
    return self.h_o(res)

  def reset(self):
    self.hidden_state.zero_()
    self.cell_state.zero_()

learn = Learner(dls4, model=LMModel6(len(vocab), 64 ,64, 64, 2), loss_func=CrossEntropyLossFlat(),
                metrics=accuracy, cbs=ModelResetter)
learn.fit_one_cycle(15, 1e-2)
