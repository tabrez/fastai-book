#%% pytorch neural network

import torch as t
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


class NN(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.hidden_size = 2
    self.w1 = nn.Parameter(t.rand(input_size, self.hidden_size))
    self.b1 = nn.Parameter(t.rand(1))
    self.w2 = nn.Parameter(t.rand(self.hidden_size, output_size))
    self.b2 = nn.Parameter(t.rand(1))


  def forward(self, batch):
    # print('batch:', batch.unsqueeze(dim=0).T)
    res = t.matmul(batch.unsqueeze(dim=0).T, self.w1) + self.b1
    res = F.relu(res)
    res = t.matmul(res, self.w2) + self.b2
    return res


class LinearXYDataset(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y


  def __len__(self):
    return len(self.x)


  def __getitem__(self, index):
    return self.x[index], self.y[index]


def driver(dl):
  model = NN(1, 1)
  print(model)
  model.train()

  loss_fn = nn.MSELoss()

  optimiser = t.optim.SGD(model.parameters(), lr=0.00001)

  for batch, (X, y) in enumerate(dl):
    # print('X:', X, 'y:', y)
    preds = model(X)
    loss = loss_fn(preds, y)
    print('batch:', batch, 'loss:', loss)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()

  return loss, model


x = t.linspace(1, 95, 55)
y = 3 * x + 78
dl = DataLoader(LinearXYDataset(x, y), shuffle=False, batch_size=2)
loss, model = driver(dl)
print('final loss:', loss)
print('final model parameters:', list(model.parameters()))


#%%
new_x = t.Tensor([99, 152])
new_y = 3 * new_x + 78
new_pred = model(new_x)
print('y:', new_y, 'pred:', new_pred)


#%%
class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, seq_len):
    super().__init__()
    self.wx = nn.Parameter(t.rand(input_size, hidden_size))
    self.bx = nn.Parameter(t.rand(1))
    self.wh = nn.Parameter(t.rand(hidden_size, hidden_size))
    self.bh = nn.Parameter(t.rand(1))
    self.wy = nn.Parameter(t.zeros(hidden_size, output_size))
    self.by = nn.Parameter(t.rand(1))
    self.h = t.zeros(hidden_size, hidden_size)
    self.seq_len = seq_len

  def forward(self, batch):
    for i in range(self.seq_len):
      res = t.matmul(batch[:,i].unsqeeze(dim=0).T, self.wx) + self.bx
      self.h = t.matmul(self.h, self.wh) + self.bh + res
      self.h = F.tanh(self.h)
    out = t.matmul(self.h, self.wy) + self.by
    self.h = self.h.detach()
    return F.softmax(out)

  def reset(self):
    self.h.zero_()

# %%
