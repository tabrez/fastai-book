#%% imports
import torch as t
import altair as alt
import pandas as pd

#%% loss function
def mse(preds, y):
  return t.mean((preds - y) ** 2)

#%% model
def linear_reg(x, W, b):
  return t.matmul(x,  W) + b

#%% trainer
def train(x, y, W, b, epochs, lr, dbg=False):
  def _print_dbg():
    print('++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'epoch: {epoch}')
    # print(f'y: {y}')
    print(f'W: {W}, b: {b}')
    # print(f'W.grad: {W.grad}, b.grad: {b.grad}')
    # print(f'loss: {loss}')

  losses = []
  for epoch in range(epochs):
    preds = linear_reg(x, W, b)
    loss = mse(preds, y)
    losses.append(loss.item())
    loss.backward()
    if dbg: _print_dbg()
    W.data = W.data - (W.grad.data * lr)
    b.data = b.data - (b.grad.data * lr)
    W.grad.zero_()
    b.grad.zero_()
  return losses, W, b

#%% driver function
def driver(x, y, epochs, lr, dbg=False, prt_summary=True):
  g = t.Generator().manual_seed(42)
  W = t.randn(2, 1, generator=g, requires_grad=True)
  b = t.randn(1, generator=g, requires_grad=True)

  lossesf, Wf, bf = train(x, y, W, b, epochs, lr, dbg)
  if prt_summary:
    print('training done')
    print('===================final results:===================')
    print(f'loss: {lossesf[-1]}, Wf: {Wf}, bf: {bf}')
  if False:
    print(f'labels: {y}')
    preds = linear_reg(x, Wf, bf)
    print(f'preds: {preds}')
    print(f'(preds - y): {preds - y}')
  return lossesf, Wf, bf

#%% plot input vs y and input vs predictions using final values of model parameters
def plot_x_y_preds(x, y, W, b):
  preds = linear_reg(x, W, b)
  data_x_y = pd.DataFrame({'x': x,
                      'y': y})
  chart_x_y = alt.Chart(data_x_y).mark_point().encode(
    alt.X('x:Q'),
    alt.Y('y:Q')
  )
  data_x_preds = pd.DataFrame({'x': x, 'preds': preds.detach().numpy() })
  chart_x_preds = alt.Chart(data_x_preds).mark_point(
    shape='triangle-up').encode(
    x='x:Q',
    y='preds:Q',
  )
  return chart_x_y + chart_x_preds

def plot_epochs_losses(epochs, losses, last_iters=5000):
  i = last_iters
  return alt.Chart(pd.DataFrame({'epoch': range(epochs)[-i:],
                                          'losses': losses[-i:]})).mark_line().encode(
                                            x='epoch:Q',
                                            y='losses:Q')

#%% two inputs
x2 = t.stack([t.arange(start=0, end=95), t.arange(start=50, end=145)], dim=1).float()
y2 = t.unsqueeze((177 * x2[:,0]) + (22 * x2[:,1]) + 54, dim=1).float()

## If you pick wildly different ranges for x2's first and second columns, there's a big
## difference in their corresponding W.grad gradients and it's hard for loss to converge to
## a minimum, normalise the columns if possible:
# mean = t.mean(x2, dim=0)
# std = t.std(x2, dim=0)
# x2 = (x2 - mean) / std
## Use a larger learning rate after normalising
## Currently, the parameters learned are wildly incorrect with above normalisation while
## the loss is really small


# print(f'x2: {x2}, y2: {y2}')
epochs = 550
## loss gets really big with a bigger learning rate e.g. lr=0.0001 for 10 samples
## try lr=0.0001 for 10 samples and lr=0.00005 for 25 samples
lossesf, Wf, bf = driver(x2, y2, epochs, lr=0.0005, dbg=True, prt_summary=True)

# x2 = t.arange(start=0, end=15)
# y = (7 * x1) + (22 * x2) + 3

# plot_x_y_preds(x, y, Wf, bf)
#%%
plot_epochs_losses(epochs, lossesf)
