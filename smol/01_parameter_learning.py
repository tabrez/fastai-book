#%%
import torch as t
import altair as alt
xx = t.arange(start=0, end=5)
# x2 = t.arange(start=0, end=15)
yy = (27 * xx) + 37
# y = (7 * x1) + (22 * x2) + 3

import pandas as pd
data = pd.DataFrame({'x': xx,
                     'y': yy})
chart_x_y = alt.Chart(data).mark_point().encode(
  alt.X('x:Q'),
  alt.Y('y:Q')
)
chart_x_y

#%% loss function
def mse(preds, labels):
  return t.mean((preds - labels) ** 2)

#%% model
def linear_reg(x, W, b):
  preds = (W * x) + b
  # print(f'pred: {preds}')
  return (W * x) + b

#%% trainer
def train(x, y, W, b, num_epochs, lr, dbg):
  def _print_dbg():
    print(f'epoch: {epoch}')
    print(f'y: {y}')
    print(f'W.grad: {W.grad}, b.grad: {b.grad}')
    print(f'W: {W}, b: {b}')
    print(f'loss: {loss}')

  losses = []
  for epoch in range(num_epochs):
    preds = linear_reg(x, W, b)
    loss = mse(preds, y)
    losses.append(loss.item())
    loss.backward()
    if dbg: _print_dbg()
    W.data = W.data - (W.grad.data * lr)
    b.data = b.data - (b.grad.data * lr)
    W.grad.data.zero_()
    b.grad.data.zero_()
  return (losses, W, b)

#%% driver function
def driver(x, y, epochs, lr, dbg=False, prt_summary=True):
  g = t.Generator().manual_seed(42)
  W = t.randn(1, generator=g, requires_grad=True)
  b = t.randn(1, generator=g, requires_grad=True)

  lossesf, Wf, bf = train(x, y, W, b, epochs, lr, dbg)
  if prt_summary:
    print('training done')
    print('===================final results:===================')
    print(f'loss: {lossesf[-1]}, Wf: {Wf}, bf: {bf}')
  if dbg:
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

def plot_epochs_losses(epochs, losses, last_iters=0):
  i = len(epochs) if last_iters == 0 else last_iters
  return alt.Chart(pd.DataFrame({'epoch': range(epochs)[-i:],
                                          'losses': losses[-i:]})).mark_line().encode(
                                            x='epoch:Q',
                                            y='losses:Q')

#%% main
x = t.arange(start=0, end=5)
y = (27 * x) + 37
epochs = 200
lossesf, Wf, bf = driver(x, y, epochs, lr=0.1)

# plot_x_y_preds(x, y, Wf, bf)
plot_epochs_losses(epochs, lossesf, 20)

#%% does it matter if we run a few more training loops with lower learning rate
# epochs = 5000
# driver(x, y, epochs, lr=0.00001)

## References:
# 1. Surprisingly similar to the code we have written above, maybe there are many more:
# https://machinelearningmastery.com/training-a-linear-regression-model-in-pytorch/
