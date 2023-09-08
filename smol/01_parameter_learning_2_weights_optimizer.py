#%% imports
import torch as t
import altair as alt
import pandas as pd

#%% loss function
def mse(preds, y):
  return t.mean((preds - y) ** 2)

#%% model
def module(n):
  g = t.Generator().manual_seed(42)
  W = t.randn(n, 1, generator=g, requires_grad=True)
  def linear_reg(x):
    return t.matmul(x,  W[:-1]) + W[-1]
  def params():
    return W
  return linear_reg, params

#%% optimizer
def optimizer(params, lr):
  def step():
      params.data -= (params.grad.data * lr)
  def zero():
      params.grad.zero_()
  return step, zero

#%% trainer
def train(x, y, m, opt, epochs, dbg=False):
  def _print_dbg():
    print('++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'epoch: {epoch}')
    print(f'y: {y}')
    print(f'W: {params()}')
    print(f'W.grad: {params().grad}')
    print(f'loss: {loss}')

  learn, params = m
  step, zero = opt
  losses = []
  for epoch in range(epochs):
    preds = learn(x)
    loss = mse(preds, y)
    losses.append(loss.item())
    loss.backward()
    if dbg: _print_dbg()
    step()
    zero()
  return losses

#%% driver function
def driver(x, y, m, epochs, lr, dbg=False, prt_summary=True):
  # b = t.randn(1, generator=g, requires_grad=True)

  learn, params = m
  opt = optimizer(params(), lr)
  lossesf = train(x, y, m, opt, epochs, dbg)
  if prt_summary:
    print('training done')
    print('===================final results:===================')
    print(f'loss: {lossesf[-1]}')
    print(f'params: {params()}')
  if dbg:
    print(f'labels: {y}')
    preds = learn(x)
    print(f'preds: {preds}')
    print(f'(preds - y): {preds - y}')
  return lossesf

#%% plot input vs y and input vs predictions using final values of model parameters
def plot_x_y_preds(x, y):
  learn, params = module(3)
  preds = learn(x)
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
x2 = t.stack([t.arange(start=0, end=95), t.arange(start=50, end=145), t.arange(start=200, end=295)], dim=1).float()
y2 = t.unsqueeze((177 * x2[:,0]) + (22 * x2[:,1]) + (71 * x2[:,2]) + 54, dim=1).float()

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
epochs = 950
## loss gets really big with a bigger learning rate e.g. lr=0.0001 for 10 samples
## try lr=0.0001 for 10 samples and lr=0.00005 for 25 samples
m = module(4)
lossesf = driver(x2, y2, m, epochs, lr=0.000005, prt_summary=True)

# x2 = t.arange(start=0, end=15)
# y = (7 * x1) + (22 * x2) + 3

# plot_x_y_preds(x, y, Wf, bf)
#%%
plot_epochs_losses(epochs, lossesf)
