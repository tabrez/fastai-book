#%% main
import torch as t
import altair as alt
import pandas as pd
import functools


#%% loss function
def mse(preds, labels):
  return t.mean((preds - labels) ** 2)


#%% model
def linear_reg(x, W, b):
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
  Ws = []
  bs = []
  for epoch in range(num_epochs):
    preds = linear_reg(x, W, b)
    loss = mse(preds, y)
    losses.append(loss.item())
    loss.backward()
    if dbg: _print_dbg()
    Ws.append(round(W.item(), 4))
    bs.append(round(b.item(), 4))
    W.data = W.data - (W.grad.data * lr)
    b.data = b.data - (b.grad.data * lr)
    W.grad.data.zero_()
    b.grad.data.zero_()
  return (losses, W, b, Ws, bs)


#%% driver function
def driver(x, y, epochs, lr, dbg=False, prt_summary=True):
  g = t.Generator().manual_seed(42)
  W = t.randn(1, generator=g, requires_grad=True)
  b = t.randn(1, generator=g, requires_grad=True)

  lossesf, Wf, bf, Ws, bs = train(x, y, W, b, epochs, lr, dbg)
  if prt_summary:
    print('training done')
    print('===================final results:===================')
    print(f'loss: {lossesf[-1]}, Wf: {Wf}, bf: {bf}')
  if dbg:
    print(f'labels: {y}')
    preds = linear_reg(x, Wf, bf)
    print(f'preds: {preds}')
    print(f'(preds - y): {preds - y}')
  return lossesf, Wf, bf, Ws, bs


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


def plot_epochs_vs(epochs, losses,  y_label, scale=None, last_iters=5000):
  i = last_iters
  y = alt.Y(y_label, type='quantitative').title(y_label)
  if scale is not None:
    y = y.scale(domain=scale)
  return alt.Chart(pd.DataFrame({'epoch': range(epochs)[-i:],
                                          y_label: losses[-i:]})).mark_line(clip=True).encode(
                                            alt.X('epoch', type='quantitative'),
                                            y)


def plot_x_y(xs, ys, x_label, y_label, x_scale=None, y_scale=None, last_iters=2000):
  i = last_iters
  x = alt.X(x_label, type='quantitative', sort=None)
  y = alt.Y(y_label, type='quantitative')
  if x_scale is not None:
    x = x.scale(domain=x_scale)
  if y_scale is not None:
    y = y.scale(domain=y_scale)
  return alt.Chart(pd.DataFrame({x_label: xs[-i:], y_label: ys[-i:]})) \
    .mark_line(clip=True) \
    .encode(x, y)


#%% main
x = t.arange(start=0, end=5)
# try the following values for x & y to see W & b diverge from their starting point towards infinity
# x = t.arange(start=10, end=15)
# try  changing the following w and b values
y = (27 * x) + 73

# Use num_epochs=500, lr=0.001 to see smooth change in parameters
# Change lr=0.1 to see jagged change in parameters but eventually smooth out with increased num_epochs value
# Change lr=0.2 to see parameters never converge to their correct values.
epochs = 50
lossesf, Wf, bf, Ws, bs = driver(x, y, epochs, lr=0.0G1)

# plot_x_y_preds(x, y, Wf, bf)
upper = plot_epochs_vs(epochs, Ws, 'W') | \
        plot_epochs_vs(epochs, bs, 'b') | \
        plot_epochs_vs(epochs, lossesf, 'loss')


lower = plot_x_y(Ws, lossesf, 'Ws', 'loss') | \
        plot_x_y(bs, lossesf, 'bs', 'loss')
alt.vconcat(upper, lower)

# use the following code to zoom into the plot using scaling
# pxy = functools.partial(plot_x_y, y_scale=(0, 150))
# lower2 = pxy(Ws, lossesf, 'Ws', 'loss', x_scale=(28, 35)) | \
#         pxy(bs, lossesf, 'bs', 'loss', x_scale=(80, 90))
# alt.vconcat(upper, lower2)

# import IPython
# IPython.display.display(alt.vconcat(upper, lower))

#%% does it matter if we run a few more training loops with lower learning rate
# epochs = 5000
# driver(x, y, epochs, lr=0.00001)

## References:
# 1. Surprisingly similar to the code we have written above, maybe there are many more:
# https://machinelearningmastery.com/training-a-linear-regression-model-in-pytorch/
# 2. Different ways to use `encode` in Altair:
# https://altair-viz.github.io/user_guide/encodings/index.html
