#%%
import torch as t
import altair as alt
x = t.arange(start=0, end=5)
# x2 = t.arange(start=0, end=15)
y = (27 * x) + 37
# y = (7 * x1) + (22 * x2) + 3

import pandas as pd
data = pd.DataFrame({'x': x,
                     'y': y})
alt.Chart(data).mark_point().encode(
  alt.X('x:Q'),
  alt.Y('y:Q')
)

#%% loss function
def mse(preds, labels):
  return t.mean((preds - labels) ** 2)

#%% model
def linear_reg(x, W, b):
  preds = (W * x) + b
  print(f'pred: {preds}')
  return (W * x) + b

#%% trainer
def train(x, y, W, b, num_epochs, lr, dbg=True):
  losses = []
  for epoch in range(num_epochs):
    if dbg:
      print(f'epoch: {epoch}')
      print(f'y: {y}')
    preds = linear_reg(x, W, b)
    loss = mse(preds, y)
    losses.append(loss.item())
    loss.backward()
    if dbg:
      print(f'W.grad: {W.grad}, b.grad: {b.grad}')
      print(f'W: {W}, b: {b}')
      print(f'loss: {loss}')
    else: print('.', end=' ')
    W.data = W.data - (W.grad.data * lr)
    b.data = b.data - (b.grad.data * lr)
    W.grad.data.zero_()
    b.grad.data.zero_()
  return (losses, W, b)

#%%
g = t.Generator().manual_seed(42)
W = t.randn(1, generator=g, requires_grad=True)
b = t.randn(1, generator=g, requires_grad=True)

epoch_nums = 200
lossesf, Wf, bf = train(x, y, W, b, epoch_nums, 0.1, dbg=True)
print('training done')

print('final results:')
print(f'loss: {lossesf[-1]}, Wf: {Wf}, bf: {bf}')
# print(f'labels: {y}')
# preds f= Wf * x + bf
# print(f'preds: {predsf}')
# print(f'(preds - y): {predsf-y}')

#%% second training loop
# epoch_nums = 5000
# lossesf2, Wf2, bf2 = train(x, y, Wf, bf, epoch_nums, 0.00001, dbg=False)
# print('final results:')
# print(f'loss: {lossesf2[-1]}, Wf: {Wf2}, bf: {bf2}')

#%%
x_y = alt.Chart(pd.DataFrame({'x': x, 'y': y})).mark_point().encode(
  x='x',
  y='y',
)
predsf = Wf * x + bf
x_preds = alt.Chart(pd.DataFrame({'x': x, 'preds': predsf.detach().numpy() })).mark_point(
  shape='triangle-up').encode(
  x='x:Q',
  y='preds:Q',
)
# x_y + x_preds
#%%
epochs_losses = alt.Chart(pd.DataFrame({'epoch': range(epoch_nums)[-20:],
                                        'losses': lossesf[-20:]})).mark_line().encode(
                                          x='epoch:Q',
                                          y='losses:Q')
epochs_losses
