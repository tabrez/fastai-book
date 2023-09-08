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
  learn, _ = module(3)
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

#%% three inputs
x2 = t.stack([t.arange(start=0, end=95), t.arange(start=50, end=145), t.arange(start=200, end=295)], dim=1).float()
y2 = t.unsqueeze((177 * x2[:,0]) + (22 * x2[:,1]) + (71 * x2[:,2]) + 54, dim=1).float()
## loss gets really big with a bigger learning rate e.g. lr=0.0001 for 10 samples
## try lr=0.0001 for 10 samples and lr=0.00005 for 25 samples
model = module(4)
epochs = 950
lossesf = driver(x2, y2, model, epochs, lr=5e-6, prt_summary=True)

# plot_x_y_preds(x, y)
# first few losse are huge, so skip them
plot_epochs_losses(epochs-5, lossesf[5:])

#%% better synthetic data
t.manual_seed(0)
true_params = t.tensor([177., 22., 71., 54.])

n_samples = 95
n_features = 3
mean = t.zeros(n_features)
cov = t.eye(n_features)
X = t.distributions.MultivariateNormal(mean, cov).sample((n_samples,))
X_bias = t.cat((X, t.ones(n_samples, 1)), dim=1)

noise = t.normal(0, 10, (n_samples,))
y = t.matmul(X_bias, true_params) + noise
y = y.view(-1, 1)

model = module(4)
epochs = 950
lossesf = driver(X, y, model, epochs, lr=.1, prt_summary=True)
