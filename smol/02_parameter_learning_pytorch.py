#%% same program using pytorch library
import torch as t
class LinearRegression(t.nn.Module):
    def __init__(self, n_features, n_outputs):
        super(LinearRegression, self).__init__()
        self.linear = t.nn.Sequential(
          # try different number of activations below: 1, 2, 3, 10
          # loss is good on some runs but not the others with lower numbers like 1, 2 - why?
          # Try with fixed seed for parameters created by nn.LinearRegression
          t.nn.Linear(n_features, 3),
          t.nn.ReLU(),
          t.nn.Linear(3, n_outputs))

    def forward(self, x):
        return self.linear(x)

def driver(n_epochs=10, lr=0.1):
  model = LinearRegression(3, 1)
  mse_loss = t.nn.MSELoss()
  optimizer = t.optim.SGD(model.parameters(), lr)

  # x1 = t.linspace(start=-1, end=1, steps=90).view(-1, 1).float()
  x1 = t.randn(90).view(-1, 1)
  x2 = x1*x1
  x3 = x1*x1*x1
  y = (122 * x1) + 73 * x2 + 12 * x3 + 12
  # print(f'y: {y}')
  # y_mean = y.mean()
  # y_std = y.std()
  # y = (y - y_mean) / y_std
  print(f'y: {y}')

  for epoch in range(n_epochs):
      y_pred = model(t.cat((x1, x2, x3), dim=1))
      loss = mse_loss(y_pred, y)

      if (epoch + 1) % 10 == 0:
          print(f'params: {list(model.parameters())}')
          print(f'grads: {[p.grad for p in model.parameters()]}')
          print(f'Epoch: {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}')

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  return loss, model

lossf, model = driver(950, 0.0001)
print('++++++++++++++++++++++++++++++++++++++++++++++')
print(f'final loss: {lossf}')
print(f'params: {list(model.parameters())}')
# print(f'grads: {[p.grad for p in model.parameters()]}')

#%%
x1 = 0.3
x2 = x1*x1
x3 = x1*x1*x1
new_y = (122 * x1) + 73 * x2 + 12 * x3 + 12
# new_y = 122 * 0.3 + 73 * 0.09 + 12 * 0.027 + 12
print(f'new_y: {new_y}')
pred_y = model(t.tensor([x1, x2, x3]))
print(f'pred_y: {pred_y}')
