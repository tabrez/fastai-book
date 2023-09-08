
#%% imports
import torch as t
import altair as alt
import pandas as pd

#%% same program using pytorch library
true_params = t.tensor([177., 22., 71., 54.])

n_features = 3
n_outputs = 1
model = t.nn.Linear(n_features, n_outputs)
mse_loss = t.nn.MSELoss()
optimizer = t.optim.SGD(model.parameters(), lr=.1)

n_samples = 95

mean = t.zeros(n_features)
cov = t.eye(n_features)
X = t.distributions.MultivariateNormal(mean, cov).sample((n_samples,))
X_bias = t.cat((X, t.ones(n_samples, 1)), dim=1)

noise = t.normal(0, 10, (n_samples,))
y = t.matmul(X_bias, true_params) + noise
y = y.view(-1, 1)

n_epochs = 1000
for epoch in range(n_epochs):
    y_pred = model(X)
    loss = mse_loss(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}')
print(f'params: {list(model.parameters())}')
