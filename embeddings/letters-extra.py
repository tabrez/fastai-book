#%% How nn.CrossEntropyLoss() works
## https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
import torch

loss_fn = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
loss = loss_fn(input, target)
print(f'input: {input}, \ntarget: {target}, \nloss: {loss}')

#%% How TensorDataset works
print(f'len of ds.input: {len(ds["input"])}, len of ds.output: {len(ds["output"])}')
tds = TensorDataset(torch.tensor(ds['input']), torch.tensor(ds['output']))
print(f'len tds: {len(tds)}, tds[0:3]: {tds[0:3]}')
print(f'{torch.tensor([1,2,3])}')

#%% flatten a tensor list
t = torch.rand([1, 5, 2])
print(f't.shape: {t.shape}, t: {t}')
print(f't.unstack(): {t.view(-1)}')

#%% create the dataset - version 1
def make_ds1(text):
  ds1 = []
  # First few elements are not present in the input and last element is not present in the output
  # alternatively add dummy elements at the start and the end of the text
  for i in range(len(text)-1):
    pair = (text[i], text[i+1])
    ds1.append(pair)

ds1 = make_ds1(text)
print(f'len of ds1: {len(ds1)}, firt few: {ds1[:5]}')

#%% create the dataset - version 2
def make_ds2(text):
  ds2 = { 'input': [], 'output': []}
  for i in range(len(text)-1):
    ds2['input'].append(text[i])
    ds2['output'].append(text[i+1])

ds2 = make_ds2(text)
print(f'len of input: {len(ds2["input"])}, len of output: {len(ds2["output"])}')

#%% make_text()
  # text = text.lower()
  # # print(text[:40])
  # table = str.maketrans('\n\t -ˈ—éəɪɡ–', '___________', string.punctuation)
  # text = text.translate(table)
  # return text
