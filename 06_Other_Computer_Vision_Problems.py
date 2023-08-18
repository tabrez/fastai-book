
# %% Imports
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.all import *
import fastai
print(f'fastai version: {fastai.__version__}')

import warnings
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet34_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet34_Weights.DEFAULT` to get the most up-to-date weights.")

# BASE_DIR='/content/gdrive/MyDrive/Colab Notebooks'
BASE_DIR='/home/tabrez/MEGA/MEGA/code/fastai-book'
#!ln -sf '/content/gdrive/MyDrive/Colab Notebooks' ~/.fastai

import pydoc
def source(fn):
  pydoc.pager(pydoc.render_doc(fn))

#%% Download the data
path = untar_data(URLs.PASCAL_2007)

#%% Explore downloaded folder
FASTAI_HOME='/home/tabrez/.fastai'
print('path', path, path.ls())
!ls -l /home/tabrez/.fastai/data/pascal_2007/train/ | wc -l
!cat /home/tabrez/.fastai/data/pascal_2007/train.csv | head -5
!cat /home/tabrez/.fastai/data/pascal_2007/train.csv | grep True | wc -l
!ls -l /home/tabrez/.fastai/data/pascal_2007/test/ | wc -l
#%%
df = pd.read_csv(path/'train.csv')
df.head()

#%%
df[df['is_valid'] == True][0:5]
df.index[df['is_valid']==True]
df.index[df['is_valid']==True].tolist()
df.index[df['is_valid']].tolist()

def get_x1(item): return item['fname']
def get_x(item): return path/'train'/item['fname']
def get_y1(item): return item['labels']
def get_y(item): return item['labels'].split(' ')
dblock = DataBlock(get_x=get_x, get_y=get_y)
dsets = dblock.datasets(df)
dsets.train[5]
#%%
def get_train_valid(df):
  train = df.index[~df['is_valid']].tolist()
  valid = df.index[df['is_valid']].tolist()
  return train, valid

db = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
               get_x=get_x,
               get_y=get_y,
               splitter=get_train_valid,
               item_tfms=RandomResizedCrop(128, min_scale=0.35))

dls = db.dataloaders(df)

#%%
def inspect_dls(dls):
  print(f'type(items): {type(dls.items)}, len(items): {len(dls.items)}')
  x, y = dls.one_batch()
  print(f'first batch -> x.shape: {x.shape}')
  print(f'first batch -> y.shape: {y.shape}')
  print(f'''\nlen(train): {len(dls.train)},\
        \nlen(valid): {len(dls.valid)},\
        \nlen(train_ds): {len(dls.train_ds)},\
        \nlen(valid_ds): {len(dls.valid_ds)}''')

  x, y = dls.train.one_batch()
  print(f'''\ntrain -> first batch -> x.shape: {x.shape},\
        \ntrain -> first batch -> y.shape: {y.shape}''')

inspect_dls(dls)

#%% Binary Cross Entropy
learn = vision_learner(dls, resnet18)
learn.model.cuda()
x, y = dls.train.one_batch()
activs = learn.model(x)
print(f'''type(x): {type(x)},
      type(activs): {type(activs)},
      type(y): {type(y)}''')
print(f'''x.shape: {x.shape},
      activs.shape: {activs.shape},
      y.shape: {y.shape}''')
print(f'''activs: {activs},
      y: {y}''')
print(f'activs[0]: {activs[0]}')

def binary_cross_entropy(activs, targets):
  s = torch.sigmoid(activs)
  return -torch.where(targets == 1, s, 1-s).log().mean()
bce_loss = binary_cross_entropy(torch.Tensor(activs), torch.Tensor(y))
print(f'bce_loss: {bce_loss}')

loss = F.binary_cross_entropy_with_logits(torch.Tensor(activs), torch.Tensor(y))
print(f'loss: {loss}')

#%% accuracy metric
# `argmax`` gets us the index of the largest value for each of the inner-most lists.
# This method works for getting predictions for single category variables like in
# pet breed or mnist classification
# print(f' activs[0:3]: {activs[0:3]}, argmax : {activs[0:3].argmax(dim=-1)}')

# accuracy for multi-category variables
def accuracy_multi(activs, targets, threshold=0.5, sigmoid=True):
  if sigmoid: activs = activs.sigmoid()
  print(f'filtered activations: {(activs>threshold).float().sum()}')
  print(f'1s in targets: {targets.sum()}')
  targets = targets.bool()
  return ((activs > threshold) == targets).float().mean()

#%% fine tune
learn = vision_learner(dls, resnet50, metrics=partial(accuracy_multi, threshold=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)

#%% check metrics with different thresholds
learn.metrics = partial(accuracy_multi, threshold=0.005)
print(f'loss at threshold 0.005: {learn.validate()}')
learn.metrics = partial(accuracy_multi, threshold=0.1)
print(f'loss at threshold 0.1: {learn.validate()}')
learn.metrics = partial(accuracy_multi, threshold=0.4)
print(f'loss at threshold 0.4: {learn.validate()}')
learn.metrics = partial(accuracy_multi, threshold=0.99)
print(f'loss at threshold 0.99: {learn.validate()}')

#%% pick the best threshold
predictions, targets = learn.get_preds()
accuracies = accuracy_multi(predictions[:3], targets[:3], threshold=0.9, sigmoid=False)
xs = torch.linspace(0.05, 0.95, 29)
accuracies = [accuracy_multi(predictions, targets, threshold=i, sigmoid=False) for i in xs]
# print(f'accuracies: {accuracies}')
plt.plot(xs, accuracies)

# %%
print(len(predictions), predictions.shape, len(targets), targets.shape, len(predictions[:3]))
print(predictions[:3], predictions[:3] > 0.5)
print(f'threshold 0.005 - predictions: {(predictions[:3] > 0.005).float().sum(dim=-1)}, targets: {targets[0:3].sum(dim=-1)}')
print(f'threshold 0.4 - predictions: {(predictions[:3] > 0.4).float().sum(dim=-1)}, targets: {targets[0:3].sum(dim=-1)}')
print(f'threshold 0.99 - predictions: {(predictions[:3] > 0.99).float().sum(dim=-1)}, targets: {targets[0:3].sum(dim=-1)}')
