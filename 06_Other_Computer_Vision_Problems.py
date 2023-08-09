
# %% Imports
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
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

#%% inspect dls
print(len(dls.items))
dlsone = dls.one_batch()
print(len(dlsone), dlsone[0].shape)
dlstrainone = dls.train.one_batch()
print(len(dlstrainone), dlstrainone[0].shape)

print(len(dls.train), len(dls.valid), len(dls.train_ds), len(dls.valid_ds))
x, y = dls.train.one_batch()
print('one batch of data:', len(x), len(y), x.shape, y.shape)
print('items: ', type(dls.items), len(dls.items), dls.items.head())
dls.show_batch(nrows=1, ncols=3)

#%% Binary Cross Entropy
learn = vision_learner(dls, resnet18)
learn.model.cuda()
x, y = dls.train.one_batch()
activations = learn.model(x)
print(f'activations.shape: {activations.shape}')
print(f'activations[0]: {activations[0]}')

def binary_cross_entropy(activations, targets):
  s = torch.sigmoid(activations)
  return -torch.where(targets == 1, 1-activations, activations).log().mean()

# %%
