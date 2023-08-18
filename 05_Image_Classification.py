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

# %%  Check cuda
torch.cuda.is_available()

# %% download dataset
path = untar_data(URLs.PETS)
path, len((path/'images').ls()), len(get_image_files(path/'images'))

# %% create DataLoader
def label_func(fname):
  return'_'.join(fname.name.split('_')[:-1])

db = DataBlock(blocks=(ImageBlock, CategoryBlock),
               get_items = get_image_files,
               splitter = RandomSplitter(valid_pct=0.2, seed=42),
               get_y = label_func,
               item_tfms=Resize(460),
               batch_tfms=aug_transforms(size=224, min_scale=0.75))
# db.summary(path/'images')
dls = db.dataloaders(path/'images', shuffle=True)

# %% inspect one batch of data
print(len(dls.items), len(dls.train), len(dls.valid), len(dls.train_ds), len(dls.valid_ds))
x, y = dls.one_batch()
print('one batch of data:', len(x), len(y), x.shape, y.shape)
print('first sample: ', dls.items[0])
# print('first sample:', x[0])
index = 0
print('category name to integer mapping:', dls.vocab)
print('category index and name of first sample:', y[index], dls.vocab[y[index]])
print('shuffle true? ', dls.shuffle)
print('first item:', dls.items[0])
print('first item:', dls.items[0])
# ??dls.get_idxs
print('first idxs:', dls.get_idxs()[0])
print('first idxs:', dls.get_idxs()[0])
print('image to tensor mapping:', dls.items[index])
# dls.show_batch(nrows=1, ncols=3)

# %% fine tune
import time
def print_time_taken(fn):
  def wrapper(*args, **kwargs):
    start_time = time.time()
    res = fn(*args, **kwargs)
    print("--- %s seconds ---" % (time.time() - start_time))
    return res
  return wrapper

def fine_tune_n_times(dls, model, epochs, lrs):
  print('---------fine-tuning the model----------------')
  for lr in lrs:
    learn = vision_learner(dls, model, metrics=accuracy)
    print(str(lr))
    learn.fine_tune(epochs, base_lr=lr)
  torch.cuda.empty_cache()

# fine_tune_n_times(dls, model=resnet34, epochs=5, lrs=[0.0001, 0.001, 0.01, 0.1, 1.])

# Learner.save and Learner.load use the default path based on following fields
# learn.path, learn.model_dir

## Fine tune the model if no saves are available to load
learn = vision_learner(dls, resnet34, metrics=accuracy)
# learn.fine_tune(2, base_lr=0.001)
## Save the model & weights so we can load it later without incurring processing costs again and again
#learn.save(file=Path(f'{BASE_DIR}/saves/oxford-iiit-pet'))

## Load the model & weights from the file if they've been saved before
learn.load(file=Path(f'{BASE_DIR}/saves/oxford-iiit-pet'))

# %% run fine_tune with freeze() enabled
# learn = vision_learner(dls, resnet34, metrics=accuracy)
# learn.freeze()
# learn.fit(5, lr=0.001)

# %% inspect learn
# fastai.learner.Learner??
type(learn)
def print_recorder_values(values):
  # learn.recorder.values has one list per epoch, each list contains
  # training error, validation error, and passed metric(e.g. `accuracy`)
  for i in range(len(values)):
    print(f'epoch {i}:')
    errors = values[i]
    for j in range(len(errors)):
      print(errors[j])
print_recorder_values(learn.recorder.values)
# learn.show_results()

# %% Cross-entropy loss
print('---------predictions for validation dataset---------------')
x, y = dls.one_batch()
len(y), y
# ??learn.get_preds
preds = learn.get_preds()
print(len(dls.items), len(dls.train_ds), len(dls.valid_ds), len(dls.train), len(dls.valid))
print(len(preds), preds[0].shape, preds[1].shape)
xv, yv = dls.valid.one_batch()
print(preds[0][0], yv[0])

print('---------predictions for first batch of training data-------')
x, y = dls.one_batch()
preds = learn.get_preds(dl=[(x, y)])
print(len(preds), preds[0].shape, preds[1].shape)
print(preds[0][0], y[0])
len(preds[0]), preds[0].sum()

# %% sigmoid
# plot_function(torch.sigmoid, min=-4, max=4)
g = torch.Generator()
g.manual_seed(42)
def fn(n):
  # return n x 2 matrix of dummy data with a standard deviation of 2
  # each of the 2 columns represents confidence that the input is 3 & 7 respectively
  return torch.randn((n, 2), generator=g) * 2
activations = fn(6)
print(activations)
print('col1: ', activations[:,0])

col1 = activations[:,0]
col2 = activations[:,1]
diff = col1 - col2
sigmoids = torch.sigmoid(diff)
sig_preds = torch.stack((sigmoids, 1-sigmoids), dim=1)
sig_preds.sum(dim=1)

# %% our softmax
def softmax(x):
  print(x, torch.exp(x))
  return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)

softmax(tensor([[1, 2, 3]])).sum()

# %% torch.softmax
softmax_preds = torch.softmax(activations, dim=1)
softmax_preds, softmax_preds.sum(dim=1)

print(math.exp(0.02), math.exp(-2.49), math.exp(1.25))
torch.softmax(torch.tensor([[0.02, -2.49, 1.25]]), dim=1).sum()

# %%
## synthetic data representing 3s and 7s in targets and our softmax predictions
## our implementation of negative loss
## 1. Take softmax of the activations from previous layer
## 2. Pick activations from column index corresponding to correct prediction
## 3. Take log of the above (filtered softmax activations)
## 4. Take negative of the log of the above (filtered softmax activations if picking the activations for correct labels)
## 5. Take mean of the above

print(activations)
targets = tensor([0, 1, 0, 1, 1, 0])
sm_acts = torch.softmax(activations, dim=1)
print(sm_acts)
sm_acts = tensor([[0.6025, 0.3975],
                  [0.5021, 0.4979],
                  [0.1332, 0.8668],
                  [0.9966, 0.0034],
                  [0.5959, 0.4041],
                  [0.3661, 0.6339]])

# print(sm_acts[0])
# print(sm_acts[1])

# print(sm_acts[0][0])
# print(sm_acts[0][1])

# print(sm_acts[0, 0])
# print(sm_acts[0, 1])

# for i,t in zip(range(6), targets):
#   print(i, t, sm_acts[i][t], 1-sm_acts[i][t])

idx = range(len(targets))
# print(sm_acts[idx][0])
# print(sm_acts[idx][1])
# print(sm_acts[idx][targets])
# print(sm_acts[idx, 0])
# print(sm_acts[idx, 1])
print(sm_acts[idx, targets])

## negative loss function from pytorch
print('our nll loss', -sm_acts[idx, targets])
print('pytorch nll loss', F.nll_loss(sm_acts, targets, reduction='none'))
plot_function(torch.log, min=0, max=4)

print('our negative log mean:', -torch.log(sm_acts[idx, targets]).mean())
print('pytorch nn cross entropy: ', nn.CrossEntropyLoss()(activations, targets))
print('pytorch F cross entropy: ', F.cross_entropy(activations, targets))
# %% create a new learner and fine tune it using a very small learning rate
# learn2.lr_find??
# type(lr_find)
## The Learning Rate Finder
learn2 = vision_learner(dls, resnet34, metrics=error_rate)
# learn2.fine_tune(1, base_lr=1)
# learn2.save(file=Path(f'{BASE_DIR}/saves/oxford-iiit-pet-lr-find'))
# learn2.load(file=Path(f'{BASE_DIR}/saves/oxford-iiit-pet-lr-find'))
lr_finder = learn2.lr_find(suggest_funcs=(minimum, steep, valley, slide))
print(f'SuggestedLRs: {lr_finder}')
for f in lr_finder._fields:
  print(f'{f}: {getattr(lr_finder, f):.3e}')

# %% default lr
print(f'default learning rate in fasti: {defaults.lr}')
result = torch.logspace(0, 4, steps=5, base=5)
result
source(learn.fit)
# %%
del learn
# del learn2
del dls
# del lr_finder
torch.cuda.empty_cache()
print('cleared cuda cache')

# %% get help and source of a python function similar to np.arange? or np.arange??
a = 'hello world'
print(type(a))
print(len(a))
print(help(len))
arr1 = np.arange(6).reshape(2, 3)
arr2 = np.linspace(0, 5, 6).reshape(2, 3)
arr3 = np.random.randn(2, 3)
print(arr3.shape)

import pydoc
def source(fn):
  pydoc.pager(pydoc.render_doc(fn))

def foo(a, b): return a + b
print(source(foo))
source(np.arange)

#%% Transfer learning
# create the learner, model is frozen initially
learn = vision_learner(dls, resnet34, metrics=error_rate)
# find the optimal learning rate
lr1 = learn.lr_find(suggest_funcs=(minimum, steep))
print('lr1: ', lr1[0], lr1[1])

# train only the added layer for 3 epochs
# learn.fit_one_cycle(3, lr1[0]/10)
learn.fit_one_cycle(3, 3e-3)
print('losses: ', learn.recorder.values)

# unfreeze the model
learn.unfreeze()
# find the new optimal learning rate for training with all the layers
lr2 = learn.lr_find(suggest_funcs=(minimum, steep))
print('lr2: ', lr2[0], lr2[1])
# train the model with all the layers for 6 epochs
# learn.fit_one_cycle(6, lr_max=lr2[0]/10)
# learn.fit_one_cycle(6, lr_max=lr2[0])
# learn.fit_one_cycle(6, lr_max=lr2[0]*10) # this one seems to work best
# learn.fit_one_cycle(6, lr_max=lr2[0]*100)
# train the model with discriminative learning rates for 12 epochs
learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))
print('losses: ', learn.recorder.values)
learn.recorder.plot_loss()

# %%
from fastai.callback.fp16 import *
learn = vision_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs=3)

# %%
# check how much GPU memory is used by pytorch for models above vs when using `to_fp16`
print(f'cuda mem allocated {torch.cuda.memory_allocated():,}')
print(f'cuda memory reserved {torch.cuda.memory_reserved():,}')

# or use pynvml as per bing chat
!pip install pynvml
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f'total: {info.total:,}')
print(f'free: {info.free:,}')
print(f'used: {info.used:,}')
