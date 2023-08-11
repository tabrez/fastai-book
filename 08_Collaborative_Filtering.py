# %% Imports
import fastbook
fastbook.setup_book()
from fastbook import *

import fastai
from fastai.collab import *
from fastai.tabular.all import *
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

# %% download the data
path = untar_data(URLs.ML_100k)
print('path', path, path.ls())
# `u.data` contains the data we need as per the `README.md` of the above dataset
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user', 'movie', 'rating', 'timestamp'])
ratings.head()

#%% Creating the DataLoaders
movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
movies.head()
ratings = ratings.merge(movies)
ratings.head()
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()

#%% inspect classes in dls
print(type(dls.classes), len(dls.classes['user']), len(dls.classes['title']))
dls.classes['user'][:5], dls.classes['title'][:5]

#%% representation for movie and user latent factors
n_users = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5
user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)

