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
print(type(dls.classes), len(dls.classes['user']), dls.classes['user'][:5], len(dls.classes['title']), dls.classes['title'][:5])
dls.classes['user'][:5], dls.classes['title'][:5]

#%% representation for movie and user latent factors
n_users = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5
user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)

print(f'user_factors[:5]: {user_factors[:5]}')
print(f'movie_factors[:5]: {movie_factors[:5]}')

#%% indexing with vectors
help(one_hot)
one_hot_3 = one_hot(3, n_users).float()
# print(f'one_hot_3: {one_hot_3}')
user_factor_3 = user_factors.T @ one_hot_3
print(f'user_factor_3: {user_factor_3}, user_factors[3]: {user_factors[3]}')

#%% some indexing tricks with pd.DataFrame
df1 = pd.DataFrame(data=[1,2,3,4,5,6,7,7,7])
df1.iloc[[6,6,6,6,6,1,6]]
df1[0], df1[[0]]
batch = tensor([[0,7],
         [0,6],
         [1,7],
         [3,1]])
batch[:,0], batch[:,1]
df1.iloc[batch[:,0]], df1.iloc[[0,0,1,3]]

x,y = dls.one_batch()
print(f'x.shape: {x.shape}')

print(f'product', batch[:,0]*batch[:,1])


#%% create our custom model
class DotProduct1(Module):
  def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
    self.user_factors = Embedding(n_users, n_factors)
    self.movie_factors = Embedding(n_movies, n_factors)
    self.y_range = y_range

  def forward(self, batch):
    users = self.user_factors(batch[:,0])
    movies = self.movie_factors(batch[:,1])
    dotp = (users * movies).sum(dim=1)
    return sigmoid_range(dotp, *self.y_range)

class DotProduct2(Module):
  def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
    self.user_factors = Embedding(n_users, n_factors)
    self.movie_factors = Embedding(n_movies, n_factors)
    self.y_range = y_range
    self.user_bias = Embedding(n_users, 1)
    self.movie_bias = Embedding(n_movies, 1)

  def forward(self, batch):
    users = self.user_factors(batch[:,0])
    movies = self.movie_factors(batch[:,1])
    user_bias = self.user_bias(batch[:,0])
    movie_bias = self.movie_bias(batch[:,1])
    # print('======================================')
    # print(f'users: {users}, \nmovies: {movies},\n users*movies: {users*movies}')
    # print(f'sum: {(users*movies).sum(dim=1, keepdim=True)}, \nuser_bias: {user_bias}, \nmovie_bias: {movie_bias}')
    dotp = (users * movies).sum(dim=1, keepdim=True) + user_bias + movie_bias
    # print(f'dotp: { dotp}')
    return sigmoid_range(dotp.sum(dim=1), *self.y_range)

#%% Use pytorch tensors instead of Embedding type
def create_params(size):
  return nn.Parameter(torch.randn(*size).normal_(0, 0.01))

class DotProduct(Module):
  def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
    self.user_factors = create_params((n_users, n_factors))
    self.movie_factors = create_params((n_movies, n_factors))
    self.y_range = y_range
    self.user_bias = create_params((n_users, 1)# wrong:)
    self.movie_bias = create_params((n_movies, 1))

  def forward(self, batch):
    users = self.user_factors[batch[:,0]]
    movies = self.movie_factors[batch[:,1]]
    user_bias = self.user_bias[batch[:,0]]
    movie_bias = self.movie_bias[batch[:,1]]
    dotp = (users * movies).sum(dim=1, keepdim=True) + user_bias + movie_bias
    return sigmoid_range(dotp.sum(dim=1), *self.y_range)

model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat(), wd=0.2)
learn.fit_one_cycle(5, 5e-3)


#%% check model
len(learn.model.movie_bias), len(learn.model.movie_bias.squeeze())
learn.model.movie_bias, learn.model.movie_bias.squeeze()
idxs1 = learn.model.movie_bias.squeeze().argsort()[:5]
# wrong:
# idxs1.tolist(), movies[movies['movie'].isin(idxs1.tolist())]
idxs2 = learn.model.movie_bias.squeeze().argsort()[-5:]
dls.classes['title'][idxs1], dls.classes['title'][idxs2],

