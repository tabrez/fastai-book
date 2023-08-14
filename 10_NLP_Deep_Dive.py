# %% Imports
import pydoc
import warnings
from fastai.text.all import *
from fastbook import *
import fastai
import fastbook

fastbook.setup_book()
print(f"fastai version: {fastai.__version__}")

warnings.filterwarnings(
  "ignore",
  message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.",
)
warnings.filterwarnings(
  "ignore",
  message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet34_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet34_Weights.DEFAULT` to get the most up-to-date weights.",
)

# BASE_DIR='/content/gdrive/MyDrive/Colab Notebooks'
BASE_DIR = "/home/tabrez/MEGA/MEGA/code/fastai-book"
#!ln -sf '/content/gdrive/MyDrive/Colab Notebooks' ~/.fastai

def source(fn):
    pydoc.pager(pydoc.render_doc(fn))

#%% Summary
# path/{'train', 'test', 'unsup'} -> files -> files[0] -> txt \
# -> first(tokenizer([txt])) -> toks

#%% Word Tokenization

path = untar_data(URLs.IMDB)
print(f'path: {path}')
files = get_text_files(path, folders=['train', 'test', 'unsup'])
print(f'len(files): {len(files)}, files[0]: {files[0]}')
file0_content = files[0].open().read();
print(f'len(file0_content): {len(file0_content)}, file0_content[:75]: {file0_content[:75]}')

tokenizer = WordTokenizer()
tokenised = tokenizer([file0_content])
first_tknsd = first(tokenised)
print(f'coll_repr(first_tknsd, 30): {coll_repr(first_tknsd, 30)}')

#%% Sub-word tokenization
files100 = L(f.open.read() for f in files)
def subword(vocab_sz):
  sp = SubwordTokenizer(vocab_sz=vocab_sz)
  sp.setup(files100)
  return sp(files100)
' '.join(first(subword(1000)))[:40]

#%%
