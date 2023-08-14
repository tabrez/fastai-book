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
print(f'''len(file[0].open().read()): {len(file0_content)},
      file[0].open().read()[:75]: {file0_content[:75]}''')
help(files[0].open().read)
files2000 = L(file.open().read() for file in files[:2000])
print(f'first 40 chars in files2000: {coll_repr(files2000[0], 40)}')

spacy = WordTokenizer()
tokenised = spacy([file0_content])
first_tknsd = first(tokenised)
print(f'spacy coll_repr(first_tknsd, 30): {coll_repr(first_tknsd, 30)}')
tokenizer = Tokenizer(spacy)
print(f'tokenizer: {coll_repr(tokenizer(file0_content), 31)}')

#%%
print(f'len(files): {len(files)}, files[0]: {files[0]}')
file0_content = files[0].open().read();
print(f'''len(file[0].open().read()): {len(file0_content)},
      file[0].open().read()[:75]: {file0_content[:75]}''')
files2000 = L(file.open().read() for file in files[:2000])
print(f'first 40 chars in files2000: {coll_repr(files2000[0], 40)}')
print(f'coll_repr(str): {coll_repr("This was a good moive", 10)}')
print(f'types: {type(files2000)}, {type(files2000[0])}, {type(files2000[0][:10])}, {type(coll_repr(files2000[0][:10]))}')
print(f'files2000[0][:10]: {files2000[0][:10]}')
print(f'coll_repr(files2000[0][:10]): {coll_repr(files2000[0])}')

#%% Sub-word tokenization
def subword(vocab_sz):
  sp = SubwordTokenizer(vocab_sz=vocab_sz)
  sp.setup(files2000)
  return sp(files2000)
print(f'len files2000[0]: {len(files2000[0])}')
print(f'len first of subword(1000): {len(first(subword(1000)))}')

print(f'first 40 chars in files2000: {files2000[0][:40]}')
print(f'subword(1000)[:40]: {first(subword(1000))[:40]}')
print(f'''first 40 tokens using subwords(1000):
      {" ".join(first(subword(1000))[:40])}''')
print(f'''first 40 tokens using subwords(200):
      {" ".join(first(subword(200))[:40])}''')

#%% Numericalization
numerify = Numericalize()
tokens200 = files2000[:200].map(tokenizer)
print(f'tokens200[0][:20]: {tokens200[0][:20]}')
numerify.setup(tokens200)
print(f'first few tokens from numerify.vocab: {coll_repr(numerify.vocab, 25)}')
print(f'indexes of tokens in file0_content: {numerify(tokens200[0])[:20]}')
print(f'type of numerify: {type(numerify(tokens200[0]))}')
print(f'len of numerify: {len(tokens200[0])}')
print(f'len of numerify: {len(numerify(tokens200[0]))}')
# why 5 indexes for one token when not passed as a list?
print(f'numerify of one token: {numerify(tokens200[0][1])}')

#%% Create batches
indexes200 = tokens200.map(numerify)
len(indexes200), len(indexes200[0]), indexes200[0][:20]
#%%
dl = LMDataLoader(indexes200)
x,y = first(dl)
print(f'x.shape: {x.shape}, y.shape: {y.shape}, len(dl): {len(dl)}')

count = 0
for lk in indexes200:
  for v in lk:
    count += 1
print(f'items in indexes200: {count}')
#  batch size(64) * items per batch(72) * number of batches(13) ~= count
# last batch may not have 72 items
print(f'items in indexes200: {64*72*13}')

print(f'x[:3, :5]: {x[:3,:5]}')
for lk in x[:3]:
  tokens = ''
  for v in lk[:5]:
    tokens += ' ' + numerify.vocab[v]
  print(f'row: {tokens + "..."}')

print(f'y[:3, :5]: {y[:3,:5]}')
for lk in y[:3]:
  tokens = ''
  for v in lk[:5]:
    tokens += ' ' + numerify.vocab[v]
  print(f'row: {tokens + "..."}')

#%% DataBlock
dls_lm = DataBlock(
  blocks=TextBlock.from_folder(path, is_lm=True),
  get_items=partial(get_text_files, folders=['train', 'test', 'unsup']),
  splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=54, seq_len=80)
dls_lm.show_batch(max_n=2)

#%% Fine-tuning the Language Model
learn = language_model_learner(
  dls_lm, AWD_LSTM, drop_mult=0.3, metrics=[accuracy, Perplexity()]).to_fp16()
learn.fit_one_cycle(1, 2e-2)
learn.save('1epoch')
# learn = learn.load('1epoch')
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)
learn.save('11epochs')
learn.save_encoder('finetuned')

# %%
