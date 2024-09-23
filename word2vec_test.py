import json
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import fasttext
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer

#%%
import time
import compress_fasttext
t0 = time.time()
ft = compress_fasttext.models.CompressedFastTextKeyedVectors.load('./models/word_embedding/cc.de.300.reduced.bin')
print("Model loaded! ({:.2f} s)".format((time.time() - t0)))

#%%
v = ft['<SOS>']
print(v)
print(type(v))

#%%
print("Loading model...")
ft = fasttext.load_model('./models/word_embedding/cc.de.300.reduced.bin')
print("Model loaded!")

#%%
print("Quantization...")
ft.quantize(input=None, qout=False, cutoff=0, retrain=False, epoch=None, lr=None,
            thread=None, verbose=None, dsub=2, qnorm=False)
print("Quantized")
ft.save_model('./models/word_embedding/quantized.de.300.bin')
print("Saved")

#%%
import compress_fasttext
import gensim
from gensim.models.fasttext import load_facebook_model

print("Loading uncompressed model...")
big_model = load_facebook_model('./models/word_embedding/cc.de.300.bin')
print("Model loaded!")
# big_model = gensim.models.fasttext.FastTextKeyedVectors.load('./models/word_embedding/cc.de.300.bin')
# print("Compression...")
# small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
# print("Compressed")
# small_model.save('./models/word_embedding/quantized.de.300.bin')
# print("Saved")

#%%
import time
from gensim.models.fasttext import  load_facebook_vectors, load_facebook_model

print("Loading model...")
t0 = time.time()
ft = load_facebook_model('./models/word_embedding/cc.de.300.reduced.bin')
print("Model loaded! ({:.2f} s)".format((time.time() - t0) / 1000))

#%%
# print("Finding nearest neighbors...")
# nn = ft.get_nearest_neighbors('Rosemarie')
# print("Search completed!")
#
# # #%%
# for word in nn:
#     print(word)

#%%
# import json
# with open('./vocab/translations/translations.json', 'r') as f:
#     data = json.load(f)

#%%
tokenizer = get_tokenizer("moses", language='de')
tokens = tokenizer("Ich möchte diesen Satz tokenisieren lassen, bitte!")
print(tokens)


