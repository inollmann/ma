import compress_fasttext
import json
import pickle
import torch
import numpy as np

file_name = 'subset'

word_file_ending = '.json'
vector_file_ending = '.pkl'

word_folder = './vocab/words/'
vectors_folder = './vocab/vectors/'

with open(word_folder + file_name + word_file_ending, 'r') as f:
    words = json.load(f)

ft = compress_fasttext.models.CompressedFastTextKeyedVectors.load('./models/word_embedding/cc.de.300.reduced.bin')

vectors = []
for word in words:
    print(word)
    vectors.append(ft[word])
vectors = np.array(vectors)
vectors = torch.from_numpy(vectors)

with open(vectors_folder + file_name + vector_file_ending, 'wb') as f:
    pickle.dump(vectors, f)

print("Completed!")
