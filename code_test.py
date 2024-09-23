# # import mediapipe as mp
# import numpy as np
# import os
# import pickle
# import json
# import csv

# #%%
# # vocab_list = []
# # for idx, file in enumerate(os.listdir('vocab/vid/gebaerdenlernen')):
# #     if file.lower().endswith('.mp4'):
# #         name = file[:-4]
# #         vocab_list.append(name)
# #     else:
# #         continue
# vocab_list = [
#     "ich",
#     "du",
#     "er",
#     "sie",
#     "gehen",
#     "fahren",
#     "mögen",
#     "können",
#     "lesen",
#     "essen",
#     "haben",
#     "schlafen",
#     "trinken",
#     "Brot",
#     "Apfel",
#     "Salat",
#     "Auto",
#     "Fahrrad",
#     "Wasser",
#     "Kaffee",
#     "Milch"
# ]
# with open('vocab/words/subset.json', 'w') as outfile:
#     json.dump(vocab_list, outfile)
#
# #%%
# import compress_fasttext
# import json
# with open('vocab/words/subset.json', 'r') as f:
#     voc = json.load(f)
#
# model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('./models/word_embedding/cc.de.300.reduced.bin')
# dict = {idx: word for idx, word in enumerate(voc)}
# vecdict = {idx: model[word] for idx, word in enumerate(voc)}
#
# #%%
# print(len(dict))

#%%
# import torch
# from seq2seq_local import Seq2Seq
#
# model = Seq2Seq()
# dgsidx = model.encode_dgs("er trinkt Mineralwasser")
# test_input = model.sentence2vec("Fischers Fritz fischt frische Fische.")
# test_output = model(test_input)

#%%
# test_input = "Du isst gerne Brot."
# inp = model.sentence2vec(test_input)
# print(inp)
# test_output = model(inp)
# test_decoded_output = model.decode_sequence(test_output)
# print(test_input)
# print(test_output)
# print(test_decoded_output)
#
# #%%
# import torch
# import tensorflow as tf
# from torch.utils.tensorboard import SummaryWriter
#
# writer = SummaryWriter('./runs')
#
# #%%
# writer.add_graph(model, inp)
# writer.close()

#%%
# import re
#
# with open('./vocab/translations/translations_subset.txt', encoding='utf-8') as f:
#     data = f.read().splitlines()
# pairs = [re.split(r'\s{2,}', line) for line in data]

#%%
# from torch.utils.data import DataLoader
# from dataset import DgsDataset

# ds = DgsDataset('./vocab/translations/translations_subset.txt')
# dl = DataLoader(ds, batch_size=1, shuffle=True)

# de, dgs, de_string, dgs_string = next(iter(dl))
# print(de, de_string)
# print(dgs, dgs_string)

#%%
# import csv
#
# with open('vocab/translations/dgs_korpus.CSV') as f:
#     csv_file = csv.reader(f, delimiter=';')
#     data = {'id': [], 'de': [], 'dgs': [], 'mouth': []}
#     cur_idx, cur_de, cur_dgs, cur_mouth = None, "", "", ""
#
#     for row in csv_file:
#         if row[0] and row[0] != ' ':
#             print(row[0])
#             cur_idx = int(row[0])
#
#         if row[1] == 'Deutsch':
#             cur_de = row[2]
#         elif row[1] == 'Gloss':
#             cur_dgs = row[2]
#         elif row[1] == 'Mundbild':
#             cur_mouth = row[2]
#
#         if not any(row) and any([cur_idx, cur_de, cur_dgs, cur_mouth]):
#             data['id'].append(cur_idx)
#             data['de'].append(cur_de)
#             data['dgs'].append(cur_dgs)
#             data['mouth'].append(cur_mouth)
#             cur_idx, cur_de, cur_dgs, cur_mouth = None, "", "", ""
#
# #%%
# for id, de, dgs, mouth in zip(data['id'], data['de'], data['dgs'], data['mouth']):
#     print(id, de, dgs, mouth)
#
# import pickle
# pickle.dump(data, open('vocab/translations/dgs_korpus.pkl', 'wb'))

#%%
# from collections import Counter
# import pickle
# import itertools
# from torchtext.data import get_tokenizer
#
# with open('vocab/translations/dgs_korpus.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# sentences = [entry.split() for entry in data['dgs']]
# tokens = list(itertools.chain.from_iterable(sentences))
#
# keys = Counter(tokens).keys()
# counts = Counter(tokens).values()
# c = Counter(tokens)
#
# vocabulary = sorted(c, key=c.get, reverse=True)
# tok = get_tokenizer("moses", language='de')
#
# #%%
# print(tok(vocabulary[56]))

#%%
# import pickle
#
# with open('vocab/translations/dgs_korpus.pkl', 'rb') as f:
#     data = pickle.load(f)
# index = 2678
# test = data['de'][index]
# print(data['de'][index])
# print(data['dgs'][index])
# del data['de'][index]
# del data['dgs'][index]
# del data['id'][index]
# del data['mouth'][index]
# print(data['de'][index])
# #%%
# pickle.dump(data, open('vocab/translations/dgs_korpus.pkl', 'wb'))

#%%
from dataset import DgsDataset

ds = DgsDataset('vocab/translations/dgs_korpus.pkl')
reduced = ds.reduced_vocabulary


